import os
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop
from torch import nn

from transformers.utils import ModelOutput

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLMHeadModel
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)

from transformers.models.auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from transformers.models.blip_2.configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2Model, Blip2VisionModel

from .base import Blip2PreTrainedModel
from .qformer import Blip2QFormerModel, Blip2TextEmbeddings, Blip2QFormerOnlyMLMHead


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


@torch.no_grad()
def concat_all_gather(tensor, with_grad=False):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    if with_grad:
        return torch.cat(all_gather_with_backprop(tensor), dim=0)

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@dataclass
class BlipIntermediateOutput(ModelOutput):
    # uni-modal features
    image_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None

    image_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    itm_logits: Optional[torch.FloatTensor] = None
    itm_labels: Optional[torch.LongTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None


@dataclass
class Blip2QFormerModelOutput(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    intermediate_output: BlipIntermediateOutput = None
    loss: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_itg: Optional[torch.FloatTensor] = None


class Blip2ForQformerTraining(Blip2PreTrainedModel):
    main_input_name = "pixel_values"
    _keep_in_fp32_modules = []

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        self.decoder_start_token_id= config.decoder_start_token_id

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))

        self.embeddings = Blip2TextEmbeddings(config.qformer_config)
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.cls = Blip2QFormerOnlyMLMHead(config.qformer_config)

        # vision projection layer
        self.vision_projection = nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)

        # text projection layer
        self.text_projection = nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)

        # image text matching head
        self.itm_head = nn.Linear(config.qformer_config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.qformer_config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embedding = value 

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def from_qformer_pretrained(self):

        bert_config = BertConfig.from_pretrained("klue/bert-base")
        bert_config.is_decoder = True
        bert_model = BertLMHeadModel.from_pretrained("klue/bert-base", config=bert_config)
        bert_state_dict = bert_model.bert.state_dict()

        new_state_dict = {}

        new_state_dict['layernorm.weight'] = bert_state_dict['embeddings.LayerNorm.weight']
        new_state_dict['layernorm.bias'] = bert_state_dict['embeddings.LayerNorm.bias']

        for key in bert_state_dict.keys():
            # if 'embeddings' in key:
            #     continue
            
            new_key = key
            
            if 'self' in key:
                new_key = key.replace('self', 'attention')
            
            if 'intermediate' in key:
                intermediate_query_key = key.replace('intermediate', 'intermediate_query')
                new_state_dict[intermediate_query_key] = bert_state_dict[key]
                
            if 'output' in key and 'attention.output' not in key:
                output_query_key = key.replace('output', 'output_query')
                new_state_dict[output_query_key] = bert_state_dict[key]
            
            new_state_dict[new_key] = bert_state_dict[key]

        m, e = self.qformer.load_state_dict(new_state_dict, strict=False)
        self.cls.load_state_dict(bert_model.cls.state_dict())
        self.embeddings.load_state_dict(bert_model.bert.embeddings.state_dict(), strict=False)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_image_text_matching_head: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:#, Blip2ImageTextMatchingModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=pixel_values.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_embeds = self.embeddings(
            query_embeds=query_tokens,
        )
        query_outputs = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            use_cache=True,
            return_dict=True,
        )
        # query_outputs = query_outputs[0] if not return_dict else query_outputs.last_hidden_state
        image_feats = nn.functional.normalize(self.vision_projection(query_outputs.last_hidden_state), dim=-1)

        # TODO: add tokenizer 
        text_embeds = self.embeddings(
            input_ids=input_ids,
        )
        text_outputs = self.qformer(
            query_embeds=text_embeds,
            query_length=0,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = text_outputs[0] if not return_dict else text_outputs.last_hidden_state
        text_feats = nn.functional.normalize(self.text_projection(question_embeds[:, 0, :]), dim=-1)

        image_feats_all = concat_all_gather(image_feats)
        text_feats_all = concat_all_gather(text_feats)

        sim_i2t = torch.matmul(image_feats, text_feats_all.t())
        sim_i2t, _ = sim_i2t.max(dim=1)

        sim_t2i = torch.matmul(image_feats_all, text_feats.t())
        sim_t2i, _ = sim_t2i.max(dim=1)
        sim_t2i = sim_t2i.t()

        # rank = dist.get_rank()
        rank = 0
        bs = image_embeds.shape[0]
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            pixel_values.device
        )

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2


        sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
        sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

        weights_t2i = F.softmax(sim_t2i, dim=1)
        weights_i2t = F.softmax(sim_i2t, dim=1)

        image_embeds_all = concat_all_gather(image_embeds, with_grad=True)
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_all[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        text_input_ids_all = concat_all_gather(input_ids)
        text_attention_mask_all = concat_all_gather(attention_mask)
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_all[neg_idx])
            text_atts_neg.append(text_attention_mask_all[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [input_ids, input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [attention_mask, attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            query_tokens_itm.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        query_embeds_itm = self.embeddings(
            input_ids=text_ids_all,
            query_embeds=query_tokens_itm,
        )

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_attention_mask_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            pixel_values.device
        )

        itm_outputs = self.qformer(
            query_embeds=query_embeds_itm,
            query_length=query_tokens_itm.shape[1],
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_attention_mask_all,
            return_dict=True,
        )
        itm_embeds = itm_outputs[0] if not return_dict else itm_outputs.last_hidden_state
        itm_outputs = self.itm_head(itm_embeds[:, : query_tokens_itm.size(1), :])
        logits_itm = itm_outputs.mean(dim=1)
        
        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(pixel_values.device)
        loss_itm = F.cross_entropy(logits_itm, itm_labels)

        # ITG
        decoder_input_ids = input_ids.clone()
        decoder_input_ids[:, 0] = self.decoder_start_token_id

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(  # 4 32
            pixel_values.device
        )

        decoder_embeds = self.embeddings(  # 4, 64, 768
            input_ids=decoder_input_ids,
            # query_embeds=query_tokens,
        )

        attention_mask_itg = torch.cat([query_atts, attention_mask], dim=1)

        itg_outputs = self.qformer(
            decoder_embeds,
            query_length=0,
            attention_mask=attention_mask_itg,
            past_key_values=query_outputs.past_key_values,
            return_dict=True,
        )
        logits_itg = self.cls(itg_outputs.last_hidden_state)

        # labels = labels.to(logits_itg.device)
        logits_itg = logits_itg[:, -labels.size(1) :, :]
        # Shift so that tokens < n predict n
        shift_logits = logits_itg[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits_itg.device)

        loss_itg = F.cross_entropy(shift_logits.view(-1, self.config.qformer_config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (loss_itc + loss_itm + loss_itg)
            return output

        return Blip2QFormerModelOutput(
                loss=loss_itc + loss_itm + loss_itg,
                loss_itc=loss_itc,
                loss_itm=loss_itm,
                loss_itg=loss_itg,
        )
