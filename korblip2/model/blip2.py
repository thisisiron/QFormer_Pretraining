import os
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from transformers.utils import ModelOutput

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLMHeadModel

from transformers.models.auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from transformers.models.blip_2.configuration_blip_2 import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2Model, Blip2VisionModel

from .base import Blip2PreTrainedModel
from .qformer import Blip2QFormerModel, Blip2TextEmbeddings


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class Blip2ForQformerTraining(Blip2PreTrainedModel):
    main_input_name = "pixel_values"
    _keep_in_fp32_modules = []

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        # config.qformer_config.use_qformer_text_input = True

        # self.tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
        # self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        assert config.qformer_config.vocab_size == config.bert_config.vocab_size
        self.embeddings = Blip2TextEmbeddings(config.qformer_config)
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # vision projection layer
        self.vision_projection = nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)

        # text projection layer
        self.text_projection = nn.Linear(config.qformer_config.hidden_size, config.image_text_hidden_size)

        # image text matching head
        self.itm_head = nn.Linear(config.qformer_config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def from_pretrained_qformer(self):

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
        self.qformer.cls = bert_model.cls

        self.embeddings.load_state_dict(bert_model.bert.embeddings.state_dict(), strict=False)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
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
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )
        query_outputs = query_outputs[0] if not return_dict else query_outputs.last_hidden_state
        image_feats = nn.functional.normalize(self.vision_projection(query_outputs), dim=-1)

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

        # ITC
        logits_per_image = torch.matmul(image_feats, text_feats.t())
        logits_per_image, _ = logits_per_image.max(dim=1)
        logits_per_text = logits_per_image.t()


        if use_image_text_matching_head:
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(query_tokens.device)
            attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)

            query_embeds = self.embeddings(
                input_ids=input_ids,
                query_embeds=query_tokens,
            )

            text_outputs = self.qformer(
                query_embeds=query_embeds,
                query_length=query_tokens.shape[1],
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=return_dict,
            )
            text_embeds = text_outputs[0] if not return_dict else text_outputs.last_hidden_state

            output = self.itm_head(text_embeds[:, : query_tokens.size(1), :])
            logits_per_image = output.mean(dim=1)
            logits_per_text = logits_per_image.t()
        else:
            # ITC
            # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # query_outputs = self.qformer(
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=image_embeds,
            #     encoder_attention_mask=image_attention_mask,
            #     return_dict=return_dict,
            # )
            # image_embeds = query_outputs[0] if not return_dict else query_outputs.last_hidden_state

            # query_embeds = self.embeddings(
            #     input_ids=input_ids,
            # )
            # text_outputs = self.qformer(
            #     query_embeds=query_embeds,
            #     query_length=0,
            #     attention_mask=attention_mask,
            #     return_dict=return_dict,
            # )
            # question_embeds = text_outputs[0] if not return_dict else text_outputs.last_hidden_state

            # normalized features
            # image_embeds = nn.functional.normalize(self.vision_projection(image_embeds), dim=-1)
            # text_embeds = nn.functional.normalize(self.text_projection(question_embeds[:, 0, :]), dim=-1)

            # cosine similarity as logits
            # logits_per_image = torch.matmul(image_embeds, text_embeds.t())
            # logits_per_image, _ = logits_per_image.max(dim=1)

            # logits_per_text = logits_per_image.t()
            pass

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return output

        # return Blip2ImageTextMatchingModelOutput(
        #     logits_per_image=logits_per_image,
        #     logits_per_text=logits_per_text,
        #     text_embeds=text_embeds,
        #     image_embeds=image_embeds,
        #     text_model_output=text_outputs,
        #     vision_model_output=vision_outputs,
        # )
