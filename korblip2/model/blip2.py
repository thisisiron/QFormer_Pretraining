from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop

from transformers.utils import (
    logging,
    ModelOutput
)
from transformers.activations import ACT2FN
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLMHeadModel
from transformers.models.blip_2.configuration_blip_2 import (
    Blip2Config,
    Blip2QFormerConfig,
    Blip2VisionConfig
)
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2VisionModel,
    Blip2PreTrainedModel,
    Blip2QFormerEncoder,
    Blip2TextEmbeddings
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


# Initialize logger
logger = logging.get_logger(__name__)


# Utility functions
def is_dist_avail_and_initialized() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
    """Gather tensors from all processes and concat them."""
    if not is_dist_avail_and_initialized():
        return tensor
        
    if with_grad:
        return torch.cat(all_gather_with_backprop(tensor), dim=0)

    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


# Model output classes 
@dataclass
class BlipIntermediateOutput(ModelOutput):
    """Intermediate outputs for BLIP model."""
    image_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None
    image_embeds_m: Optional[torch.FloatTensor] = None 
    text_embeds_m: Optional[torch.FloatTensor] = None
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    itm_logits: Optional[torch.FloatTensor] = None
    itm_labels: Optional[torch.LongTensor] = None
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None


@dataclass
class Blip2QFormerModelOutput(ModelOutput):
    """Outputs of the QFormer model."""
    intermediate_output: BlipIntermediateOutput = None
    loss: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_itg: Optional[torch.FloatTensor] = None


class Blip2QFormerModel(Blip2PreTrainedModel):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: Blip2QFormerConfig):
        super().__init__(config)
        self.config = config

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = Blip2QFormerEncoder(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_embeds: torch.FloatTensor,
        query_length: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        query_length = (
            query_length if query_length is not None else query_embeds.shape[1] if query_embeds is not None else 0
        )

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - query_length if past_key_values is not None else 0
        )

        embedding_output = self.layernorm(query_embeds)
        embedding_output = self.dropout(embedding_output)

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if isinstance(encoder_attention_mask, list):
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class Blip2QFormerPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Blip2QFormerLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = Blip2QFormerPredictionHeadTransform(config)

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


class Blip2QFormerOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = Blip2QFormerLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class Blip2ForQformerTraining(Blip2PreTrainedModel):
    main_input_name = "pixel_values"
    _keep_in_fp32_modules = []
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]
    
    def __init__(self, config: Blip2Config):
        super().__init__(config)
        self.decoder_start_token_id = config.decoder_start_token_id

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=config.qformer_config.initializer_range)

        config.qformer_config.use_qformer_text_input = True
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
        self.embeddings.word_embeddings = value 

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias
    
    def from_qformer_pretrained(self, qformer_model_name_or_path: str):

        bert_config = BertConfig.from_pretrained(qformer_model_name_or_path)
        bert_config.is_decoder = True
        bert_model = BertLMHeadModel.from_pretrained(qformer_model_name_or_path, config=bert_config)
        bert_state_dict = bert_model.bert.state_dict()
    
        new_state_dict = {}

        new_state_dict['layernorm.weight'] = bert_state_dict['embeddings.LayerNorm.weight']
        new_state_dict['layernorm.bias'] = bert_state_dict['embeddings.LayerNorm.bias']

        for key in bert_state_dict.keys():
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
        # self.cls.load_state_dict(bert_model.cls.state_dict())
        embeddings_dict = {}
        embeddings_dict['word_embeddings.weight'] = bert_state_dict['embeddings.word_embeddings.weight']
        embeddings_dict['position_embeddings.weight'] = bert_state_dict['embeddings.position_embeddings.weight']
        m, e = self.embeddings.load_state_dict(embeddings_dict, strict=False)
        # print(m, e)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_image_text_matching_head: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_loss: Optional[bool] = None,
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
        
        loss = None
        if return_loss:
            loss = loss_itc + loss_itm + loss_itg
        
        if not return_dict:
            output = (loss_itc + loss_itm + loss_itg)
            return output

        return Blip2QFormerModelOutput(
                loss=loss,
                loss_itc=loss_itc,
                loss_itm=loss_itm,
                loss_itg=loss_itg,
        )