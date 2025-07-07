from torch import nn
from transformers import Speech2TextModel, Speech2TextForConditionalGeneration
from .avhubert import AVHubertModel
from .av_transformer_decoder import AVTransformerDecoder
from .av2text_config import AV2TextConfig
import torch
from typing import Optional
from transformers.generation.utils import Cache


class AV2TextModel(Speech2TextModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AVHubertModel(config)
        self.decoder = AVTransformerDecoder(config)
        
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embed_tokens.weight

class AV2TextForConditionalGeneration(Speech2TextForConditionalGeneration):
    config_class = AV2TextConfig
    def __init__(self, config):
        super().__init__(config)        
        self.model = AV2TextModel(config)
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.decoder.embed_tokens.weight
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs
        )
        del model_inputs["video"]
        return model_inputs