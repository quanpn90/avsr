from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Attention
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
import torch
from typing import Dict, List, Optional, Tuple, Any

from transformers.utils import logging
from src.nets.backend.backbones.modules.flash_attention_utils import _flash_attention_packed_forward

logger = logging.get_logger(__name__)


# Copied from transformers.models.hubert.modeling_hubert.HubertFlashAttention2 with Hubert->Wav2Vec2
class AVHuBERTFlashAttention2(Wav2Vec2Attention):
    """
    Wav2Vec2 flash attention module. This module inherits from `Wav2Vec2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment,
        # that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1)
        # produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    def _reshape(self, tensor: torch.Tensor, total: int):
        return tensor.view(total, self.num_heads, self.head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            flash_kwargs=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        total, _ = hidden_states.size()

        # get query proj
        query_states = self._reshape(self.q_proj(hidden_states), total)

        assert is_cross_attention == False
        assert past_key_value == None
        assert self.is_decoder == False
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        # if (
        #         is_cross_attention
        #         and past_key_value is not None
        #         and past_key_value[0].shape[2] == key_value_states.shape[1]
        # ):
        #     # reuse k,v, cross_attentions
        #     key_states = past_key_value[0].transpose(1, 2)
        #     value_states = past_key_value[1].transpose(1, 2)
        # elif is_cross_attention:
        #     # cross_attentions
        #     key_states = self._reshape(self.k_proj(key_value_states), -1, bsz)
        #     value_states = self._reshape(self.v_proj(key_value_states), -1, bsz)
        # elif past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)
        #     key_states = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
        #     value_states = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
        # else:
        # self_attention
        key_states = self._reshape(self.k_proj(hidden_states), total)
        value_states = self._reshape(self.v_proj(hidden_states), total)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # totals = query_states.size(0)

        attn_output = _flash_attention_packed_forward(
            query_states,
            key_states,
            value_states,
            dropout=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            cu_seq_lens_q=flash_kwargs["cu_seq_lens_q"],
            cu_seq_lens_k=flash_kwargs["cu_seq_lens_k"],
            max_length_q=flash_kwargs["max_length_q"],
            max_length_k=flash_kwargs["max_length_k"],
            query_length=flash_kwargs["query_length"],

        )

        attn_output = attn_output.reshape(total, -1)
        attn_output = self.out_proj(attn_output)

        # if not output_attentions:
        # flash attention doesn't return attn_weights.
        attn_weights = None

        return attn_output, attn_weights, past_key_value
