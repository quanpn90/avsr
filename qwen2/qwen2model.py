# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2Audio model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict
from contextlib import nullcontext

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import auto_docstring, logging
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig, Qwen2AudioEncoderConfig
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from transformers.models.qwen2_audio.modeling_qwen2_audio import (Qwen2AudioPreTrainedModel,
                                                                  Qwen2AudioCausalLMOutputWithPast,
                                                                  Qwen2AudioForConditionalGeneration,
                                                                  Qwen2AudioMultiModalProjector)

from optimized.layer_norm import MemoryEfficientLayerNorm, fast_layer_norm_cuda
from optimized.rms_norm import EfficientQwen2RMSNorm, Qwen2RMSNorm, fast_rms_norm_cuda
from optimized.swiglu_mlp import MLPSwiGLU, copy_weight, swiglu_mlp_cuda, SwiGLU_mlp
from optimized.apex_norm import FusedRMSNorm

from qwen2.models.qwen2audio_attention import Qwen2AudioPackedFlashAttention
from qwen2.models.qwen_audio_encoder import MemoryOptimizedQwen2AudioEncoderLayer, MemoryOptimizedQwen2AudioEncoder

from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

try:
    import xentropy_cuda
    from optimized.softmax_crossentropy import SoftmaxCrossEntropyLoss

    softmax_xentropy = SoftmaxCrossEntropyLoss.apply
    fast_xentropy = True
except (ModuleNotFoundError, AttributeError):
    softmax_xentropy = None
    fast_xentropy = False

#
# if is_flash_attn_available():
#     from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


class MemoryEfficientQwen2ForCausalLM(Qwen2ForCausalLM):

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                compute_logits=False,
                **kwargs: Unpack[KwargsForCausalLM],
                ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if compute_logits:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = hidden_states

        loss = None
        if labels is not None:
            raise NotImplementedError("labels must not be given to the MemoryEfficientQwen2ForCausalLM")
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# class Qwen2AudioMultiModalProjector(nn.Module):
#     def __init__(self, config: Qwen2AudioConfig):
#         super().__init__()
#         self.linear = nn.Linear(config.audio_config.d_model, config.text_config.hidden_size, bias=True)
#
#     def forward(self, audio_features):
#         hidden_states = self.linear(audio_features)
#         return hidden_states


# class Qwen2AudioForConditionalGeneration(Qwen2AudioPreTrainedModel, GenerationMixin):
#     def __init__(self, config: Qwen2AudioConfig):
#         super().__init__(config)
#         self.audio_tower = AutoModel.from_config(config.audio_config)  # Usually a `Qwen2AudioEncoder` instance
#
#         self.multi_modal_projector = Qwen2AudioMultiModalProjector(config)
#         self.vocab_size = config.text_config.vocab_size
#         self.language_model = AutoModelForCausalLM.from_config(config.text_config)
#         if self.language_model._tied_weights_keys is not None:
#             self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]
#
#         self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
#         self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
#         self.post_init()


class MemoryEfficientQwen2AudioLM(Qwen2AudioForConditionalGeneration):
    _no_split_modules = ["Qwen2AudioPackedFlashAttention"]

    def do_freeze_audio_encoder(self):

        for param in self.audio_tower.parameters():
            param.requires_grad = False

        for param in self.multi_modal_projector.parameters():
            param.requires_grad = False

        self.freeze_audio_encoder = True;

    def __init__(self, config: Qwen2AudioConfig):
        # call init from the grandparent class
        Qwen2AudioPreTrainedModel.__init__(self, config)

        # changes AutoModel/Qwen2AudioEncoder to MemoryOptimizedQwen2AudioEncoder
        self.audio_tower = MemoryOptimizedQwen2AudioEncoder(
            config.audio_config)  # Usually a `Qwen2AudioEncoder` instance
        # self.audio_tower = AutoModel.from_config(config.audio_config)  # Usually a `Qwen2AudioEncoder` instance

        self.multi_modal_projector = Qwen2AudioMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        # changes AutoModel to MemoryEfficientQwen2ForCausalLM
        self.language_model = MemoryEfficientQwen2ForCausalLM(config.text_config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.post_init()

        # new options for continual learning / batch_ensembles learning
        self.teacher = None
        self.teacher_distillation = 0
        self.label_smoothing = 0
        self.kl_scale = 0.
        self.kl_type = "kl_div"

        self.freeze_audio_encoder = False
        self.label_smoothing = 0.0

    # @property
    # def padding_side(self):
    #     return self._padding_side
    #
    # @padding_side.setter
    # def padding_side(self, padding_side: str):
    #     if padding_side not in ["left", "right"]:
    #         raise ValueError(f"{padding_side} is not `left` or `right`.")
    #     self._padding_side = padding_side
    #
    # def get_input_embeddings(self):
    #     return self.language_model.get_input_embeddings()
    #
    # def set_input_embeddings(self, value):
    #     self.language_model.set_input_embeddings(value)
    #
    # def get_output_embeddings(self):
    #     return self.language_model.get_output_embeddings()
    #
    # def set_output_embeddings(self, new_embeddings):
    #     self.language_model.set_output_embeddings(new_embeddings)
    #
    # def set_decoder(self, decoder):
    #     self.language_model.set_decoder(decoder)
    #
    # def get_decoder(self):
    #     return self.language_model.get_decoder()

    def _merge_input_ids_with_audio_features(
            self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels
    ):
        """
        Merge input_ids with with audio features into final embeddings

        Args:
            audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = self.padding_side == "left"
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_id
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.config.audio_token_id) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, self.config.ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                                                           token_placeholder_num.sum(-1) - (
                                                           attention_mask == 0).long().sum(-1)
                                                   )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

    # @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            feature_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2AudioCausalLMOutputWithPast]:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, feature_sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        >>> model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""
        if not hasattr(self.config, "audio_token_id"):
            self.config.audio_token_id = self.config.audio_token_index

        # print("Input tokens:", input_ids.size())
        # if input_features is not None:
        #     print("Audio inputs:", input_features.size())
        #
        # if inputs_embeds is not None:
        #     print("Embed inputs:", inputs_embeds.size())

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        target_device = self.audio_tower.device

        # Audio inputs
        if input_features is not None:
            input_features = input_features.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and audios
            if input_features is not None and input_ids.shape[1] != 1:
                context = torch.no_grad() if self.freeze_audio_encoder else nullcontext()
                with context:
                    audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                        feature_attention_mask.sum(-1)
                    )
                    batch_size, _, max_mel_seq_len = input_features.shape
                    max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                    # Create a sequence tensor of shape (batch_size, max_seq_len)
                    seq_range = (
                        torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                        .unsqueeze(0)
                        .expand(batch_size, max_seq_len)
                    )
                    lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                    # Create mask
                    padding_mask = seq_range >= lengths_expand

                    if self.audio_tower._attn_implementation != "flash_attention_2":
                        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                            batch_size, 1, max_seq_len, max_seq_len
                        )
                        audio_attention_mask = audio_attention_mask_.to(
                            dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
                        )
                        audio_attention_mask[audio_attention_mask_] = float("-inf")
                    else:
                        audio_attention_mask = (1 - padding_mask.float()).bool()

                    # TODO: add no_grad to this part if the audio_tower is frozen.

                    audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
                    selected_audio_feature = audio_outputs.last_hidden_state
                    audio_features = self.multi_modal_projector(selected_audio_feature)

                    # if we have consecutive audio tokens, then it means we expanded input_ids in processing
                    audio_tokens = input_ids == self.config.audio_token_id
                    legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0

                    if legacy_processing:
                        logger.warning_once(
                            "Expanding inputs for audio tokens in Qwen2Audio should be done in processing."
                        )
                        inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                            audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
                        )
                    else:
                        num_audios, max_audio_tokens, embed_dim = audio_features.shape
                        audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None,
                                              :]
                        audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                        audio_features = audio_features[audio_features_mask]

                        n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
                        n_audio_features = audio_features.shape[0]

                        if n_audio_tokens != n_audio_features:
                            raise ValueError(
                                f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                            )
                        special_audio_mask = (input_ids == self.config.audio_token_id).to(inputs_embeds.device)
                        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        # After replacing the <AUDIO> with audio encoded features

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # TODO : get the output hidden states from language model only, not the states
        logits = outputs[0]

        loss = None
        ce_loss = 0
        additional_losses = {}

        # TODO : get the 
        if labels is not None:

            full_length = logits.size(1)

            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

            mask = shift_labels != -100
            shift_logits = shift_logits[mask]
            shift_labels = shift_labels[mask]

            label_smoothing = self.label_smoothing if self.training else 0.0

            # do hidden -> logits after filtering the items to save memory
            shift_logits = self.language_model.lm_head(shift_logits)
            if fast_xentropy:
                half_to_float = (shift_logits.dtype == torch.float16) or (shift_logits.dtype == torch.bfloat16)

                # Flatten the tokens
                _shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                _shift_labels = shift_labels.view(-1).to(shift_logits.device)

                total = _shift_logits.size(0)
                ce_loss = softmax_xentropy(_shift_logits, _shift_labels,
                                           label_smoothing,  # smoothing
                                           -100,  # padding
                                           half_to_float)

                ce_loss = ce_loss.sum().div(total)

            else:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

                ce_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
                )

            if self.kl_scale > 0 and self.training:
                kl_loss = self._compute_kl()
                kl_loss = kl_loss.to(dtype=ce_loss.dtype)
                kl_loss_data = kl_loss.item()

                loss = ce_loss + kl_loss * self.kl_scale

            else:
                kl_loss, kl_loss_data = 0, 0
                loss = ce_loss

            ce_loss_data = ce_loss.item()
            additional_losses["ce_loss"] = ce_loss_data
            if self.kl_scale > 0.0:
                additional_losses["kl_loss"] = kl_loss_data

        else:
            logits = self.language_model.lm_head(logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return BayesianQwen2AudioCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            additional_losses=additional_losses
        )

    def _compute_kl(self) -> torch.Tensor:
        """
        Computes the total KL divergence from all Bayesian modules in self.model.
        Assumes each Bayesian module implements a .kl_loss() method returning a scalar.
        Returns:
            total_kl (torch.Tensor): The summed KL loss.
        """
        # If you need it on the same device as the model,
        # we can create a 0.0 tensor on the correct device:
        device = next(self.language_model.parameters()).device
        total_kl = torch.tensor(0.0, device=device)

        # Traverse all submodules in self.model
        for name, module in self.audio_tower.named_modules():
            # Check if the module has a .kl_loss() method
            if hasattr(module, "regularization_loss") and callable(module.kl_loss):
                # Accumulate
                total_kl += module.regularization_loss(type=self.kl_type)

        for name, module in self.language_model.named_modules():
            # Check if the module has a .kl_loss() method
            if hasattr(module, "regularization_loss") and callable(module.kl_loss):
                # Accumulate
                total_kl += module.regularization_loss(type=self.kl_type)
        # print(f"{total_kl.shape}, {total_kl}"+"=="*30)
        return total_kl


@dataclass
class BayesianQwen2AudioCausalLMOutputWithPast(Qwen2AudioCausalLMOutputWithPast):
    additional_losses: Optional[Dict[str, torch.FloatTensor]] = None


def create_qwen2audio_model(model_name,
                            config,
                            torch_dtype,
                            trust_remote_code=True,
                            attn_implementation="flash_attention_2",
                            low_cpu_mem_usage=False,
                            device_map="none",
                            mem_efficient=True):
    # here model_name can be either the huggingface path, or the local path
    print("[INFO] Creating Qwen2Audio model from %s " % model_name)

    def replace_audio_encoder(model):
        config = model.audio_tower.config

        old_encoder = model.audio_tower

        new_encoder = MemoryOptimizedQwen2AudioEncoder(config)

        # Copy weights from the old layer to the new one
        new_encoder.load_state_dict(old_encoder.state_dict())

        dtype = next(old_encoder.parameters()).dtype

        new_encoder.to(dtype)

        model.audio_tower = new_encoder

        # for i in range(len(model.audio_tower.layers)):
        #     old_layer = model.audio_tower.layers[i]
        #     new_layer = MemoryOptimizedQwen2AudioEncoderLayer(config)
        #
        #     # Copy weights from the old layer to the new one
        #     new_layer.load_state_dict(old_layer.state_dict())
        #
        #     dtype = next(old_layer.parameters()).dtype
        #     new_layer.to(dtype)
        #
        #     # Replace the layer in the encoder
        #     model.audio_tower.layers[i] = new_layer

    def replace_layernorm_with_memory_efficient(model):
        for name, module in model.named_children():
            # Check if the current module is LayerNorm
            if isinstance(module, torch.nn.LayerNorm):

                custom_layer = MemoryEfficientLayerNorm(module.normalized_shape, module.eps)

                # Copy weights and biases
                custom_layer.weight.data.copy_(module.weight.data)
                custom_layer.bias.data.copy_(module.bias.data)

                # convert to the right type
                custom_layer.to(module.weight.dtype)
                # Replace with MemoryEfficientLayerNorm
                setattr(model, name, custom_layer)
            else:
                # Recursively apply to submodules
                replace_layernorm_with_memory_efficient(module)

    def replace_rmsnorm_with_memory_efficient(model):
        for name, module in model.named_children():
            # Check if the current module is LayerNorm
            if isinstance(module, Qwen2RMSNorm):

                # custom_layer = EfficientQwen2RMSNorm(module.weight.numel(), module.variance_epsilon)
                custom_layer = FusedRMSNorm(module.weight.numel(), eps=module.variance_epsilon, memory_efficient=True)

                # Copy weights and biases
                custom_layer.weight.data.copy_(module.weight.data)

                # convert to the right type
                custom_layer.to(module.weight.dtype)
                # Replace with MemoryEfficientLayerNorm
                setattr(model, name, custom_layer)
            else:
                # Recursively apply to submodules
                replace_rmsnorm_with_memory_efficient(module)

    def replace_mlp_with_memory_efficient(model):
        for name, module in model.named_children():
            # Check if the current module is LayerNorm
            if isinstance(module, Qwen2MLP):

                # custom_layer = MLPSwiGLU(module.hidden_size, module.intermediate_size)
                custom_layer = SwiGLU_mlp(module.hidden_size, module.intermediate_size)

                # Copy weights
                copy_weight(module, custom_layer)

                # convert to the right type
                custom_layer.to(module.gate_proj.weight.dtype)
                setattr(model, name, custom_layer)
            else:
                # Recursively apply to submodules
                replace_mlp_with_memory_efficient(module)

    model_class = MemoryEfficientQwen2AudioLM if mem_efficient else Qwen2AudioForConditionalGeneration

    if device_map != "none":
        model = model_class.from_pretrained(model_name,
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=low_cpu_mem_usage,
                                            torch_dtype=torch_dtype,
                                            attn_implementation=attn_implementation,
                                            device_map=device_map,
                                            )
    else:
        model = model_class.from_pretrained(model_name,
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=low_cpu_mem_usage,
                                            torch_dtype=torch_dtype,
                                            attn_implementation=attn_implementation
                                            )
    if mem_efficient:
        replace_audio_encoder(model)
        #     replace_layer_with_weights(model, model.config)
        if fast_layer_norm_cuda:
            replace_layernorm_with_memory_efficient(model)

        if fast_rms_norm_cuda is not None:
            replace_rmsnorm_with_memory_efficient(model)

        if swiglu_mlp_cuda is not None:
            replace_mlp_with_memory_efficient(model)

    return model


def set_attention_dropout(model, dropout_prob):
    for name, module in model.named_modules():
        # Qwen2 uses Qwen2Attention as the class name for attention layers
        if module.__class__.__name__ == "Qwen2Attention":
            # Replace the Dropout module
            # module.dropout = nn.Dropout(dropout_prob)
            module.attention_dropout = dropout_prob
            # If you want, print to confirm
            # print(f"Set {name}.dropout.p to {dropout_prob}")
