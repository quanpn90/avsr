import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Callable
from flash_attn import flash_attn_func

from transformers.cache_utils import Cache
from transformers import Qwen2AudioForConditionalGeneration, GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from contextlib import nullcontext

from qwen2.qwen2model import (MemoryEfficientQwen2AudioLM, MemoryOptimizedQwen2AudioEncoder,
                              Qwen2AudioConfig, Qwen2AudioCausalLMOutputWithPast,
                              BayesianQwen2AudioCausalLMOutputWithPast,
                              fast_xentropy, softmax_xentropy)
from src.nets.backend.backbones.avhubert import AVHubertModel
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from src.nets.backend.nets_utils import make_non_pad_mask

# from llama3.llama3avconfig import LLama3AVConfig
from alva.alva_config import AlvaConfig
from transformers import AutoModel, PreTrainedModel, LlamaForCausalLM, SiglipVisionModel
from transformers.models.whisper.modeling_whisper import WhisperModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP

from transformers.utils import logging
from transformers.integrations.flash_attention import flash_attention_forward

from qwen2.qwen2model import fast_layer_norm_cuda, fast_rms_norm_cuda, fast_xentropy, swiglu_mlp_cuda
from optimized.apex_norm import FusedRMSNorm
from optimized.layer_norm import MemoryEfficientLayerNorm
from optimized.swiglu_mlp import SwiGLU_mlp, copy_weight

from src.nets.backend.ctc import CTC

logger = logging.get_logger(__name__)


class AlvaPreTrainedModel(PreTrainedModel):
    config_class = AlvaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    freeze_audio_encoder = False
    freeze_video_encoder = False
    label_smoothing = 0.0

    def _init_weights(self, module):
        # important: this ported version of Qwen2Audio isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.audio_config.initializer_range
        )

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class AlvaCrossModalAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AlvaConfig):
        super().__init__()
        self.config = config
        self.audio_dim = config.audio_config.d_model
        self.video_dim = config.video_config.hidden_size
        self.embed_dim = config.avhubert_config.encoder_embed_dim
        self.num_heads = config.avhubert_config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = 0.0
        self.is_causal = False

        self.k_proj = nn.Linear(self.video_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.video_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.audio_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
            self,
            audio_hidden_states: torch.Tensor,
            image_patches: torch.Tensor,
            # attention_mask: Optional[torch.Tensor] = None,
            # output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input shape: Batch x Time x Channel"""

        batch_time, num_patches, image_dim = image_patches.shape
        bsz, seq_length, audio_dim = audio_hidden_states.shape

        assert batch_time == (bsz * seq_length)

        queries = self.q_proj(audio_hidden_states)
        keys = self.k_proj(image_patches)
        values = self.v_proj(image_patches)

        queries = queries.view(batch_time, 1, self.num_heads, self.head_dim)
        keys = keys.view(batch_time, num_patches, self.num_heads, self.head_dim)
        values = values.view(batch_time, num_patches, self.num_heads, self.head_dim)

        # TODO: allow for eager or sdpa self-attention later if necessary
        # attention_interface = flash_attention_forward
        # : Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and output_attentions:
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output = flash_attn_func(
            queries, keys, values, self.dropout, softmax_scale=self.scale, causal=False
        )

        # attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        # print(queries.size(), keys.size(), values.size())
        #
        # attn_output, attn_weights = attention_interface(
        #     self,
        #     queries,
        #     keys,
        #     values,
        #     attention_mask,
        #     is_causal=self.is_causal,
        #     scaling=self.scale,
        #     dropout=0.0 if not self.training else self.dropout,
        # )

        attn_output = attn_output.reshape(batch_time, 1, self.embed_dim).contiguous()
        attn_output = attn_output.view(batch_time, self.embed_dim).view(bsz, seq_length, self.embed_dim).contiguous()

        attn_output = self.out_proj(attn_output)

        # if not output_attentions:
        #     attn_weights = None

        audio_output = queries.view(bsz, seq_length, self.embed_dim)
        video_output = attn_output.view(bsz, seq_length, self.embed_dim)

        return audio_output, video_output


class AlvaAVProjector(nn.Module):
    def __init__(self, config: AlvaConfig):
        super().__init__()
        self.linear = nn.Linear(config.avhubert_config.encoder_embed_dim,
                                config.text_config.hidden_size, bias=False)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


class AlvaLM(AlvaPreTrainedModel, GenerationMixin):
    config_class = AlvaConfig
    whisper_encoder_mask_prob = 0.0
    avhubert_encoder_mask_prob = 0.0

    def __init__(self, config: AlvaConfig,
                 attn_implementation="flash_attention_2",
                 load_pretrained=False):
        super().__init__(config)
        self.config = config
        self.config._attn_implementation = attn_implementation
        self.config.video_config._attn_implementation = attn_implementation
        self.config.audio_config._attn_implementation = attn_implementation
        self.config.text_config._attn_implementation = attn_implementation
        self.config.avhubert_config._attn_implementation = attn_implementation

        # Frame encoder
        self.vision_tower = SiglipVisionModel(config.video_config)
        # self.vision_projector = AlvaVideoProjector(config)

        # Whisper
        self.audio_tower = MemoryOptimizedQwen2AudioEncoder(config.audio_config)
        # self.audio_projector = AlvaAudioProjector(config)

        self.cross_modal_projector = AlvaCrossModalAttention(config)

        # Llama Config
        self.vocab_size = config.text_config.vocab_size
        self.language_model = LlamaForCausalLM(config.text_config)
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        config.avhubert_config.audio_dropout = 1.0
        config.avhubert_config.modality_dropout = config.avhubert_audio_mask_prob

        # hardcode?
        config.avhubert_config._attn_implementation = "flash_attention_2"
        self.avhubert_encoder = AVHubertModel(config.avhubert_config)
        self.av_projector = AlvaAVProjector(config)

        if config.ctc_loss_alpha > 0:

            args = config.avhubert_config

            self.ctc = CTC(
                args.odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.post_init()

        if load_pretrained:
            self.load_pretrained_avhubert()
            self.load_pretrained_qwen2audio()
            self.load_pretrained_llama()
            self.load_pretrained_llava_one()

        # new options for continual learning / batch_ensembles learning
        # self.teacher = None
        # self.teacher_distillation = 0
        # self.label_smoothing = 0
        # self.kl_scale = 0.
        # self.kl_type = "kl_div"

    def load_pretrained_avhubert(self,
                                 encoder_pretrained_checkpoint="nguyenvulebinh/avhubert_encoder_large_noise_pt_noise_ft_433h",
                                 cache_dir="model-bin/cache"):
        print("Loading pretrained encoder from", encoder_pretrained_checkpoint)
        encoder_pretrained = self.avhubert_encoder.from_pretrained(
            encoder_pretrained_checkpoint,
            cache_dir=cache_dir,
        )

        self.avhubert_encoder.load_state_dict(encoder_pretrained.state_dict())

        print("[INFO] Loaded pretrained weights for AV-HuBERT")

    def load_pretrained_qwen2audio(self,
                                   model_name="openai/whisper-large-v3",
                                   cache_dir="model-bin/cache"):

        pretrained_whisper = WhisperModel.from_pretrained(model_name, cache_dir=cache_dir)

        self.audio_tower.load_state_dict(pretrained_whisper.encoder.state_dict())

        print("[INFO] Loaded pretrained weights for Whisper Audio Tower")

    def load_pretrained_llama(self,
                              model_name="llama-extended",
                              cache_dir="model-bin/cache"):

        pretrained_llama = LlamaForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

        self.language_model.load_state_dict(pretrained_llama.state_dict())

        print("[INFO] Loaded pretrained weights for LLAMA3 language model")

    def load_pretrained_llava_one(self,
                                  model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                  cache_dir="model-bin/cache"):

        from transformers import LlavaOnevisionForConditionalGeneration
        llm = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                                     cache_dir=cache_dir,
                                                                     torch_dtype=torch.bfloat16,
                                                                     device_map="auto"
                                                                     )

        self.vision_tower.load_state_dict(llm.model.vision_tower.state_dict())
        print("[INFO] Loaded pretrained weights for LLAMA-ONE language model")

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def set_audio_mask(self, args):

        self.config.whisper_encoder_mask_prob = args.whisper_encoder_mask_prob
        self.config.avhubert_audio_mask_prob = args.avhubert_audio_mask_prob

        # set
        self.config.avhubert_config.audio_dropout = 1.0
        self.config.avhubert_config.modality_dropout = self.config.avhubert_audio_mask_prob
        self.avhubert_encoder.modality_dropout = self.config.avhubert_config.modality_dropout

        self.whisper_encoder_mask_prob = args.whisper_encoder_mask_prob
        self.avhubert_encoder_mask_prob = args.avhubert_audio_mask_prob

    def forward_lm(self, attention_mask, position_ids, past_key_values,
                   inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):

        # outputs = self.language_model(
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.language_model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=None
        )

        hidden_states = outputs.last_hidden_state

        return CausalLMOutputWithPast(
            loss=None,
            logits=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_features: Optional[torch.FloatTensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            feature_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            ctc_labels: Optional[torch.LongTensor] = None,
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

        # if self.training:
        #     print("[INPUTS]", input_ids)

        if not hasattr(self.config, "audio_token_id"):
            self.config.audio_token_id = self.config.audio_token_index

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

        ctc_av_features = None
        video_lengths = None

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

                    # print("[INFO] Audio Tower processing finished")

                    audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
                    selected_audio_feature = audio_outputs.last_hidden_state
                    audio_features = selected_audio_feature

                    # if we have consecutive audio tokens, then it means we expanded input_ids in processing
                    audio_tokens = input_ids == self.config.audio_token_id
                    legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0

                    if legacy_processing:
                        raise NotImplementedError("legacy processing is not supported")

                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
                    audio_features_mask = audio_features_mask < audio_output_lengths[:, None]

                    # cut off the padded positions to prepare for  AVHuBERT
                    max_length_without_pad = max(audio_output_lengths.tolist())
                    audio_features = audio_features[:, :max_length_without_pad, :]
                    audio_features_mask = audio_features_mask[:, :max_length_without_pad]

                    # TODO: when we also have pixels, we
                    if pixel_values_videos is not None:
                        B, T, C, H, W = pixel_values_videos.shape

                        pixel_values_videos = pixel_values_videos.view(B * T, C, H, W)

                        # TODO: maybe apply lora to vision_tower
                        with torch.no_grad():
                            video_outputs = self.vision_tower(pixel_values_videos)
                            video_features = video_outputs.last_hidden_state

                        # print("[INFO] Video Tower processing finished")

                        audio_features, video_features = self.cross_modal_projector(audio_features, video_features)

                        # print("[INFO] Cross-Modality projection finished")

                    else:
                        audio_features = self.cross_modal_projector.q_proj(audio_features)
                        video_features = torch.zeros_like(audio_features)

                    av_features = self.avhubert_encoder.forward_av_features(audio_features,
                                                                            video_features,
                                                                            audio_features_mask)

                    # print("[INFO] AV-HuBERT Alignment processing finished")

                    ctc_av_features = av_features
                    video_lengths = audio_features_mask.long().sum(dim=1)

                    av_features = av_features[audio_features_mask]
                    n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
                    n_audio_features = av_features.shape[0]

                    if n_audio_tokens != n_audio_features:
                        raise ValueError(
                            f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                        )
                    special_audio_mask = (input_ids == self.config.audio_token_id).to(inputs_embeds.device)
                    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    av_features = av_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    av_features = self.av_projector(av_features)

                    # print(inputs_embeds.size(), av_features.size(), special_audio_mask.size())

                    # randomly mask out the features
                    # av_features = torch.nn.functional.dropout(av_features, self.whisper_encoder_mask_prob,
                    #                                              training=self.training)
                    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, av_features)

                    # print("[INFO] AV Embedding processing finished")

        # After replacing the <AUDIO> and <VIDEO> with audio encoded features
        outputs = self.forward_lm(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print("[INFO] LLM processing finished")

        logits = outputs[0]

        loss = None
        ce_loss = 0
        additional_losses = {}

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

            # if self.kl_scale > 0 and self.training:
            #     kl_loss = self._compute_kl()
            #     kl_loss = kl_loss.to(dtype=ce_loss.dtype)
            #     kl_loss_data = kl_loss.item()
            #
            #     loss = ce_loss + kl_loss * self.kl_scale
            #
            # else:
            # kl_loss, kl_loss_data = 0, 0
            # loss = ce_loss
            if self.config.ctc_loss_alpha > 0.0:
                alpha = self.config.ctc_loss_alpha
                loss_ctc, ys_hat = self.ctc(ctc_av_features, video_lengths, ctc_labels)
                loss = (1 - alpha) * ce_loss + alpha * loss_ctc
                additional_losses["ctc_loss"] = loss_ctc.item()
            else:
                loss_ctc = None
                loss = ce_loss

            ce_loss_data = ce_loss.item()
            additional_losses["ce_loss"] = ce_loss_data

            # if self.kl_scale > 0.0:
            #     additional_losses["kl_loss"] = kl_loss_data

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


def create_alva_model(load_path,
                      config,
                      torch_dtype,
                      trust_remote_code=True,
                      attn_implementation="flash_attention_2",
                      low_cpu_mem_usage=False,
                      device_map="none",
                      mem_efficient=True,
                      create_if_missing=False):
    # here model_name can be either the huggingface path, or the local path
    print("[INFO] Creating ALVA model from %s" % load_path)

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
            if isinstance(module, LlamaRMSNorm):

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
            if isinstance(module, LlamaMLP):

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

    model_class = AlvaLM

    # load = len(model_name > 0)

    if load_path and os.path.exists(load_path):

        if device_map != "none":
            model = model_class.from_pretrained(load_path,
                                                trust_remote_code=True,
                                                low_cpu_mem_usage=low_cpu_mem_usage,
                                                torch_dtype=torch_dtype,
                                                attn_implementation=attn_implementation,
                                                device_map=device_map)
        else:
            model = model_class.from_pretrained(load_path,
                                                trust_remote_code=True,
                                                low_cpu_mem_usage=low_cpu_mem_usage,
                                                torch_dtype=torch_dtype,
                                                attn_implementation=attn_implementation)
    else:
        print("No checkpoint found, creating new model from config.")
        model = model_class(config)
        model = model.to(torch_dtype)
        # Optionally save as prototype if in training mode and path is given
        if create_if_missing and load_path:
            print(f"Saving prototype model to {load_path}")
            model.save_pretrained(load_path)

    if mem_efficient:
        if fast_layer_norm_cuda:
            replace_layernorm_with_memory_efficient(model)

        if fast_rms_norm_cuda is not None:
            replace_rmsnorm_with_memory_efficient(model)

        if swiglu_mlp_cuda is not None:
            replace_mlp_with_memory_efficient(model)

    return model


AutoModel.register(AlvaConfig, AlvaLM)

if __name__ == "__main__":
    cache_dir = "/hkfs/work/workspace/scratch/pj5699-bayesian_speechlm/text2asr/data-bin/cache"
    import datasets
    from alva.alva_processor import AlvaProcessor
    import torch
    from transformers import (AutoTokenizer, AutoProcessor, LlavaOnevisionProcessor,
                              AutoConfig, LlavaOnevisionForConditionalGeneration)

    from torch.profiler import profile, ProfilerActivity

    from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead

    from src.dataset.avhubert_dataset import load_video_rgb, load_audio, cut_or_pad

    config = AlvaConfig.from_pretrained("alva-base")

    alva_model = AlvaLM(config, attn_implementation="flash_attention_2",
                        load_pretrained=True)

    # print(alva_model)
    alva_model = alva_model.cuda().to(torch.bfloat16)
    # alva_model.load_

    processor = AlvaProcessor.from_pretrained("alva-base")

    # vision_tower = SiglipVisionModel(config.video_config)
    #
    # config.video_config._attn_implementation = "flash_attention_2"
    # pooler = AlvaCrossModalAttention(config)
    #
    # vision_tower = vision_tower.cuda().to(torch.bfloat16)
    #
    # config.audio_config._attn_implementation = "flash_attention_2"
    # audio_tower = MemoryOptimizedQwen2AudioEncoder(config.audio_config)
    #
    # audio_tower = audio_tower.cuda().to(torch.bfloat16)
    # pooler = pooler.cuda().to(torch.bfloat16)
    #
    # print("AUDIO TOWER")
    # print(audio_tower)
    #

    #
    # create data

    test_dataset = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=True,
                                         cache_dir=cache_dir).remove_columns(['__key__', '__url__'])['train']

    iterator = iter(test_dataset)

    video_rgbs = []
    texts = []
    audios = []

    prompt_template = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech:"
    rate_ratio = 640

    for i in range(2):
        sample = next(iterator)
        # print(sample.keys())
        video_array = sample['video']
        video_rgb = load_video_rgb(video_array)
        audio = load_audio(video_array)

        audio = cut_or_pad(audio, len(video_rgb) * rate_ratio).squeeze().numpy()

        video_rgbs.append(video_rgb)
        audios.append(audio)

        text = sample["label"].decode(encoding="utf-8")
        text = prompt_template + "<|begin_of_text|>" + text.lower() + processor.tokenizer.eos_token

        texts.append(text)

    inputs = processor(texts, audios, video_rgbs, return_tensors="pt", padding=True)

    for key in inputs:

        inputs[key] = inputs[key].cuda()
        if inputs[key].dtype == torch.float32:  # torch.float is an alias for torch.float32
            inputs[key] = inputs[key].to(torch.bfloat16)

        print(key, inputs[key].shape, inputs[key].dtype)

    # input_ids: Optional[torch.LongTensor] = None,
    # input_features: Optional[torch.FloatTensor] = None,
    # pixel_values_videos: Optional[torch.FloatTensor] = None,
    # attention_mask: Optional[torch.Tensor] = None,
    # feature_attention_mask: Optional[torch.Tensor] = None,
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with torch.inference_mode():
            llm_output = alva_model(input_ids=inputs["input_ids"],
                                    input_features=inputs["input_features"],
                                    pixel_values_videos=inputs["pixel_values_videos"],
                                    attention_mask=inputs["attention_mask"],
                                    feature_attention_mask=inputs["feature_attention_mask"])

    names = [e.key for e in prof.key_averages()]
    hits = [n for n in names if "flash_attn" in n.lower()]
    print("ALVA Profiler hits:", hits)

    print(llm_output)

    print("ALVA created and run successfully!")

    alva_model.save_pretrained("alva-base")

    print("ALVA saved successfully!")

    #
    # videos = inputs["pixel_values_videos"]
    #
    # B, T, C, H, W = videos.shape
    #
    # videos = videos.view(B * T, C, H, W)
    #
    # # with torch.no_grad():
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    #     with torch.inference_mode():
    #         video_outputs = vision_tower(videos)
    #
    # names = [e.key for e in prof.key_averages()]
    # hits = [n for n in names if "flash_attn" in n.lower()]
    # print("VIDEO Profiler hits:", hits)
    #
    # video_features = video_outputs.last_hidden_state
    #
    # print(video_features.size())
    #
    # # pooled_outputs = pooler(last_hidden_state)
    #
    # # print(pooled_outputs.size())
    #
    # audio_inputs = inputs["input_features"]
    # audio_mask = inputs["feature_attention_mask"]
    # #
    # # print(audio_inputs.size())
    # # print(audio_mask.size())
    #
    # # print(audio_mask)
    #
    # audio_feat_lengths, audio_output_lengths = audio_tower._get_feat_extract_output_lengths(
    #     audio_mask.sum(-1)
    # )
    #
    # batch_size, _, max_mel_seq_len = audio_inputs.shape
    # max_seq_len = (max_mel_seq_len - 2) // 2 + 1
    # # Create a sequence tensor of shape (batch_size, max_seq_len)
    # seq_range = (
    #     torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
    #     .unsqueeze(0)
    #     .expand(batch_size, max_seq_len)
    # )
    # lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
    # # Create mask
    # padding_mask = seq_range >= lengths_expand
    #
    # audio_attention_mask = (1 - padding_mask.float()).bool()
    #
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    #     with torch.inference_mode():
    #         audio_outputs = audio_tower(audio_inputs,
    #                                     audio_attention_mask).last_hidden_state
    # names = [e.key for e in prof.key_averages()]
    # hits = [n for n in names if "flash_attn" in n.lower()]
    # print("AUDIO Profiler hits:", hits)
    # # print(audio_outputs.size())
    # # print(audio_attention_mask.size())
    #
    # num_audios, max_audio_tokens, embed_dim = audio_outputs.shape
    # audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None,
    #                       :]
    # audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
    #
    # # print(audio_features_mask.size())
    # # print(audio_features_mask)
    #
    # max_length_without_pad = max(audio_output_lengths.tolist())
    # audio_features = audio_outputs[:, :max_length_without_pad, :]
    # audio_features_mask = audio_features_mask[:, :max_length_without_pad]
    #
    # print(audio_features.size())
    # print(audio_features_mask.size())
    #
    # # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    # with torch.inference_mode():
    #     audio_embeds, video_embeds = pooler(audio_features, video_features)
    #
    # print(audio_embeds.size(), video_embeds.size())
    #
    # # AV-HuBERT run
    # config.avhubert_config._attn_implementation = "flash_attention_2"
    # avhubert_encoder = AVHubertModel(config.avhubert_config)
    # avhubert_encoder = avhubert_encoder.cuda().to(torch.bfloat16)
    #
    # print(avhubert_encoder)
    #
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    #     with torch.inference_mode():
    #         av_features = avhubert_encoder.forward_av_features(audio_embeds, video_embeds,
    #                                                            audio_features_mask)
    # print("AV Profiler hits:", hits)
    #
    # print(av_features.size())
