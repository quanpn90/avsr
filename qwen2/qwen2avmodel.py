import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict
from transformers.cache_utils import Cache
from transformers import Qwen2AudioForConditionalGeneration

from contextlib import nullcontext

from qwen2.qwen2model import (MemoryEfficientQwen2AudioLM, Qwen2AudioConfig, Qwen2AudioCausalLMOutputWithPast,
                              BayesianQwen2AudioCausalLMOutputWithPast, fast_xentropy, softmax_xentropy)
from src.nets.backend.backbones.avhubert import AVHubertModel
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from src.nets.backend.nets_utils import make_non_pad_mask

from qwen2.qwen2avconfig import Qwen2AudioVideoConfig
from transformers import AutoModel

from transformers.utils import logging

logger = logging.get_logger(__name__)


class MemoryEfficientQwen2AVLM(MemoryEfficientQwen2AudioLM):
    config_class = Qwen2AudioVideoConfig
    whisper_encoder_mask_prob = 0.0
    avhubert_encoder_mask_prob = 0.0

    def __init__(self, config: Qwen2AudioVideoConfig):
        # first we create audio encoder and language model
        super().__init__(config.qwen2audio_config)
        self.config = config

        # next we create avhubert encoder
        # always drop audio, video is never dropped
        config.avhubert_config.audio_dropout = 1.0
        config.avhubert_config.modality_dropout = config.avhubert_audio_mask_prob
        self.avhubert_encoder = AVHubertModel(config.avhubert_config)
        self.freeze_av_encoder = False

        # finally create the link between avencoder and language model
        video_embed_size = self.avhubert_encoder.encoder_embed_dim

        self.video_feature_projector = nn.Linear(video_embed_size,
                                                 self.config.qwen2audio_config.text_config.hidden_size,
                                                 bias=True)

    def set_audio_mask(self, args):

        self.config.whisper_encoder_mask_prob = args.whisper_encoder_mask_prob
        self.config.avhubert_audio_mask_prob = args.avhubert_audio_mask_prob

        # set
        self.config.avhubert_config.audio_dropout = 1.0
        self.config.avhubert_config.modality_dropout = self.config.avhubert_audio_mask_prob
        self.avhubert_encoder.modality_dropout = self.config.avhubert_config.modality_dropout

        self.whisper_encoder_mask_prob = args.whisper_encoder_mask_prob
        self.avhubert_encoder_mask_prob = args.avhubert_audio_mask_prob

    def load_pretrained_avhubert(self,
                                 encoder_pretrained_checkpoint="nguyenvulebinh/avhubert_encoder_large_noise_pt_noise_ft_433h",
                                 cache_dir="model-bin/cache"):
        print("Loading pretrained encoder from", encoder_pretrained_checkpoint)
        encoder_pretrained = self.avhubert_encoder.from_pretrained(
            encoder_pretrained_checkpoint,
            cache_dir=cache_dir,
        )

        self.avhubert_encoder.load_state_dict(encoder_pretrained.state_dict())

    def load_pretrained_qwen2audio(self,
                                   model_name="Qwen/Qwen2-Audio-7B",
                                   cache_dir="model-bin/cache"):

        pretrained_qwen2audio = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)

        self.audio_tower.load_state_dict(pretrained_qwen2audio.audio_tower.state_dict())
        self.language_model.load_state_dict(pretrained_qwen2audio.language_model.state_dict())
        self.multi_modal_projector.load_state_dict(pretrained_qwen2audio.multi_modal_projector.state_dict())

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_features: Optional[torch.FloatTensor] = None,
            # av_features: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
            audios=None,
            videos=None,
            attention_mask: Optional[torch.Tensor] = None,
            feature_attention_mask: Optional[torch.Tensor] = None,
            av_feature_attention_mask: Optional[torch.Tensor] = None,
            audio_lengths=None,
            video_lengths=None,
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

                    # if legacy_processing:
                    #     # logger.warning_once(
                    #     #     "Expanding inputs for audio tokens in Qwen2Audio should be done in processing."
                    #     # )
                    #     # inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                    #     #     audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
                    #     # )
                    #     raise NotImplementedError
                    # else:
                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None,
                                          :]
                    audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                    audio_features = audio_features[audio_features_mask]

                    n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
                    n_audio_features = audio_features.shape[0]

                    if n_audio_tokens != n_audio_features:
                        print(self.config.audio_token_id)
                        print(input_ids)
                        raise ValueError(
                            f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                        )
                    special_audio_mask = (input_ids == self.config.audio_token_id).to(inputs_embeds.device)
                    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)

                    # randomly mask out the features
                    audio_features = torch.nn.functional.dropout(audio_features, self.whisper_encoder_mask_prob,
                                                                 training=self.training)
                    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

            # else:
            #     print("Ignore audio", input_ids.shape)

            # 3. merge text/audio and videos
            if videos is not None and input_ids.shape[1] != 1:
                context = torch.no_grad() if self.freeze_av_encoder else nullcontext()
                with context:

                    # TODO: import make_non_pad_mask
                    av_padding_mask = make_non_pad_mask(video_lengths).to(videos.device)
                    # attention_mask =
                    avhubert_features = self.avhubert_encoder(
                        input_features=audios,
                        video=videos,
                        attention_mask=av_padding_mask
                    )

                    selected_video_feature = avhubert_features.last_hidden_state

                    video_features = self.video_feature_projector(selected_video_feature)

                    # if we have consecutive audio tokens, then it means we expanded input_ids in processing
                    # video_tokens = input_ids == self.config.video_token_id

                    num_videos, max_video_tokens, embed_dim = video_features.shape
                    video_features_mask = torch.arange(max_video_tokens, device=audio_output_lengths.device)[None, :]
                    # print(video_features_mask.size(), video_lengths.size())
                    video_features_mask = video_features_mask < video_lengths[:, None]
                    video_features = video_features[video_features_mask]

                    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                    n_video_features = video_features.shape[0]

                    if n_video_tokens != n_video_features:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )
                    special_video_mask = (input_ids == self.config.video_token_id).to(inputs_embeds.device)
                    special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

        #     else:
        #         print("Ignore video", input_ids.shape)
        # # After replacing the <AUDIO> with audio encoded features

        # print(use_cache, past_key_values)

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

            # print(logits.size())

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


from qwen2.qwen2model import (MemoryOptimizedQwen2AudioEncoder, MemoryEfficientLayerNorm,
                              Qwen2RMSNorm, FusedRMSNorm, Qwen2MLP, SwiGLU_mlp, copy_weight)

from qwen2.qwen2model import fast_layer_norm_cuda, fast_rms_norm_cuda, fast_xentropy, swiglu_mlp_cuda


def create_qwen2av_model(load_path,
                         config,
                         torch_dtype,
                         trust_remote_code=True,
                         attn_implementation="flash_attention_2",
                         low_cpu_mem_usage=False,
                         device_map="none",
                         mem_efficient=True,
                         create_if_missing=False):
    # here model_name can be either the huggingface path, or the local path
    print("[INFO] Creating Qwen2AV model from %s" % load_path)

    def replace_audio_encoder(model):
        config = model.audio_tower.config

        old_encoder = model.audio_tower

        new_encoder = MemoryOptimizedQwen2AudioEncoder(config)

        # Copy weights from the old layer to the new one
        new_encoder.load_state_dict(old_encoder.state_dict())

        dtype = next(old_encoder.parameters()).dtype

        new_encoder.to(dtype)

        model.audio_tower = new_encoder

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

    model_class = MemoryEfficientQwen2AVLM

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
        # model = model_class(config)
        # model = model.to(torch_dtype)

        print("No checkpoint found, creating new model from config.")
        model = model_class(config)
        model = model.to(torch_dtype)
        # Optionally save as prototype if in training mode and path is given
        if create_if_missing and load_path:
            print(f"Saving prototype model to {load_path}")
            model.save_pretrained(load_path)

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


AutoModel.register(Qwen2AudioVideoConfig, MemoryEfficientQwen2AVLM)
