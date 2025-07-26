# from qwen2.qwen2model import MemoryEfficientQwen2AudioLM, Qwen2AudioConfig
from transformers.models.qwen2_audio.configuration_qwen2_audio import (PretrainedConfig,
                                                                       AutoConfig,
                                                                       CONFIG_MAPPING,
                                                                       Qwen2AudioConfig,
                                                                       Qwen2AudioEncoderConfig)

from transformers import LlamaConfig
from transformers import WhisperConfig
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
import json

default_whisper = "openai/whisper-large-v3"
default_llama = "meta-llama/Meta-Llama-3-8B"
default_qwen2 = "Qwen/Qwen2-Audio-7B"


class LLama3AVConfig(PretrainedConfig):
    model_type = "llama3_av"
    attribute_map = {
        "audio_token_id": "audio_token_index",
        "video_token_id": "video_token_index",
    }
    sub_configs = {"text_config": LlamaConfig, "audio_config": Qwen2AudioEncoderConfig,
                   "avhubert_config": AVHubertAVSRConfig}

    qwen_audio_mask_prob: float = 0.0
    avhubert_audio_mask_prob: float = 0.0
    finetune_avhubert: bool = False
    finetune_qwen2audio: bool = False
    finetune_llm: bool = False

    # hardcode the indices?
    audio_token_index = 128256
    video_token_index = 128259

    def __init__(
            self,
            text_config=None,
            audio_config=None,
            avhubert_config=None,
            whisper_encoder_mask_prob=0.0,
            avhubert_audio_mask_prob=0.0,
            finetune_avhubert: bool = False,
            finetune_qwen2audio: bool = False,
            finetune_llm: bool = False,
            **kwargs,
    ):

        # Ensure sub-configs are config objects, not dicts
        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "qwen2_audio_encoder"
            )
            audio_config = Qwen2AudioEncoderConfig(**audio_config)
        elif audio_config is None:

            # maybe set the default to be Whisper Large v3
            audio_config = Qwen2AudioEncoderConfig.from_pretrained(default_qwen2)
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "llama"
            )
            text_config = LlamaConfig(**text_config)
        elif text_config is None:
            text_config = LlamaConfig.from_pretrained(default_llama)

        self.text_config = text_config

        if isinstance(avhubert_config, dict):
            avhubert_config["model_type"] = avhubert_config.get("model_type", "avhubert_avsr")
            avhubert_config = AVHubertAVSRConfig(**avhubert_config)
        elif avhubert_config is None:
            avhubert_config = AVHubertAVSRConfig()

        self.avhubert_config = avhubert_config

        self.whisper_encoder_mask_prob = whisper_encoder_mask_prob
        self.avhubert_audio_mask_prob = avhubert_audio_mask_prob

        self.finetune_llm = finetune_llm
        self.finetune_qwen2audio = finetune_qwen2audio
        self.finetune_avhubert = finetune_avhubert

        super().__init__(**kwargs)


AutoConfig.register("llama3_av", LLama3AVConfig)
