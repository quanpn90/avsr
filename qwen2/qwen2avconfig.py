# from qwen2.qwen2model import MemoryEfficientQwen2AudioLM, Qwen2AudioConfig
from transformers import PretrainedConfig, Qwen2Config, Qwen2AudioConfig, AutoConfig, Qwen2AudioEncoderConfig

from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
import json

default_qwen2 = "Qwen/Qwen2-Audio-7B"


class Qwen2AVConfig(PretrainedConfig):
    model_type = "qwen2_av"
    attribute_map = {
        "audio_token_id": "audio_token_index",
        "video_token_id": "video_token_index",
    }
    sub_configs = {"text_config": Qwen2Config, "audio_config": Qwen2AudioEncoderConfig,
                   "avhubert_config": AVHubertAVSRConfig}

    qwen_audio_mask_prob: float = 0.0
    avhubert_audio_mask_prob: float = 0.0
    finetune_avhubert: bool = False
    finetune_qwen2audio: bool = False
    finetune_llm: bool = False

    video_token_index = 154931
    audio_token_index = 151646

    pad_token_id = 151643
    eos_token_id = 151645

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
        # self.video_token_index = video_token_index
        _default_qwen2audio_config = Qwen2AudioConfig.from_pretrained(default_qwen2)

        # Ensure sub-configs are config objects, not dicts
        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "qwen2_audio_encoder"
            )
            audio_config = Qwen2AudioEncoderConfig(**audio_config)
        elif audio_config is None:

            # maybe set the default to be Whisper Large v3
            audio_config = _default_qwen2audio_config.audio_config
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "qwen2"
            )
            text_config = Qwen2Config(**text_config)
        elif text_config is None:
            text_config = _default_qwen2audio_config.text_config

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


AutoConfig.register("qwen2_av", Qwen2AVConfig)
