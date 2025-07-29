from qwen2.qwen2model import MemoryEfficientQwen2AudioLM, Qwen2AudioConfig
from transformers.models.qwen2_audio.configuration_qwen2_audio import (PretrainedConfig,
                                                                       AutoConfig,
                                                                       CONFIG_MAPPING)

from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
import json

default_qwen2 = "Qwen/Qwen2-Audio-7B"


class Qwen2AudioVideoConfig(PretrainedConfig):
    model_type = "qwen2_av"
    attribute_map = {
        "audio_token_id": "audio_token_index",
        "video_token_id": "video_token_index",
    }
    sub_configs = {"qwen2audio_config": Qwen2AudioConfig, "avhubert_config": AVHubertAVSRConfig}

    qwen_audio_mask_prob: float = 0.0
    avhubert_audio_mask_prob: float = 0.0
    finetune_avhubert: bool = False
    finetune_qwen2audio: bool = False
    finetune_llm: bool = False

    video_token_index = 154931
    audio_token_index = 151646

    def __init__(
            self,
            qwen2audio_config=None,
            avhubert_config=None,
            video_token_index=154931,
            whisper_encoder_mask_prob=0.0,
            avhubert_audio_mask_prob=0.0,
            finetune_avhubert: bool = False,
            finetune_qwen2audio: bool = False,
            finetune_llm: bool = False,
            **kwargs,
    ):
        # self.video_token_index = video_token_index

        # Ensure sub-configs are config objects, not dicts
        if isinstance(qwen2audio_config, dict):
            qwen2audio_config["model_type"] = qwen2audio_config.get("model_type", "qwen2_audio")
            qwen2audio_config = Qwen2AudioConfig(**qwen2audio_config)
        elif qwen2audio_config is None:
            qwen2audio_config = Qwen2AudioConfig.from_pretrained(default_qwen2)

        self.qwen2audio_config = qwen2audio_config

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

        self.text_config = self.qwen2audio_config.text_config if self.qwen2audio_config is not None else None
        self.audio_config = self.qwen2audio_config.audio_config if self.qwen2audio_config is not None else None
        self.audio_token_index = self.qwen2audio_config.audio_token_index if self.qwen2audio_config is not None else None

        super().__init__(**kwargs)

    # @property
    # def text_config(self):
    #     return self.qwen2audio_config.text_config
    #
    # @text_config.setter
    # def text_config(self, value):
    #     self.qwen2audio_config.text_config = value
    #
    # @property
    # def audio_config(self):
    #     return self.qwen2audio_config.audio_config
    #
    # @audio_config.setter
    # def audio_config(self, value):
    #     self.qwen2audio_config.audio_config = value
    #
    # @property
    # def audio_token_index(self):
    #     return self.qwen2audio_config.audio_token_index
    #
    # @audio_token_index.setter
    # def audio_token_index(self, value):
    #     self.qwen2audio_config.audio_token_index = value

AutoConfig.register("qwen2_av", Qwen2AudioVideoConfig)
