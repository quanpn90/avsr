from qwen2.qwen2model import MemoryEfficientQwen2AudioLM, Qwen2AudioConfig
from transformers.models.qwen2_audio.configuration_qwen2_audio import (PretrainedConfig,
                                                                       AutoConfig,
                                                                       CONFIG_MAPPING)

from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig


class Qwen2AudioVideoConfig(PretrainedConfig):
    model_type = "qwen2_av"
    attribute_map = {
        "audio_token_id": "audio_token_index",
        "video_token_id": "video_token_index",
    }
    sub_configs = {"qwen2audio_config": Qwen2AudioConfig, "avhubert_config": AVHubertAVSRConfig}

    def __init__(
            self,
            qwen2audio_config=None,
            avhubert_config=None,
            video_token_index=154931,
            **kwargs,
    ):
        self.video_token_index = video_token_index

        # Ensure sub-configs are config objects, not dicts
        if isinstance(qwen2audio_config, dict):
            qwen2audio_config["model_type"] = qwen2audio_config.get("model_type", "qwen2_audio")
            qwen2audio_config = Qwen2AudioConfig(**qwen2audio_config)
        self.qwen2audio_config = qwen2audio_config

        if isinstance(avhubert_config, dict):
            avhubert_config["model_type"] = avhubert_config.get("model_type", "avhubert_avsr")
            avhubert_config = AVHubertAVSRConfig(**avhubert_config)
        self.avhubert_config = avhubert_config

        # Optionally propagate the audio token index from the subconfig
        self.audio_token_index = qwen2audio_config.audio_token_index

        self.text_config = qwen2audio_config.text_config
        self.audio_config = qwen2audio_config.audio_config

        super().__init__(**kwargs)
