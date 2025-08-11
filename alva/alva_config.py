import os
# from qwen2.qwen2model import MemoryEfficientQwen2AudioLM, Qwen2AudioConfig
from transformers.models.qwen2_audio.configuration_qwen2_audio import (PretrainedConfig,
                                                                       AutoConfig,
                                                                       CONFIG_MAPPING,
                                                                       Qwen2AudioConfig,
                                                                       Qwen2AudioEncoderConfig)

from transformers import LlamaConfig
from transformers import WhisperConfig
from transformers import SiglipVisionConfig, LlavaOnevisionConfig
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
import json

default_whisper = "openai/whisper-large-v3"
default_llama = "meta-llama/Meta-Llama-3-8B"
default_qwen2 = "Qwen/Qwen2-Audio-7B"
default_llava = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


class AlvaConfig(PretrainedConfig):
    model_type = "alva"
    attribute_map = {
        "audio_token_id": "audio_token_index",
        "video_token_id": "video_token_index",
    }
    sub_configs = {"text_config": LlamaConfig,
                   "audio_config": Qwen2AudioEncoderConfig,
                   "video_config": SiglipVisionConfig,
                   "avhubert_config": AVHubertAVSRConfig}

    qwen_audio_mask_prob: float = 0.0
    avhubert_audio_mask_prob: float = 0.0
    finetune_avhubert: bool = False
    finetune_qwen2audio: bool = False
    finetune_llm: bool = False

    # hardcode the indices?
    audio_token_index = 128256
    video_token_index = 128259
    pad_token_id = 128001

    ctc_loss_alpha = 0.2

    def __init__(
            self,
            text_config=None,
            audio_config=None,
            video_config=None,
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

            audio_config = Qwen2AudioEncoderConfig.from_pretrained(default_qwen2)
        self.audio_config = audio_config

        # Ensure sub-configs are config objects, not dicts
        if isinstance(video_config, dict):
            video_config["model_type"] = (
                video_config["model_type"] if "model_type" in video_config else "siglip_vision_model"
            )
            video_config = SiglipVisionConfig(**video_config)
        elif video_config is None:

            llava_config = LlavaOnevisionConfig.from_pretrained(default_llava)
            video_config = llava_config.vision_config

        self.video_config = video_config

        if isinstance(text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "llama"
            )
            text_config = LlamaConfig(**text_config)
        elif text_config is None:
            text_config = LlamaConfig.from_pretrained(default_llama)

            # for the new added tokens
            text_config.vocab_size = text_config.vocab_size + 6

        self.text_config = text_config

        if isinstance(avhubert_config, dict):
            avhubert_config["model_type"] = avhubert_config.get("model_type", "avhubert_avsr")
            avhubert_config = AVHubertAVSRConfig(**avhubert_config)
        elif avhubert_config is None:
            from src.tokenizer.spm_tokenizer import TextTransform

            sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         "src/tokenizer/spm/unigram/unigram5000.model")
            dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "src/tokenizer/spm/unigram/unigram5000_units.txt")
            text_transform = TextTransform(
                sp_model_path=sp_model_path,
                dict_path=dict_path,
            )

            avhubert_config = AVHubertAVSRConfig(odim=len(text_transform.token_list),
                                                 attn_implementation="flash_attention_2")

        self.avhubert_config = avhubert_config

        self.whisper_encoder_mask_prob = whisper_encoder_mask_prob
        self.avhubert_audio_mask_prob = avhubert_audio_mask_prob

        self.finetune_llm = finetune_llm
        self.finetune_qwen2audio = finetune_qwen2audio
        self.finetune_avhubert = finetune_avhubert

        super().__init__(**kwargs)


AutoConfig.register("alva", AlvaConfig)

if __name__ == "__main__":
    import torch
    from transformers import (AutoTokenizer, AutoProcessor, LlavaOnevisionProcessor,
                              AutoConfig, LlavaOnevisionForConditionalGeneration)
    from src.dataset.avhubert_dataset import load_video_rgb, load_audio, cut_or_pad

    alva_config = AlvaConfig()

    print(alva_config)

    print("Created Alva config successfully")

    alva_config.save_pretrained("alva_base")

    default_llava = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    import datasets

    cache_dir = "/hkfs/work/workspace/scratch/pj5699-bayesian_speechlm/text2asr/data-bin/cache"



    llm = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                                 cache_dir=cache_dir,
                                                                 torch_dtype=torch.bfloat16,
                                                                 device_map="auto"
                                                                 )

    print(llm)
