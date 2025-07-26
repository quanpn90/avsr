from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperModel
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder
from transformers import AutoConfig, WhisperConfig, AutoModel
from transformers import Qwen2AudioForConditionalGeneration
from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig

default_whisper = "openai/whisper-large-v3"
default_qwen2 = "Qwen/Qwen2-Audio-7B"
cache_dir = "data-bin/cache"


#

# whisper_encoder = whisper.encoder

whisper_config = WhisperConfig.from_pretrained(default_whisper, cache_dir=cache_dir)

# qwen2_encoder = Qwen2AudioEncoder(whisper_config)

qwen2_config = AutoConfig.from_pretrained(default_qwen2, cache_dir=cache_dir)

print(qwen2_config.audio_config)
print("============================")
print(whisper_config)

qwen2audio = Qwen2AudioForConditionalGeneration.from_pretrained(default_qwen2, cache_dir=cache_dir)

whisper = WhisperModel.from_pretrained(default_whisper, cache_dir=cache_dir)

qwen2audio.audio_tower.load_state_dict(whisper.encoder.state_dict())


qwen2audio_config = Qwen2AudioEncoderConfig.from_pretrained(default_qwen2)

print(qwen2audio_config)