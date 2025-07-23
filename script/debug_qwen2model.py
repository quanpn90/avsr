import torch
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import (Qwen2AudioPreTrainedModel,
                                                                  Qwen2AudioCausalLMOutputWithPast,
                                                                  Qwen2AudioForConditionalGeneration)

# This script expands the vocabulary to accommodate the AUDIO and VIDEO tokens

model_name = "Qwen/Qwen2-Audio-7B"

# <|AUDIO|> exists in the vocabulary!: 151646
# <|audio_bos|> exists in the vocabulary!: 151647
# <|audio_eos|> exists in the vocabulary!: 151648
# <|VIDEO|> exists in the vocabulary!: 154931
# <|video_bos|> exists in the vocabulary!: 154932
# <|video_eos|> exists in the vocabulary!: 154933

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")

print(model.get_input_embeddings().weight)
print(model.language_model.lm_head.weight)

embedding_size = model.language_model.model.embed_tokens.weight.size(0)
model.resize_token_embeddings(embedding_size + 3)


print(model.get_input_embeddings().weight.data_ptr() == model.language_model.lm_head.weight.data_ptr())


video_ids = [154931, 154932, 154933]
audio_ids = [151646, 151647, 151648]

with torch.no_grad():
    for video_id, audio_id in zip(video_ids, audio_ids):
        model.get_input_embeddings().weight.data[video_id] = model.get_input_embeddings().weight.data[audio_id] + \
                                                        0.01 * torch.randn_like(model.get_input_embeddings().weight.data[audio_id])
        model.language_model.lm_head.weight.data[video_id] = model.language_model.lm_head.weight.data[audio_id] + \
                                                        0.01 * torch.randn_like(model.language_model.lm_head.weight.data[audio_id])


model_path = "/hkfs/work/workspace/scratch/pj5699-bayesian_speechlm/avsr/lsr_vox_cocktail/" + "av_tokenizer"
model.save_pretrained(model_path)
print("DONE")
