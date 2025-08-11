import datasets
import torch
from transformers import Qwen2AudioProcessor

from qwen2.av_processor import Qwen2AVProcessor
from qwen2.qwen2avconfig import Qwen2AVConfig, Qwen2AudioConfig, AVHubertAVSRConfig
from src.dataset.avhubert_dataset import load_audio, load_video

default_qwen2 = "Qwen/Qwen2-Audio-7B"
cache_dir = "data-bin/cache"

qwen2audio_config = Qwen2AudioConfig.from_pretrained(default_qwen2, cache_dir=cache_dir)
vocab_size = qwen2audio_config.text_config.vocab_size
avhubert_config = AVHubertAVSRConfig(odim=vocab_size)

qwen2av_config = Qwen2AVConfig(avhubert_config=avhubert_config)

q2_processor = Qwen2AudioProcessor.from_pretrained(default_qwen2, trust_remote_code=True)
tokenizer = q2_processor.tokenizer

tokenizer.add_tokens(["<|VIDEO|>"])
tokenizer.add_tokens(["<|video_bos|>"])
tokenizer.add_tokens(["<|video_eos|>"])


print(tokenizer.eos_token)
print(tokenizer.eos_token_id)
print(tokenizer.bos_token)
print(tokenizer.bos_token_id)
print(tokenizer.pad_token)
print(tokenizer.pad_token_id)

processor = Qwen2AVProcessor(feature_extractor=q2_processor.feature_extractor,
                             tokenizer=q2_processor.tokenizer, debug=True)


print("Processor initialized successfully!")
#
# print(processor)

test_dataset = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=True,
                                     cache_dir=cache_dir).remove_columns(['__key__', '__url__'])['train']

iterator = iter(test_dataset)
sample = next(iterator)
sample2 = next(iterator)

video_array = sample['video']
text = sample["label"].decode(encoding="utf-8")

audio = load_audio(video_array).squeeze().numpy()
video = load_video(video_array)


prompt_template = "<|video_bos|><|VIDEO|><|video_eos|><|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech:"
text = prompt_template + "<|en|>" + text.lower() + tokenizer.eos_token

video_array2 = sample2['video']
text2 = sample2["label"].decode(encoding="utf-8") + " but I don't know and I cannot do anything aganist it"

audio2 = load_audio(video_array2).squeeze().numpy()
video2 = load_video(video_array2)

text2 = prompt_template + "<|en|>" + text2.lower() + tokenizer.eos_token


print(video.size())
print(video2.size())
inputs = processor(text=[text, text2], audio=[audio, audio2], video=[video, video2],
                   return_tensors="pt", padding=True)

print(text)
print(inputs.keys())

print("testing successfully!!")

print("processor saved successfully!!")


# now we create the model


from transformers import Qwen2AudioForConditionalGeneration
#

torch_dtype = torch.bfloat16

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B",
                                                           device_map="cuda:0", cache_dir=cache_dir,
                                                           torch_dtype=torch_dtype)

print(model.get_input_embeddings().weight)
print(model.language_model.lm_head.weight)

embedding_size = model.language_model.model.embed_tokens.weight.size(0)
model.resize_token_embeddings(embedding_size + 3, mean_resizing=False)

video_ids = [154931, 154932, 154933]
audio_ids = [151646, 151647, 151648]

with torch.no_grad():
    for video_id, audio_id in zip(video_ids, audio_ids):
        model.get_input_embeddings().weight.data[video_id] = model.get_input_embeddings().weight.data[audio_id] + \
                                                        0.01 * torch.randn_like(model.get_input_embeddings().weight.data[audio_id])
        model.language_model.lm_head.weight.data[video_id] = model.language_model.lm_head.weight.data[audio_id] + \
                                                        0.01 * torch.randn_like(model.language_model.lm_head.weight.data[audio_id])

qwen2av_config.text_config.vocab_size += 3

from qwen2.qwen2avmodel import MemoryEfficientQwen2AVLM
model_class = MemoryEfficientQwen2AVLM

avmodel = model_class(qwen2av_config)
avmodel = avmodel.to(torch_dtype).cuda()

avmodel.load_pretrained_avhubert(cache_dir=cache_dir)

avmodel.language_model.load_state_dict(model.language_model.state_dict())
avmodel.audio_tower.load_state_dict(model.audio_tower.state_dict())
avmodel.multi_modal_projector.load_state_dict(model.multi_modal_projector.state_dict())

# avmodel.load_state_dict(model.state_dict(), strict=False)

print(avmodel)

avmodel.save_pretrained("qwen2av_base")
qwen2av_config.save_pretrained("qwen2av_base")
processor.save_pretrained("qwen2av_base")