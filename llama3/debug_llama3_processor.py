import torch


from llama3.av_processor import Llama3AVProcessor
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoProcessor
from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, load_audio, load_video
import datasets

default_whisper = "openai/whisper-large-v3"
default_llama = "llama-extended"
cache_dir = "data-bin/cache"
default_qwen2 = "Qwen/Qwen2-Audio-7B"


# feature_extractor = WhisperFeatureExtractor.from_pretrained(default_whisper,
#                                                             cache_dir=cache_dir)

qwen2_processor = AutoProcessor.from_pretrained(default_qwen2, cache_dir=cache_dir)

feature_extractor = qwen2_processor.feature_extractor

tokenizer = AutoTokenizer.from_pretrained(default_llama,
                                          cache_dir=cache_dir)

eos = tokenizer.eos_token
print(eos)
eos_id = tokenizer.eos_token_id
print(eos_id)

text = "I have no idea"
print(tokenizer(text + tokenizer.eos_token, add_special_tokens=False)["input_ids"])



# <|AUDIO|> 128256
# <|audio_bos|> 128257
# <|audio_eos|> 128258
# <|VIDEO|> 128259
# <|video_bos|> 128260
# <|video_eos|> 128261

processor = Llama3AVProcessor(feature_extractor=feature_extractor,
                              tokenizer=tokenizer,
                              # debug=True
                              )

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
text = prompt_template + "<|begin_of_text|>" + text.lower() + tokenizer.eos_token


video_array2 = sample2['video']
text2 = sample2["label"].decode(encoding="utf-8") + " but I don't know"

audio2 = load_audio(video_array2).squeeze().numpy()
video2 = load_video(video_array2)

text2 = prompt_template + "<|begin_of_text|>" + text2.lower() + tokenizer.eos_token


inputs = processor(text=[text, text2], audio=[audio, audio2], video=[video, video2],
                   return_tensors="pt", padding=True)

print(text)
print(inputs.keys())

print("testing successfully!!")

processor.save_pretrained("llama-extended")

print("processor saved successfully!!")
