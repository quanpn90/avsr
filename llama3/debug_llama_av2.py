import os

local_rank = int(os.environ.get("LOCAL_RANK", 0))
from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, load_audio, load_video
import datasets

import transformers
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import LlamaModel, LlamaForCausalLM, Qwen2AudioForConditionalGeneration
from llama3.llama3avmodel import MemoryEfficientLLama3AVLM
from llama3.llama3avconfig import LLama3AVConfig, LlamaConfig
from llama3.av_processor import Llama3AVProcessor
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig

torch_dtype = torch.bfloat16
attention_type = "flash_attention_2"
device = torch.device(f"cuda:{local_rank}")
cache_dir = "data-bin/cache"

# config = LLama3AVConfig.from_pretrained("llama3av-base")
#
# model = MemoryEfficientLLama3AVLM.from_pretrained("llama3av-base", torch_dtype=torch_dtype,
#                                                   device_map={"": device},
#                                                   attn_implementation=attention_type)

processor = Llama3AVProcessor.from_pretrained("llama3av-base")
processor.tokenizer.pad_token = processor.tokenizer.eos_token

print(processor.tokenizer.eos_token_id)
print(processor.tokenizer.bos_token_id)
print(processor.tokenizer.bos_token)
#
# model.save_pretrained("llama3av-base")
processor.save_pretrained("llama3av-base")

test_dataset = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=True,
                                     cache_dir=cache_dir).remove_columns(['__key__', '__url__'])['train']

sample = next(iter(test_dataset))

video_array = sample['video']
text = sample["label"].decode(encoding="utf-8")

audio = load_audio(video_array).squeeze().numpy()
video = load_video(video_array)

prompt_template = "<|video_bos|><|VIDEO|><|video_eos|><|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech:"


def _concat_prompt(target_text):
    # add space between or not add space between?

    return f"{prompt_template}{processor.tokenizer.bos_token}{target_text}{processor.tokenizer.eos_token}"


# text = prompt_template + text.lower() + processor.tokenizer.bos_token

text = _concat_prompt(text.lower())

inputs = processor(text, audio, video, return_tensors="pt")

print(text)
# print(inputs.keys())

print(inputs)

input_ids = inputs["input_ids"][0]

print(processor.tokenizer.bos_token_id)
print(input_ids)
bos_pos = (input_ids == processor.tokenizer.bos_token_id).nonzero(as_tuple=True)[0]

assert (len(bos_pos) == 2)

label = input_ids.clone()

last_bos = bos_pos[1]
label[:last_bos + 1] = -100

print(label)

print("testing successfully!!")
