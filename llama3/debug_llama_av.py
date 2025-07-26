import os
local_rank = int(os.environ.get("LOCAL_RANK", 0))



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

processor = Llama3AVProcessor.from_pretrained("llama-extended")

print(processor)

# first we have to create a config

llama3_config = LlamaConfig.from_pretrained("llama-extended")

avhubert_config = AVHubertAVSRConfig(odim=llama3_config.vocab_size,
                                     attn_implementation=attention_type)


# the audio config will be default Qwen2Audio
config = LLama3AVConfig(text_config=llama3_config,
                        avhubert_config=avhubert_config)

print(config)



config._attn_implementation = attention_type
model = MemoryEfficientLLama3AVLM(config)

model.to(torch_dtype).to(device)

print(model)

print("Created model successfully")

model.load_pretrained_avhubert()
model.load_pretrained_qwen2audio()
model.load_pretrained_llama()


print("Loaded pretrained weights successfully")

model.save_pretrained("llama3av-base")
processor.save_pretrained("llama3av-base")
