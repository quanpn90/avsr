import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoProcessor, AutoConfig, Qwen2AudioProcessor
from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, load_audio, load_video
from typing import List, Union

from qwen2.av_processor import Qwen2AudioVideoProcessor

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Python <3.12

model_name = "Qwen/Qwen2-Audio-7B"

processor = Qwen2AudioProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = processor.tokenizer

tokenizer.add_tokens(["<|VIDEO|>"])
tokenizer.add_tokens(["<|video_bos|>"])
tokenizer.add_tokens(["<|video_eos|>"])

new_processor = Qwen2AudioVideoProcessor.from_qwen2_processor(processor)

tokenizer = new_processor.tokenizer

vocab = tokenizer.get_vocab()

for token in ["<|AUDIO|>", "<|audio_bos|>", "<|audio_eos|>", "<|VIDEO|>", "<|video_bos|>", "<|video_eos|>"]:
    if token in vocab:
        print("%s exists in the vocabulary!: %d" % (token, vocab[token]))
    else:
        print("%s does NOT exist in the vocabulary." % token)

# test tokenizer

import datasets

cache_dir = "data-bin/cache/"
test_dataset = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=True,
                                     cache_dir=cache_dir).remove_columns(['__key__', '__url__'])['train']

sample = next(iter(test_dataset))

video_array = sample['video']
text = sample["label"].decode(encoding="utf-8")
audio = load_audio(video_array).squeeze().numpy()
video = load_video(video_array)

prompt_template = "<|video_bos|><|VIDEO|><|video_eos|><|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech:"
text = prompt_template + text.lower()

inputs = new_processor(text, audio, video)

# print(inputs.keys())
print(inputs["input_ids"])

print("testing successfully!!")

# processor_path = "/hkfs/work/workspace/scratch/pj5699-bayesian_speechlm/avsr/lsr_vox_cocktail/" + "av_tokenizer"
# new_processor.save_pretrained(processor_path)

# running script
# PYTHONPATH=$HOME/avsr/ python debug_qwen_tokenizer.py
