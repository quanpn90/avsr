import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict

from sympy.polys.polyoptions import Auto

#
# from llava.model.language_model.f16 import F16Config
# from llava.model.language_model.f16 import F16ForCausalLM

from src.dataset.avhubert_dataset import load_video_rgb, load_audio, cut_or_pad

cache_dir = "/hkfs/work/workspace/scratch/pj5699-bayesian_speechlm/text2asr/data-bin/cache"

# config = F16Config.from_json_file("f16-checkpoint/config.json")
#
# print(config)
#
# f16_model = F16ForCausalLM(config)
#
# print(f16_model)
#
#
# def strip_module_prefix(state_dict, prefix="module."):
#     """
#     Returns a new OrderedDict with the given prefix removed from each key (if present).
#     """
#     new_state_dict = OrderedDict()
#     plen = len(prefix)
#     for k, v in state_dict.items():
#         if k.startswith(prefix):
#             new_state_dict[k[plen:]] = v
#         else:
#             new_state_dict[k] = v
#     return new_state_dict
#
#
# ckpt = torch.load("f16-checkpoint/model.bin", map_location="cpu")
#
# clean_ckpt = strip_module_prefix(ckpt, prefix="module.")
#
# f16_model.load_state_dict(clean_ckpt)
#
# f16_model.save_pretrained("llava-f16")

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LlavaOnevisionVideoProcessor
from transformers import LlavaOnevisionProcessor, AutoConfig
import datasets

processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                cache_dir=cache_dir)

print("VIDEO PROCESSOR")
print(processor.video_processor)
print(type(processor.video_processor))

config = AutoConfig.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                    cache_dir=cache_dir)

# video_processor = LlavaOnevisionVideoProcessor(config)

# print(video_processor)

# print("IMAGE PROCESSOR")
# print(processor.image_processor)
#
test_dataset = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=True,
                                     cache_dir=cache_dir).remove_columns(['__key__', '__url__'])['train']

iterator = iter(test_dataset)
# sample = next(iterator)
# video_array = sample['video']
# #
# video_rgb = load_video_rgb(video_array)
# # text = sample["label"].decode(encoding="utf-8")
# #
# print(video_rgb.shape)
# #
# # print(video_rgb.shape)
# #
# # from torchvision.transforms.functional import to_pil_image
# #
# # frames_pil = [to_pil_image(frame) for frame in video_rgb]  # list[PIL.Image]
#
# out = processor.video_processor(video_rgb, return_tensors="pt")

video_rgbs = []

for i in range(5):
    sample = next(iterator)
    video_array = sample['video']
    video_rgb = load_video_rgb(video_array)

    video_rgbs.append(video_rgb)

out = processor.video_processor(video_rgbs, return_tensors="pt", padding="max_length")

for key in out:
    tensor = out[key]
    print(key, tensor.shape)
#
llm = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                             cache_dir=cache_dir,
                                                             torch_dtype=torch.bfloat16,
                                                             device_map="auto"
                                                             )

vision_tower = llm.model.vision_tower
print(vision_tower)
#
#
# # pixel_values = out['pixel_values_videos']
# # pixel_values = pixel_values.cuda().to(torch.bfloat16)
# # batch_size, frames, channels, height, width = pixel_values.shape
# # pixel_values = pixel_values.view(batch_size * frames, channels, height, width)
# # video_features = vision_tower(pixel_values, output_hidden_states=True)
#
#
# # vision_feature_layer = -1
# # selected_video_feature = video_features.hidden_states[vision_feature_layer]
# # selected_video_feature = selected_video_feature[:, 1:]
# #
# # video_features = selected_video_feature
# # # video_features = llm.model.multi_modal_projector(selected_video_feature)
#
#
# # TODO: save the vision tower
# # print(video_features.size())
#
#
# default_qwen2 = "Qwen/Qwen2-Audio-7B"
# qwen2_processor = AutoProcessor.from_pretrained(default_qwen2, cache_dir=cache_dir)
#
# # print(qwen2_processor)
#
# audio_feature_extractor = qwen2_processor.feature_extractor
#
# audio = load_audio(video_array).squeeze(1)
#
# print(audio.shape)
# print(video_rgb.shape)
#
# ratio = 640
# audio = cut_or_pad(audio, len(video_rgb) * ratio)
#
# qwen2_processor = audio_feature_extractor
#
# audio_kwargs = dict()
# audio_kwargs["return_attention_mask"] = True
# audio_kwargs["padding"] = "max_length"
# audio_features = audio_feature_extractor(audio, return_tensors="pt", **audio_kwargs)
#
# attention_mask = audio_features['attention_mask']
# feature_length = attention_mask.sum(-1).tolist()
# print(feature_length)
#
#
# audio_length = feature_length.pop(0)
# input_length = (audio_length - 1) // 2 + 1
# num_audio_tokens = (input_length - 2) // 2 + 1
#
# print(audio_features)
#
# print(num_audio_tokens)
