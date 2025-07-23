try:
    from torchcodec.decoders import AudioDecoder
except ImportError:
    AudioDecoder = None

import re
import torch
import torchaudio.transforms as T
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from src.dataset.avhubert_dataset import AudioTransform, load_audio, load_video
from src.dataset.avhubert_dataset import AdaptiveTimeMask, AddMultiSpk, AddNoise, FunctionalModule, cut_or_pad
from src.dataset.avhubert_dataset import FBanksAndStack, collate_pad, VideoTransform

from src.dataset.qwen_audio_dataset import normalize_transcript, WavAudioTransform


QWEN2_LANG_TO_TOKEN = {

    "en": "<|en|>",
    "es": "<|es|>",
    "zh": "<|zh|>",
    "ar": "<|ar|>",
    "de": "<|de|>",
    "tr": "<|tr|>",
    "ja": "<|ja|>",

    # TODO: automatically add from official repo
}

LANG_TOKENS = set(QWEN2_LANG_TO_TOKEN.values())






@dataclass
class Qwen2AVDataCollator:
    # text_transform: TextTransform = None
    # processor: Any
    feature_extractor: Any
    text_processor: Any
    model_config: Any
    uid_mapper: Any
    dataset: Any

    audio_transform: WavAudioTransform = None
    video_transform: VideoTransform = None
    spec_time_masking = T.TimeMasking(time_mask_param=30)
    spec_freq_masking = T.FrequencyMasking(freq_mask_param=30)
    rate_ratio: int = 640

    def __init__(self, processor,
                 prompt_template="<|video_bos|><|VIDEO|><|video_eos|> <|audio_bos|><|AUDIO|><|audio_eos|> Transcribe this speech:",
                 eos_token="<|endoftext|>",
                 eos_token_id=151643,
                 audio_eos_token_id=151648,
                 video_transform=None,
                 audio_transform=None,
                 wav_augment=False,
                 spec_augment=False,
                 include_text=False):
        self.processor = processor
        self.prompt_template = prompt_template
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        self.prompt_len = len(self.processor.tokenizer(self.prompt_template)["input_ids"])

        self.lang_ids = {token: processor.tokenizer.convert_tokens_to_ids(token) for token in LANG_TOKENS}

        self.do_wav_augment = wav_augment
        self.do_spec_augment = spec_augment

        self.include_text = include_text
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.fbank_and_stack = FBanksAndStack()

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        audio_list = list()
        video_list = list()
        text_list = list()
        language = "en"
        samples = list()
        for feature in features:
            if "start_time" in feature and "end_time" in feature:
                video = load_video(feature["video"], feature["start_time"], feature["end_time"])
            else:
                video = load_video(feature["video"])

            if "start_time" in feature and "end_time" in feature:
                audio = load_audio(feature["video"], feature["start_time"], feature["end_time"])
            else:
                audio = load_audio(feature["video"])

            # here we don't have to cut or pad if there is no video involved
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            video = self.video_transform(video)
            video_list.append(video)

            # here we divide the audio transform into 2 parts
            # the augmentation part can be used for both Whisper and AVHubert
            # the fbank & stack is only for AVHubert
            audio = self.audio_transform(audio)
            audio_fbank = self.fbank_and_stack(audio)

            # note: the huggingface processor takes numpy instead of torch
            audio = audio.squeeze().numpy()
            audio_list.append(audio)

            label = feature["label"]
            text_list.append(normalize_transcript(label))

            samples.append({"video": video, "audio": audio_fbank})

        # Prepare prompt+target format for Qwen2-Audio
        label_list = [
            self._concat_prompt(example, QWEN2_LANG_TO_TOKEN[language])
            for example in text_list
        ]

        # sample rate taken from avhubert_dataset as 16000
        inputs = self.processor(
            audio=audio_list,
            text=label_list,
            video=video_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True  # pads both input_ids and audio
        )

        inputs_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs_ids.clone()

        batch_size = labels.size(0)

        for i in range(batch_size):

            label = labels[i]
            eos_pos = (label == self.audio_eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) == 0:
                raise ValueError("Missing <|audio_eos|> token.")
            eos_idx = eos_pos.item()

            # Mask everything up to and including <|audio_eos|>
            labels[i, :eos_idx + 1] = -100

            # Find the first <|lang|> token after <|audio_eos|>
            lang_id = self.lang_ids[QWEN2_LANG_TO_TOKEN[language]]
            lang_pos = (label == lang_id).nonzero(as_tuple=True)[0]
            if len(lang_pos) == 0:
                # print(batch[i]["language"], lang_id, lang_pos)
                raise ValueError("Missing <|lang|> token.")
            lang_idx = lang_pos.item()

            # Mask everything between eos and lang token (excluding lang token itself)
            if lang_idx > eos_idx + 1:
                labels[i, eos_idx + 1:lang_idx] = -100

        # labels.masked_fill_(attention_mask.ne(1), -100)
        inputs["labels"] = labels

        if self.include_text:
            inputs["tgt_txt"] = text_list

        # process inputs for avhubert
        avhubert_inputs = collate_pad(samples)
        
        inputs["videos"] = avhubert_inputs["videos"].permute(0, 2, 1, 3, 4)
        inputs["audios"] = avhubert_inputs["audios"].permute(0, 2, 1)
        inputs["video_lengths"] = avhubert_inputs["video_lengths"]
        inputs["audio_lengths"] = avhubert_inputs["audio_lengths"]

        return inputs

    def _concat_prompt(self, target_text, language_token):
        # add space between or not add space between?
        return f"{self.prompt_template} {language_token} {target_text} {self.eos_token}"


# class Qwen2AVEvalCollator(Qwen2AVDataCollator):
#
#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#
#         audio_list = list()
#         text_list = list()
#         language = "en"
#         for feature in features:
#
#             if "start_time" in feature and "end_time" in feature:
#                 audio = load_audio(feature["video"], feature["start_time"], feature["end_time"])
#             else:
#                 audio = load_audio(feature["video"])
#
#             # here we don't have to cut or pad if there is no video involved
#             # audio = cut_or_pad(audio, len(video) * self.rate_ratio)
#             #
#             # video = self.video_transform(video)
#             audio = self.audio_transform(audio)
#             audio = audio.squeeze().numpy()
#             # print(audio, audio.shape)
#             audio_list.append(audio)
#
#             # if "label" in feature:
#             #     label = self.text_transform.tokenize(feature["label"])
#             #     samples.append({"video": video, "audio": audio, "label": label})
#             # else:
#             #     samples.append({"video": video, "audio": audio})
#
#         # Prepare prompt+target format for Qwen2-Audio
#         label_list = [
#             self._get_prompt(QWEN2_LANG_TO_TOKEN[language])
#             for _ in features
#         ]
#
#         # sample rate taken from avhubert_dataset as 16000
#         inputs = self.processor(
#             audio=audio_list,
#             text=label_list,
#             sampling_rate=16000,
#             return_tensors="pt",
#             padding=True  # pads both input_ids and audio
#         )
#
#         # inputs_ids = inputs["input_ids"]
#         # attention_mask = inputs["attention_mask"]
#
#         return inputs
#
#     def _get_prompt(self, language_token):
#         # add space between or not add space between?
#         return f"{self.prompt_template} {language_token}"

