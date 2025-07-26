from src.dataset.qwen_av_dataset import Qwen2AVDataCollator
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

class LLama3AVDataCollator(Qwen2AVDataCollator):
    # <|AUDIO|> 128256
    # <|audio_bos|> 128257
    # <|audio_eos|> 128258
    # <|VIDEO|> 128259
    # <|video_bos|> 128260
    # <|video_eos|> 128261

    bos_token = "<|begin_of_text|>"
    bos_token_id = 128000
    eos_token = "<|endoftext|>"
    eos_token_id = 128001
    audio_eos_token_id = 128258

    version = 2.0

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
            self._concat_prompt(example)
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
            bos_pos = (label == self.bos_token_id).nonzero(as_tuple=True)[0]

            # there must be two bos positions
            assert len(bos_pos) == 2

            # Mask everything up to and including the last bos token
            last_bos = bos_pos[1]
            labels[i, :last_bos + 1] = -100

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

    def _concat_prompt(self, target_text):
        return f"{self.prompt_template}{self.bos_token}{target_text}{self.eos_token}"
