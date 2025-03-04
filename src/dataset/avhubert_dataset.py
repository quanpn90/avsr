import os

import torch
import torchaudio
import torchvision
import random
from dataclasses import dataclass
from src.tokenizer.spm_tokenizer import TextTransform
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from python_speech_features import logfbank
import torch.nn.functional as F

def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid_rgb = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in vid_rgb.numpy()]
    vid = torch.from_numpy(np.stack(frames)).unsqueeze(1)
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    
    if path.endswith(".mp4") and os.path.exists(path.replace(".mp4", ".wav")):
        waveform, sample_rate = torchaudio.load(path.replace(".mp4", ".wav"), normalize=True)
    else:
        waveform, sample_rate = torchaudio.load(path, normalize=True)
    assert sample_rate == 16000
    return waveform.transpose(1, 0)


class FBanksAndStack(torch.nn.Module):
    def __init__(self, stack_order=4):
        super().__init__()
        self.stack_order = stack_order

    def stacker(self, feats):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % self.stack_order != 0:
            res = self.stack_order - len(feats) % self.stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, self.stack_order, feat_dim)).reshape(-1, self.stack_order*feat_dim)
        return feats

    def forward(self, x):
        # x: T x 1
        # return: T x F*stack_order
        audio_feats = logfbank(x.squeeze().numpy(), samplerate=16000).astype(np.float32) # [T, F]
        audio_feats = self.stacker(audio_feats) # [T/stack_order_audio, F*stack_order_audio]
        
        with torch.no_grad():
            audio_feats = F.layer_norm(torch.from_numpy(audio_feats), audio_feats.shape[1:])
        return audio_feats

def normalize_audio(waveform):
    max_val = torch.abs(waveform).max()
    return waveform / max_val if max_val > 0 else waveform

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=None,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        if noise_filename is None:
            # self.noise = torch.randn(1, 16000)
            self.noise = None
        else:
            self.noise, sample_rate = torchaudio.load(noise_filename)
            assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.noise is None:
            return speech
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()
    
class AddMultiSpk(torch.nn.Module):
    def __init__(
        self,
        speech_dataset=None,
        snr_target=None,
        interferer_spk=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20]
        self.interferer_spk = [interferer_spk] if interferer_spk else [0, 0, 1, 2]
        self.speech_dataset = speech_dataset

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.speech_dataset is None:
            return speech
        speech_length = speech.size(0) / 16000
        if speech_length < 2:
            return speech
        
        num_interferer = random.choice(self.interferer_spk)
        interferer_signal = None
        for _ in range(num_interferer):
            interferer = random.choice(self.speech_dataset)
            if 25 * 2 <= interferer["length"] <= 25 * 10:
                # print(interferer)
                interferer = load_audio(interferer["video"])
                interferer = cut_or_pad(interferer, len(speech))
                if interferer_signal is None:
                    interferer_signal = interferer
                else:
                    snr_level = torch.tensor([random.choice([-5, 0, 5, 10, 15])])
                    interferer_signal = torchaudio.functional.add_noise(interferer_signal.t(), interferer.t(), snr_level).t()        
        
        if interferer_signal is None:
            return speech
        
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        speech = torchaudio.functional.add_noise(speech.t(), interferer_signal.t(), snr_level).t()
        
        return speech


class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                # torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                # torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self, subset, speech_dataset=None, snr_target=None):
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                AdaptiveTimeMask(6400, 16000),
                AddMultiSpk(speech_dataset=speech_dataset),
                AddNoise(),
                FBanksAndStack(),
                # FunctionalModule(
                #     lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                # ),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(snr_target=snr_target)
                if snr_target is not None
                else FunctionalModule(lambda x: x),
                FBanksAndStack(),
                # FunctionalModule(
                #     lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                # ),
            )

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)



# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "label" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out
    
@dataclass
class DataCollator:
    text_transform: TextTransform = None
    video_transform: VideoTransform = None
    audio_transform: AudioTransform = None
    rate_ratio: int = 640

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # {"video": video, "audio": audio, "target": token_id}
        samples = []
        for feature in features:
            video = load_video(feature["video"])

            audio = load_audio(feature["video"])
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            label = self.text_transform.tokenize(feature["label"])
            
            samples.append({"video": video, "audio": audio, "label": label})
        
        batch = collate_pad(samples)
        
        batch['videos'] = batch['videos'].permute(0, 2, 1, 3, 4)
        batch['audios'] = batch['audios'].permute(0, 2, 1)
        
        return batch