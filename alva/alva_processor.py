# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Qwen2Audio.
"""

import warnings
from typing import List, Union
import torch

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils.deprecation import deprecate_kwarg
from transformers import LlavaOnevisionVideoProcessor


class AlvaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {},
        "video_kwargs": {}
    }


# def pad_video_repeat_last(video: torch.Tensor, target_len: int) -> torch.Tensor:
#     """
#     Pads video in time to target_len by repeating the last frame.
#     video: [T, C, H, W]
#     target_len: int
#     Returns: [target_len, C, H, W]
#     """
#     T, C, H, W = video.shape
#
#     assert T <= target_len
#
#     if T >= target_len:
#         return video[:target_len]
#     pad_frames = target_len - T
#     last = video[-1:].expand(pad_frames, C, H, W)  # no data copy
#     return torch.cat([video, last], dim=0)


# def pad_video_with_mask(video, target_len):
#     T, C, H, W = video.shape
#     mask = torch.zeros(target_len, dtype=torch.bool)
#     mask[:T] = True
#     return pad_video_repeat_last(video, target_len), mask

class AlvaProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2Audio processor which wraps a Qwen2Audio feature extractor and a Qwen2Audio tokenizer into a single processor.

    [`Qwen2AudioProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2AudioProcessor.__call__`] and [`~Qwen2AudioProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the default chat template
                is used.
        audio_token (`str`, *optional*, defaults to `"<|AUDIO|>"`):
            The token to use for audio tokens.
        audio_bos_token (`str`, *optional*, defaults to `"<|audio_bos|>"`):
            The token to use for audio bos tokens.
        audio_eos_token (`str`, *optional*, defaults to `"<|audio_eos|>"`):
            The token to use for audio eos tokens.
    """

    attributes = ["feature_extractor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template", "audio_token", "audio_bos_token", "audio_eos_token"]
    feature_extractor_class = "WhisperFeatureExtractor"
    video_processor_class = "LlavaOnevisionVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
            self,
            feature_extractor=None,
            tokenizer=None,
            video_processor=None,
            chat_template=None,
            audio_token="<|AUDIO|>",
            audio_bos_token="<|audio_bos|>",
            audio_eos_token="<|audio_eos|>",
            video_token="<|VIDEO|>",
            video_bos_token="<|video_bos|>",
            video_eos_token="<|video_eos|>",
            debug=False
    ):
        if chat_template is None:
            chat_template = self.default_chat_template
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token

        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.video_bos_token = tokenizer.video_bos_token if hasattr(tokenizer, "video_bos_token") else video_bos_token
        self.video_eos_token = tokenizer.video_eos_token if hasattr(tokenizer, "video_eos_token") else video_eos_token

        super().__init__(feature_extractor, tokenizer, video_processor, chat_template=chat_template)

        vocab = self.tokenizer.get_vocab()

        for token in [self.audio_token, self.audio_bos_token, self.audio_eos_token]:
            assert token in vocab

            if debug:
                print(token, vocab[token])

        self.debug = debug
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def pad_video_features(self, features, output_kwargs=None):

        if features is None:
            return None

        lengths = torch.tensor([feature.size(0) for feature in features], dtype=torch.long)
        T_max = int(lengths.max().item())
        B = len(features)
        P, D = features[0].size(1), features[0].size(2)

        dtype = features[0].dtype
        device = features[0].device

        # Preallocate output + mask
        out = torch.zeros((B, T_max, P, D), dtype=dtype, device=device)
        non_blocking = True
        # print("output size", out.size())
        # Fill per-sample (batch size is tiny; this is fine)
        for i, v in enumerate(features):
            v = v.to(device=device, dtype=dtype, non_blocking=non_blocking)
            T = v.size(0)
            out[i, :T].copy_(v)  # copy real frames
            # mask[i, :T] = True
            if T < T_max:
                # Repeat last frame without materializing copies
                last = v[-1:].expand(T_max - T, -1, -1)  # [pad, C, H, W]
                out[i, T:T_max].copy_(last)

        return out

    def get_video_lengths(self, videos,
                          num_av, output_kwargs=None):

        if videos is None:
            return None, [0] * num_av

        non_blocking = True

        assert len(videos) > 0
        # Ensure tensors
        vids = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in videos]
        # Basic sizes
        lengths = torch.tensor([v.shape[0] for v in vids], dtype=torch.long)
        T_max = int(lengths.max().item())
        B = len(vids)
        C, H, W = vids[0].shape[1:]
        dtype = vids[0].dtype
        device = vids[0].device

        # Preallocate output + mask
        out = torch.empty((B, T_max, C, H, W), dtype=dtype, device=device)

        # attention-mask (masked position = 0)
        mask = torch.zeros((B, T_max), dtype=torch.bool, device=device)

        # Fill per-sample (batch size is tiny; this is fine)
        for i, v in enumerate(vids):
            v = v.to(device=device, dtype=dtype, non_blocking=non_blocking)
            T = v.shape[0]
            out[i, :T].copy_(v)  # copy real frames
            mask[i, :T] = True
            if T < T_max:
                # Repeat last frame without materializing copies
                last = v[-1:].expand(T_max - T, -1, -1, -1)  # [pad, C, H, W]
                out[i, T:T_max].copy_(last)

        video_inputs = self.video_processor(out, return_tensors="pt")

        return video_inputs, lengths.tolist()

    def get_audio_lengths(self, audio,
                          num_av, output_kwargs=None):

        if audio is None:

            return None, [0] * num_av

        else:
            output_kwargs["audio_kwargs"]["return_attention_mask"] = True
            output_kwargs["audio_kwargs"]["padding"] = "max_length"
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

            # rename attention_mask to prevent conflicts later on
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()

            return audio_inputs, audio_lengths

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            audio: Union[np.ndarray, List[np.ndarray]] = None,
            video: Union[torch.Tensor, List[torch.Tensor]] = None,
            video_features: Union[torch.Tensor, List[torch.Tensor]] = None,
            **kwargs: Unpack[AlvaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
        """

        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            AlvaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # lets process both video and audio at the same time
        assert audio is not None or video is not None, "Either audio or video must be provided"

        if audio is not None and video is not None:
            # ensure we have as much audios as audio tokens
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)

            num_audios, num_videos = 0, 0

            if audio is not None:
                num_audios = 1 if type(audio) is np.ndarray else len(audio)

            if video is not None:
                num_videos = 1 if (type(video) is torch.Tensor) else len(video)

            if audio is not None and video is not None:
                assert num_videos == num_audios, (
                    f"Expected the number of video and audio tensors to match, "
                    f"but got {num_videos} videos and {num_audios} audios."
                )

            num_av = max(num_videos, num_audios)

            if num_audio_tokens != num_av:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"
                )

            # first, we get the audio features
            # Some kwargs should not be changed so we can expand text with audio tokens below
            # output_kwargs["audio_kwargs"]["return_attention_mask"] = True
            # output_kwargs["audio_kwargs"]["padding"] = "max_length"
            # audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            #
            # # rename attention_mask to prevent conflicts later on
            # audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

            # audio_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()

            audio_inputs, audio_lengths = self.get_audio_lengths(audio, num_av, output_kwargs)

            if video_features is None:
                video_inputs, video_lengths = self.get_video_lengths(video, num_av, output_kwargs)
            else:
                video_inputs, video_lengths = None, None

            expanded_text = []
            for sample in text:

                replace_str = []
                while self.audio_token in sample:

                    # if audio_length = 0, the floor division will give us 0 tokens
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    num_audio_tokens = (input_length - 2) // 2 + 1

                    # num_video_tokens = num_video_frames = video_lengths.pop(0)
                    #
                    # if audio is not None and video is not None:
                    #     assert num_audio_tokens == num_video_tokens, (
                    #         f"Expected the number of audio and video tokens to match, "
                    #         f"but got {num_audio_tokens} audio tokens and {num_video_tokens} video tokens."
                    #     )
                    num_video_tokens = 0

                    # select the non-zero tokens (audio or video)
                    num_av_tokens = max(num_audio_tokens, num_video_tokens)

                    expanded_audio_token = self.audio_token * num_av_tokens

                    audio_token_start_idx = sample.find(self.audio_token)
                    audio_token_end_idx = audio_token_start_idx + len(self.audio_token)

                    has_bos = (
                            sample[audio_token_start_idx - len(self.audio_bos_token): audio_token_start_idx]
                            == self.audio_bos_token
                    )
                    has_eos = (
                            sample[audio_token_end_idx: audio_token_end_idx + len(self.audio_eos_token)]
                            == self.audio_eos_token
                    )

                    # Check if this audio token is surrounded by bos/eos tokens
                    if not has_bos and not has_eos:
                        expanded_audio_token = self.audio_bos_token + expanded_audio_token + self.audio_eos_token

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, inputs, modalities=["audio"])

        # we don't have to update video_inputs, they will be handled during the tokenizer
        if audio is not None:
            inputs.update(audio_inputs)

        if video_inputs is not None:
            inputs.update(video_inputs)

        if video_features is not None:
            assert isinstance(video_features, list)
            inputs["video_features"] = self.pad_video_features(video_features)
        else:
            inputs["video_features"] = None

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))

    @property
    # NOTE: we don't have default templates anymore, and the below is kept only because the hub config is not yet updated!
    def default_chat_template(self):
        """
        This default vicuna template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content is a list of strings and audios.
        * If the content element is an audio, the template will output a sequence of <|AUDIO|> tokens

        Example:

        ```python
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
                {"type": "text", "text": "What's that sound?"},
            ]},
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"},
                {"type": "text", "text": "How about this one?"},
            ]},
        ]

        result = template.render(messages=messages, add_generation_prompt=True)
        ```
        """
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
            "{% if loop.first and message['role'] != 'system' %}"
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "{% endif %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}<|im_end|>\n"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if 'audio' in content or 'audio_url' in content or message['type'] == 'audio' %}"
            "{% set audio_count.value = audio_count.value + 1 %}"
            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            "{% elif 'text' in content %}"
            "{{ content['text'] }}"
            "{% endif %}"
            "{% endfor %}"
            "<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on


__all__ = ["AlvaProcessor"]

if __name__ == "__main__":

    # TODO: create AVProcessor

    default_whisper = "openai/whisper-large-v3"
    # default_llama = "meta-llama/Meta-Llama-3-8B"
    default_llama = "llama-extended"
    default_qwen2 = "Qwen/Qwen2-Audio-7B"
    default_llava = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    import datasets

    cache_dir = "data-bin/cache"

    from transformers import (AutoTokenizer, AutoProcessor, LlavaOnevisionProcessor,
                              AutoConfig, LlavaOnevisionForConditionalGeneration)
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    from transformers.models.llava_onevision.configuration_llava_onevision import LlavaOnevisionConfig
    from src.dataset.avhubert_dataset import load_video_rgb, load_audio, cut_or_pad
    from script.compute_llava_features import process_sample_id, load_npz_zstd
    import os

    root_dir = "./LLAVA-ONE-features-3x3/"

    # we need to create 1 tokenizer, 1 audio frame extractor, 1 video feature extractor

    tokenizer = AutoTokenizer.from_pretrained(default_llama,
                                              cache_dir=cache_dir)

    qwen2_processor = AutoProcessor.from_pretrained(default_qwen2, cache_dir=cache_dir)

    feature_extractor = qwen2_processor.feature_extractor

    processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                                                        cache_dir=cache_dir)

    video_processor = processor.video_processor

    alva_processor = AlvaProcessor(feature_extractor, tokenizer, video_processor)

    print("Created ALVA processor successfully!")

    test_dataset = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=False,
                                         cache_dir=cache_dir).remove_columns(['__key__', '__url__'])['train']

    ds_name = "lrs2"
    split_name = "train"
    test_dataset = test_dataset.add_column("dataset_name", [ds_name] * len(test_dataset))
    test_dataset = test_dataset.add_column("split_name", [split_name] * len(test_dataset))

    iterator = iter(test_dataset)

    video_rgbs = []
    texts = []
    audios = []
    features = []

    prompt_template = "<|video_bos|><|VIDEO|><|video_eos|><|audio_bos|><|AUDIO|><|audio_eos|>Transcribe this speech:"
    rate_ratio = 640

    # the data collator will be written off this test

    for i in range(2):
        sample = next(iterator)
        # print(sample.keys())
        video_array = sample['video']
        video_rgb = load_video_rgb(video_array)
        audio = load_audio(video_array).squeeze()

        audio = cut_or_pad(audio, len(video_rgb) * rate_ratio).numpy()

        video_rgbs.append(video_rgb)
        audios.append(audio)

        text = sample["label"].decode(encoding="utf-8")
        text = prompt_template + "<|begin_of_text|>" + text.lower() + tokenizer.eos_token

        texts.append(text)

        hashed_sample_id = process_sample_id(sample["sample_id"])
        ds_name = sample["dataset_name"]
        split = sample["split_name"]

        output_path = os.path.join(root_dir, ds_name, split, hashed_sample_id)
        feats = load_npz_zstd(output_path, "tokens_fp16")
        feats = torch.from_numpy(feats).to(torch.bfloat16)

        features.append(feats)

    inputs = alva_processor(texts, audios, video_rgbs, features,
                            return_tensors="pt", padding=True)

    # for key in inputs:
    #     print(key)
    # print(inputs["pixel_values_videos"])
    # print(inputs)
    for key in inputs:
        print(key, inputs[key].size())

    feature_attention_mask = inputs["feature_attention_mask"]
    feature_lengths = torch.sum(feature_attention_mask.long(), dim=1)

    print(feature_lengths)
    # pixels = inputs["pixel_values_videos"]

    # print(type(pixels))
    # print(pixels.size())

    print("Tested ALVA processor successfully!")

    alva_processor.save_pretrained("alva-base")
    #
    # llava_config = LlavaOnevisionConfig.from_pretrained(default_llava, cache_dir=cache_dir)
    #
    # # print(type(llava_config))
    # vision_config = llava_config.vision_config
    # print(vision_config)
    #
    # vision_model = SiglipVisionModel(vision_config)
    #
    # print(vision_model)

    # llm = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
    #                                                              cache_dir=cache_dir,
    #                                                              torch_dtype=torch.bfloat16,
    #                                                              device_map="auto"
    #                                                              )
    #
    # vision_tower = llm.model.vision_tower
    # print(vision_tower)
