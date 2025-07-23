import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoProcessor, AutoConfig, Qwen2AudioProcessor
from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, load_audio, load_video
from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Python <3.12
import warnings

from transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessorKwargs

class Qwen2AudioVideoProcessor(Qwen2AudioProcessor):
    valid_kwargs = ["chat_template",
                    "audio_token", "audio_bos_token", "audio_eos_token",
                    "video_token", "video_bos_token", "video_eos_token"]

    def __init__(
            self,
            feature_extractor=None,
            tokenizer=None,
            chat_template=None,
            audio_token="<|AUDIO|>",
            audio_bos_token="<|audio_bos|>",
            audio_eos_token="<|audio_eos|>",
            video_token="<|VIDEO|>",
            video_bos_token="<|video_bos|>",
            video_eos_token="<|video_eos|>",
    ):
        super().__init__(feature_extractor,
                         tokenizer=tokenizer,
                         chat_template=chat_template,
                         audio_token=audio_token,
                         audio_bos_token=audio_bos_token,
                         audio_eos_token=audio_eos_token)

        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.video_bos_token = tokenizer.video_bos_token if hasattr(tokenizer, "video_bos_token") else video_bos_token
        self.video_eos_token = tokenizer.video_eos_token if hasattr(tokenizer, "video_eos_token") else video_eos_token

        vocab = self.tokenizer.get_vocab()

        for token in [self.video_token, self.video_bos_token, self.video_eos_token]:
            assert token in vocab

    def get_video_lengths(self, videos):

        if type(videos) is torch.Tensor:
            videos = [videos]

        lengths = list()

        for video in videos:
            # decoder = VideoDecoder(video)
            # num_frames = decoder.metadata.num_frames
            # print("num frames:", num_frames)
            num_frames = video.shape[0]
            lengths.append(num_frames)

        return lengths

    @override
    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            audio: Union[np.ndarray, List[np.ndarray]] = None,
            video: Union[torch.Tensor, List[torch.Tensor]] = None,
            audios=None,  # kept for BC
            **kwargs: Unpack[Qwen2AudioProcessorKwargs],
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

        # Handle BC when user passes deprecated keyword argument
        if audios is not None and audio is None:
            audio = audios
            warnings.warn(
                "You may have used the keyword argument for the `audio` inputs. It is strongly recommended to pass inputs with keyword arguments "
                "with keys `audio` and `text`. From transformers v4.55 `audio` will be the only acceptable keyword argument.",
                FutureWarning,
            )

        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            Qwen2AudioProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            # ensure we have as much audios as audio tokens
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = 1 if type(audio) is np.ndarray else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token{'s' if num_audio_tokens > 1 else ''} in provided text but received {num_audios} audio{'s' if num_audios > 1 else ''}"
                )

            # Some kwargs should not be changed so we can expand text with audio tokens below
            output_kwargs["audio_kwargs"]["return_attention_mask"] = True
            output_kwargs["audio_kwargs"]["padding"] = "max_length"
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

            # rename attention_mask to prevent conflicts later on
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

            expanded_text = []
            audio_lengths = audio_inputs["feature_attention_mask"].sum(-1).tolist()

            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    num_audio_tokens = (input_length - 2) // 2 + 1

                    expanded_audio_token = self.audio_token * num_audio_tokens

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

        # if we send the video data and if the text prompt has <|VIDEO|>
        if video is not None:
            # ensure we have as much audios as audio tokens
            num_video_tokens = sum(sample.count(self.video_token) for sample in text)

            # TODO: add option to check if its either bytes or path
            num_videos = 1 if (type(video) is torch.Tensor) else len(video)
            if num_video_tokens != num_videos:
                raise ValueError(
                    f"Found {num_video_tokens} {self.video_token} token{'s' if num_video_tokens > 1 else ''} "
                    f"in provided text but received {num_videos} audio{'s' if num_videos > 1 else ''}"
                )

            # Some kwargs should not be changed so we can expand text with audio tokens below
            # output_kwargs["audio_kwargs"]["return_attention_mask"] = True
            # output_kwargs["audio_kwargs"]["padding"] = "max_length"
            # audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            video_lengths = self.get_video_lengths(video)

            # rename attention_mask to prevent conflicts later on
            # audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

            expanded_text = []

            for sample in text:
                replace_str = []
                while self.video_token in sample:

                    # the number of frames is the number of tokens
                    video_length = video_lengths.pop(0)
                    num_video_tokens = video_length

                    expanded_video_token = self.video_token * num_video_tokens

                    video_token_start_idx = sample.find(self.video_token)
                    video_token_end_idx = video_token_start_idx + len(self.video_token)

                    has_bos = (
                            sample[video_token_start_idx - len(self.video_bos_token): video_token_start_idx]
                            == self.video_bos_token
                    )
                    has_eos = (
                            sample[video_token_end_idx: video_token_end_idx + len(self.video_eos_token)]
                            == self.video_eos_token
                    )

                    # Check if this video token is surrounded by bos/eos tokens
                    if not has_bos and not has_eos:
                        expanded_video_token = self.video_bos_token + expanded_video_token + self.video_eos_token

                    replace_str.append(expanded_video_token)
                    sample = sample.replace(self.video_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, inputs, modalities=["audio", "video"])

        if audio is not None:
            inputs.update(audio_inputs)
        # we will update the video inputs later in the data collator

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)

    @classmethod
    def from_qwen2_processor(cls, processor, **kwargs):
        """
        Instantiate AudioVideoProcessor from an existing AudioProcessor.
        """
        feature_extractor = getattr(processor, "feature_extractor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        chat_template = getattr(processor, "chat_template", None)
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        audio_bos_token = getattr(processor, "audio_bos_token", "<|audio_bos|>")
        audio_eos_token = getattr(processor, "audio_eos_token", "<|audio_eos|>")
        # Add any video kwargs from **kwargs or use defaults
        return cls(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            audio_bos_token=audio_bos_token,
            audio_eos_token=audio_eos_token,
            **kwargs
        )
