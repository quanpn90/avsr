import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import argparse
import json
import math
import glob
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset
import torchvision
import torchaudio
import datasets
from jiwer import wer
from src.tokenizer.norm_text import norm_string
import tempfile
import webvtt
from eval_util import split_iterable_dataset

# Add src to path
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.tokenizer.spm_tokenizer import TextTransform
from src.talking_detector.segmentation import segment_by_asd
from src.cluster.conv_spks import (
    get_speaker_activity_segments,
    calculate_conversation_scores,
    cluster_speakers,
    get_clustering_f1_score
)


class BaseInferenceModel(ABC):
    """Abstract base class for all inference models"""

    def __init__(self, checkpoint_path=None, cache_dir=None, beam_size=3, worker_id=0):
        self.model = None
        self.text_transform = None
        self.av_data_collator = None
        self.beam_search = None
        self.tokenizer = None
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir or "./model-bin"
        self.beam_size = beam_size
        self.worker_id = worker_id

        # bind the inference model into the gpu
        torch.cuda.set_device(self.worker_id)

    @abstractmethod
    def load_model(self):
        """Load the specific model architecture"""
        pass

    @abstractmethod
    def inference(self, videos, audios, **kwargs):
        """Perform inference on audio-visual data"""
        pass

    def get_tokenizer_paths(self):
        """Get paths for tokenizer files"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        sp_model_path = os.path.join(base_dir, "src/tokenizer/spm/unigram/unigram5000.model")
        dict_path = os.path.join(base_dir, "src/tokenizer/spm/unigram/unigram5000_units.txt")
        return sp_model_path, dict_path


class AVSRCocktailModel(BaseInferenceModel):
    """AVSR Cocktail model implementation"""

    def load_model(self):
        from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, DataCollator
        from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
        from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig

        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )

        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")

        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,
        )

        # Load model
        model_path = self.checkpoint_path or "nguyenvulebinh/AVSRCocktail"
        print(f"Loading model from {model_path}")
        avsr_model = AVHubertAVSR.from_pretrained(model_path, cache_dir=self.cache_dir)
        avsr_model.eval().cuda()
        # device = torch.device(f"cuda:{self.worker_id}")
        self.model = avsr_model.avsr
        self.beam_search = get_beam_search_decoder(self.model, self.text_transform.token_list, beam_size=self.beam_size)

    def inference(self, videos, audios, **kwargs):
        avhubert_features = self.model.encoder(
            input_features=audios,
            video=videos,
        )
        audiovisual_feat = avhubert_features.last_hidden_state
        audiovisual_feat = audiovisual_feat.squeeze(0)

        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted


class AutoAVSRModel(BaseInferenceModel):
    """Auto AVSR model implementation"""

    def load_model(self):
        from src.dataset.av_dataset import AudioTransform, VideoTransform, DataCollator
        from src.auto_avsr.configuration_avsr import AutoAVSRConfig
        from src.auto_avsr.avsr_model import AutoAVSR, get_beam_search_decoder

        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )

        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")

        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,ffffffffff
        )

        # Load model    
        avsr_config = AutoAVSRConfig()
        avsr_model = AutoAVSR(avsr_config)
        ckpt_path = self.checkpoint_path or "./model-bin/auto_avsr/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
        print(f"Loading model from {ckpt_path}")
        pretrained_weights = torch.load(ckpt_path, weights_only=True)
        avsr_model.avsr.load_state_dict(pretrained_weights)
        avsr_model.eval().cuda()
        self.model = avsr_model.avsr
        self.beam_search = get_beam_search_decoder(self.model, self.text_transform.token_list, beam_size=self.beam_size)

    def inference(self, videos, audios, **kwargs):
        video_feat, _ = self.model.encoder(videos, None)
        audio_feat, _ = self.model.aux_encoder(audios, None)
        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        audiovisual_feat = audiovisual_feat.squeeze(0)

        nbest_hyps = self.beam_search(audiovisual_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[:min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted


class MuAViCModel(BaseInferenceModel):
    """MuAViC model implementation"""

    def load_model(self):
        from src.dataset.avhubert_dataset import AudioTransform, VideoTransform, DataCollator
        from src.avhubert_muavic.avhubert2text import AV2TextForConditionalGeneration
        from transformers import Speech2TextTokenizer

        # Load text transform
        sp_model_path, dict_path = self.get_tokenizer_paths()
        self.text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )

        # Load data collator
        audio_transform = AudioTransform(subset="test")
        video_transform = VideoTransform(subset="test")

        self.av_data_collator = DataCollator(
            text_transform=self.text_transform,
            audio_transform=audio_transform,
            video_transform=video_transform,
        )

        # Load model
        model_name = self.checkpoint_path or 'nguyenvulebinh/AV-HuBERT-MuAViC-en'
        print(f"Loading model from {model_name}")
        self.model = AV2TextForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.tokenizer = Speech2TextTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.model = self.model.cuda().eval()

    def inference(self, videos, audios, **kwargs):
        attention_mask = torch.BoolTensor(audios.size(0), audios.size(-1)).fill_(False).cuda()
        output = self.model.generate(
            audios,
            attention_mask=attention_mask,
            video=videos,
        )
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].upper()
        return output


class InferenceEngine:
    """Main inference engine that handles model selection and processing"""

    def __init__(self,
                 model_type: str,
                 checkpoint_path=None,
                 cache_dir=None,
                 beam_size=3,
                 max_length=15,
                 worker_id=0
                 ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir
        self.beam_size = beam_size
        self.max_length = max_length
        self.worker_id = worker_id
        self.model_impl = self._get_model_implementation()


    def _get_model_implementation(self) -> BaseInferenceModel:
        """Factory method to get the appropriate model implementation"""
        if self.model_type == "avsr_cocktail":
            return AVSRCocktailModel(self.checkpoint_path, self.cache_dir, self.beam_size, worker_id=self.worker_id)
        elif self.model_type == "auto_avsr":
            return AutoAVSRModel(self.checkpoint_path, self.cache_dir, self.beam_size, worker_id=self.worker_id)
        elif self.model_type == "muavic_en":
            return MuAViCModel(self.checkpoint_path, self.cache_dir, self.beam_size, worker_id=self.worker_id)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def load_model(self):
        """Load the selected model"""
        print(f"Loading {self.model_type} model...")
        self.model_impl.load_model()
        print(f"{self.model_type} model loaded successfully!")

    def chunk_video(self, video_path, asd_path=None, max_length=15):
        """Split video into chunks for inference"""
        if asd_path is not None:
            with open(asd_path, "r") as f:
                asd = json.load(f)

            # Convert frame numbers to integers and sort them
            frames = sorted([int(f) for f in asd.keys()])
            # Find the minimum frame number to normalize frame indices
            min_frame = min(frames)

            segments_by_frames = segment_by_asd(asd, {
                "max_chunk_size": max_length,  # in seconds
            })
            # Normalize frame indices, for inference, don't care about the actual frame indices
            segments = [((seg[0] - min_frame) / 25, (seg[-1] - min_frame) / 25) for seg in segments_by_frames]

        else:
            # Get video duration
            audio, rate = torchaudio.load(video_path)
            video_duration = audio.shape[1] / rate
            # num chunks
            num_chunks = math.ceil(video_duration / max_length)
            chunk_size = math.ceil(video_duration / num_chunks)
            segments = []
            # Convert to integer steps for range
            steps = int(video_duration * 100)  # Convert to centiseconds for precision
            step_size = int(chunk_size * 100)
            for i in range(0, steps, step_size):
                start_time = i / 100
                end_time = min((i + step_size) / 100, video_duration)
                segments.append((start_time, end_time))

        return segments

    def format_vtt_timestamp(self, timestamp):
        """Format timestamp for VTT output"""
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        milliseconds = int((timestamp - int(timestamp)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def infer_processed_sample(self, video):
        sample = {
            "video": video
        }
        sample_features = self.model_impl.av_data_collator([sample])
        audios = sample_features["audios"].cuda()
        videos = sample_features["videos"].cuda()
        audio_lengths = sample_features["audio_lengths"].cuda()
        video_lengths = sample_features["video_lengths"].cuda()

        try:
            output = self.model_impl.inference(videos, audios)
        except Exception as e:
            print(f"Error during inference for segment {sample}")
            raise e

        return output

    def infer_video(self, video_path, asd_path=None, offset=0., desc=None):
        """Perform inference on a video file"""
        segments = self.chunk_video(video_path, asd_path, max_length=self.max_length)
        segment_output = []

        for seg in tqdm(segments, desc="Processing segments" if desc is None else desc, total=len(segments)):
            # Prepare sample
            sample = {
                "video": video_path,
                "start_time": seg[0],
                "end_time": seg[1],
            }
            sample_features = self.model_impl.av_data_collator([sample])
            audios = sample_features["audios"].cuda()
            videos = sample_features["videos"].cuda()
            audio_lengths = sample_features["audio_lengths"].cuda()
            video_lengths = sample_features["video_lengths"].cuda()

            try:
                output = self.model_impl.inference(videos, audios)
            except Exception as e:
                print(f"Error during inference for segment {sample}")
                raise e

            segment_output.append(output)

            # GPU Memory Cleanup
            del audios, videos, audio_lengths, video_lengths, sample_features
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return [
            {
                "start_time": seg[0] + offset,
                "end_time": seg[1] + offset,
                "text": output
            } for seg, output in zip(segments, segment_output)
        ]

    def mcorec_session_infer(self, session_dir, output_dir):
        """Process a complete MCoReC session"""
        # Load session metadata
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Process speaker clustering
        speaker_segments = {}
        for speaker_name, speaker_data in metadata.items():
            list_tracks_asd = []
            for track in speaker_data['central']['crops']:
                list_tracks_asd.append(os.path.join(session_dir, track['asd']))
            uem_start = speaker_data['central']['uem']['start']
            uem_end = speaker_data['central']['uem']['end']
            speaker_activity_segments = get_speaker_activity_segments(list_tracks_asd, uem_start, uem_end)
            speaker_segments[speaker_name] = speaker_activity_segments

        scores = calculate_conversation_scores(speaker_segments)
        clusters = cluster_speakers(scores, list(speaker_segments.keys()))
        output_clusters_file = os.path.join(output_dir, "speaker_to_cluster.json")
        with open(output_clusters_file, "w") as f:
            json.dump(clusters, f, indent=4)

            # Process speaker transcripts
        for speaker_name, speaker_data in tqdm(metadata.items(), desc="Processing speakers", total=len(metadata)):
            print()
            speaker_track_hypotheses = []
            for idx, track in enumerate(speaker_data['central']['crops']):
                video_path = os.path.join(session_dir, track['lip'])
                asd_path = os.path.join(session_dir, track['asd']) if 'asd' in track else None
                with open(os.path.join(session_dir, track['crop_metadata']), "r") as f:
                    crop_metadata = json.load(f)
                track_start_time = crop_metadata['start_time']
                hypotheses = self.infer_video(video_path, asd_path, offset=track_start_time,
                                              desc=f"Processing speaker {speaker_name} track {idx + 1} of {len(speaker_data['central']['crops'])}")
                speaker_track_hypotheses.extend(hypotheses)

                # GPU Memory Cleanup after each track
                torch.cuda.empty_cache()

            output_file = os.path.join(output_dir, f"{speaker_name}.vtt")
            with open(output_file, "w") as f:
                f.write("WEBVTT\n\n")
                for hyp in speaker_track_hypotheses:
                    text = hyp["text"].strip().replace("<unk>", "").strip()
                    start_time = self.format_vtt_timestamp(hyp["start_time"])
                    end_time = self.format_vtt_timestamp(hyp["end_time"])
                    if len(text) == 0:
                        continue
                    f.write(f"{start_time} --> {end_time}\n{text}\n\n")


def _eval_lrs2_single_worker(args, dataset, worker_id, result_queue):

    print(dataset, flush=True)
    engine = InferenceEngine(args.model_type, args.checkpoint_path,
                             args.cache_dir, args.beam_size, args.max_length,
                             worker_id=worker_id)
    engine.load_model()

    output_list = []
    label_list = []

    for sample in tqdm(dataset, desc="Processing samples", position=worker_id):
        label = str(sample['label'], encoding='utf-8')
        label_norm = norm_string(label.replace("<unk>", ""))

        video = sample['video']

        output = engine.infer_processed_sample(video)
        output_norm = norm_string(output.replace("<unk>", ""))

        output_list.append(output_norm)
        label_list.append(label_norm)

    result_queue.put((worker_id, output_list, label_list))


def eval_lrs2(args, dataset):
    num_gpus = torch.cuda.device_count()

    # Shared queue for results
    result_queue = mp.Queue()

    if num_gpus > 1:
        dataset_chunks = split_iterable_dataset(dataset, num_gpus)
        # Spawn processes
        processes = []
        for gpu_id in range(num_gpus):
            process = mp.Process(target=_eval_lrs2_single_worker,
                                 args=(args, dataset_chunks[gpu_id], gpu_id,
                                       result_queue))
            process.start()
            processes.append(process)

    else:
        _eval_lrs2_single_worker(args, dataset, 0,
                                 result_queue)

    all_results = list()

    # stack all datas into the result
    for _ in range(num_gpus):
        all_results.append(result_queue.get())

    # Sort by worker_id
    all_results.sort(key=lambda x: x[0])

    # Concatenate outputs and labels in the original order
    ordered_outputs = []
    ordered_labels = []
    for _, outputs, labels in all_results:
        ordered_outputs.extend(outputs)
        ordered_labels.extend(labels)

    wer_score = wer(reference=ordered_labels, hypothesis=ordered_outputs)
    # print(f"WER: {wer_score}")
    return wer_score


def eval_avcocktail(engine, video_dataset, label_dataset, set_name=None):
    wer_scores = {}
    # Process label
    label_list = []
    label_start_time = []
    start_time = None
    end_time = None
    with tempfile.NamedTemporaryFile(suffix=".vtt") as temp_file:
        with open(temp_file.name, "w") as f:
            f.write(str(label_dataset['label'][0], encoding='utf-8'))
        for utt_id, caption in enumerate(webvtt.read(temp_file.name)):
            if caption.text == "":
                continue
            label_list.append(caption.text)
            caption_start_time = caption.start_time.hours * 3600 + caption.start_time.minutes * 60 + caption.start_time.seconds + caption.start_time.milliseconds / 1000
            caption_end_time = caption.end_time.hours * 3600 + caption.end_time.minutes * 60 + caption.end_time.seconds + caption.end_time.milliseconds / 1000
            if start_time is None:
                start_time = caption_start_time
            if end_time is None:
                end_time = caption_end_time
            if caption_start_time < start_time:
                start_time = caption_start_time
            if caption_end_time > end_time:
                end_time = caption_end_time
            label_start_time.append(caption_start_time)
        # sort label list by start time
        label_list = [label for _, label in sorted(zip(label_start_time, label_list))]
        label_text = norm_string(" ".join(label_list))

    for chunk_type in ['asd_chunk', 'fixed_chunk', 'gold_chunk']:
        # Process video
        output_list = []
        output_list_start_time = []
        for sample in tqdm(video_dataset[chunk_type], desc=f"Processing {chunk_type} {set_name}",
                           total=len(video_dataset[chunk_type])):
            # for seg_path in list_segments:
            seg_start_time = float(str(sample['start_time'], encoding='utf-8'))
            seg_end_time = float(str(sample['end_time'], encoding='utf-8'))
            if seg_start_time + 1 < start_time or seg_end_time - 1 > end_time:
                continue
            output = engine.infer_processed_sample(sample['video'])
            output_list.append(output)
            output_list_start_time.append(seg_start_time)
        output_list = [output for _, output in sorted(zip(output_list_start_time, output_list))]
        output_text = norm_string(" ".join(output_list).replace("<unk>", ""))
        spk_wer = wer(reference=label_text, hypothesis=output_text)
        wer_scores[chunk_type] = spk_wer
    return wer_scores, len(label_text.split())


def main():
    parser = argparse.ArgumentParser(description="Unified inference script for multiple AVSR models")

    # Model selection argument
    parser.add_argument(
        '--model_type',
        type=str,
        # required=True,
        default='avsr_cocktail',
        choices=['avsr_cocktail', 'auto_avsr', 'muavic_en'],
        help='Type of model to use for inference'
    )

    # Input/output arguments
    parser.add_argument(
        '--dataset_name',
        type=str,
        # required=True, 
        default='lrs2',
        choices=['lrs2', 'AVCocktail'],
        help='Path to folder containing session data (supports glob patterns with *)'
    )

    # Input/output arguments
    parser.add_argument(
        '--set_id',
        type=str,
        # required=True, 
        default='*',
        choices=['test', 'test_snr_n5_interferer_1', 'test_snr_n5_interferer_2', 'test_snr_0_interferer_1',
                 'test_snr_0_interferer_2', 'test_snr_5_interferer_1', 'test_snr_5_interferer_2',
                 'test_snr_10_interferer_1', 'test_snr_10_interferer_2'] + [f'video_{i}' for i in range(0, 51)],
        help='Path to folder containing session data (supports glob patterns with *)'
    )

    # Model checkpoint arguments
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to model checkpoint or pretrained model name'
    )

    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./model-bin',
        help='Directory to cache downloaded models (default: ./model-bin)'
    )

    # Optional arguments    _
    parser.add_argument(
        '--max_length',
        type=int,
        default=15,
        help='Maximum length of video segments in seconds (default: 15)'
    )

    parser.add_argument(
        '--beam_size',
        type=int,
        default=3,
        help='Beam size for beam search decoding (default: 3)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--output_dir_name',
        type=str,
        default='output',
        help='Name of the output directory within each session (default: output)'
    )

    args = parser.parse_args()

    # Initialize inference engine


    if args.dataset_name == 'lrs2':
        if args.set_id not in ['test', 'test_snr_n5_interferer_1', 'test_snr_n5_interferer_2',
                               'test_snr_0_interferer_1', 'test_snr_0_interferer_2', 'test_snr_5_interferer_1',
                               'test_snr_5_interferer_2', 'test_snr_10_interferer_1', 'test_snr_10_interferer_2', '*']:
            raise ValueError(
                f"Invalid set_id: {args.set_id} for LRS2 dataset. Only test and test_snr_n5_interferer_1, test_snr_n5_interferer_2, test_snr_0_interferer_1, test_snr_0_interferer_2, test_snr_5_interferer_1, test_snr_5_interferer_2, test_snr_10_interferer_1, test_snr_10_interferer_2, * are supported")
        if args.set_id == '*':
            list_wer_scores = []
            for set_id in ['test', 'test_snr_n5_interferer_1', 'test_snr_n5_interferer_2', 'test_snr_0_interferer_1',
                           'test_snr_0_interferer_2', 'test_snr_5_interferer_1', 'test_snr_5_interferer_2',
                           'test_snr_10_interferer_1', 'test_snr_10_interferer_2']:
                print(f"Inferring {args.dataset_name}/{set_id} sessions using {args.model_type} model")
                dataset = datasets.load_dataset("nguyenvulebinh/AVYT", args.dataset_name, cache_dir='./data-bin/cache',
                                                streaming=False)[set_id]
                wer_score = eval_lrs2(args, dataset)
                list_wer_scores.append(wer_score)
                print(f"WER {set_id}: {wer_score:.4f}")
            print(f"Average WER: {sum(list_wer_scores) / len(list_wer_scores):.4f}")
            print("HELLO1")
        else:
            print(f"Inferring {args.dataset_name}/{args.set_id} sessions using {args.model_type} model")
            dataset = datasets.load_dataset("nguyenvulebinh/AVYT", args.dataset_name, cache_dir='./data-bin/cache',
                                            streaming=True)[args.set_id]

            print("HELLO2")
            wer_score = eval_lrs2(args, dataset)
            print(f"WER: {wer_score}")
    elif args.dataset_name == 'AVCocktail':
        engine = InferenceEngine(args.model_type, args.checkpoint_path, args.cache_dir, args.beam_size, args.max_length)
        engine.load_model()

        if args.set_id not in [f'video_{i}' for i in range(0, 51)] + ['*']:
            raise ValueError(
                f"Invalid set_id: {args.set_id} for AVCocktail dataset. Only video_i for i in range(0, 51) and * are supported")
        if args.set_id == '*':
            list_wer_scores = {}
            for set_id in [f'video_{i}' for i in range(0, 51)]:
                print(f"Inferring {args.dataset_name}/{set_id} sessions using {args.model_type} model")
                video_dataset = datasets.load_dataset("nguyenvulebinh/AVCocktail", set_id, cache_dir='./data-bin/cache')
                label_dataset = \
                    datasets.load_dataset("nguyenvulebinh/AVCocktail", 'labels', cache_dir='./data-bin/cache')[set_id]
                wer_scores, num_words = eval_avcocktail(engine, video_dataset, label_dataset, set_name=set_id)
                for chunk_type, wer_score in wer_scores.items():
                    if chunk_type not in list_wer_scores:
                        list_wer_scores[chunk_type] = []
                    list_wer_scores[chunk_type].extend([wer_score] * num_words)
                    print(f"WER {set_id} {chunk_type}: {wer_score:.4f}")
            for chunk_type, wer_scores in list_wer_scores.items():
                print(f"Average WER {chunk_type}: {sum(wer_scores) / len(wer_scores):.4f}")
        else:
            engine = InferenceEngine(args.model_type, args.checkpoint_path, args.cache_dir, args.beam_size,
                                     args.max_length)
            engine.load_model()

            print(f"Inferring {args.dataset_name}/{args.set_id} sessions using {args.model_type} model")
            video_dataset = datasets.load_dataset("nguyenvulebinh/AVCocktail", args.set_id,
                                                  cache_dir='./data-bin/cache')
            label_dataset = datasets.load_dataset("nguyenvulebinh/AVCocktail", 'labels', cache_dir='./data-bin/cache')[
                args.set_id]
            wer_scores, _ = eval_avcocktail(engine, video_dataset, label_dataset)
            for chunk_type, wer_score in wer_scores.items():
                print(f"WER {chunk_type}: {wer_score:.4f}")


if __name__ == "__main__":
    main()
