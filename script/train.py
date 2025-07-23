import os
import sys

os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import torch
from datasets import load_from_disk
from src.dataset.avhubert_dataset import load_audio, load_video, cut_or_pad, AudioTransform, VideoTransform, \
    DataCollator
from src.tokenizer.spm_tokenizer import TextTransform
from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from transformers import TrainingArguments
from src.custom_trainer import AVSRTrainer
from transformers.trainer_utils import IntervalStrategy

import safetensors.torch
import datasets
import time
import argparse

from optimized.modules import replace_layernorm_with_memory_efficient


try:
    from torchsummary import summary
except ImportError:
    from torchinfo import summary

os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"  # seconds
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ['WANDB_PROJECT'] = 'mcorec'

world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")

import logging
from transformers import modeling_utils

_FLASH_ATTN = "flash_attention_2"
_EAGER_ATTN = "eager"


def skip_check_and_enable_flash_attn_2(cls, config, **kwargs):
    # TODO: add option to switch
    # config._attn_implementation = "eager"
    config._attn_implementation = _FLASH_ATTN


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


if not is_main_process():
    logging.basicConfig(level=logging.ERROR)
    # Also set loggers for popular libs
    for name in ['transformers', 'datasets', 'torch', 'accelerate']:
        logging.getLogger(name).setLevel(logging.ERROR)

if not is_main_process():
    sys.stdout = open(os.devnull, "w")


# NCCL_DEBUG=WARN OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 script/train.py \
# --streaming_dataset \
# --include_mcorec \
# --batch_size 6 \
# --max_steps 400000 \
# --gradient_accumulation_steps 2 \
# --save_steps 2000 \
# --eval_steps 2000 \
# --log_interval 25 \
# --learning_rate 1e-4 \
# --warmup_steps 4000 \
# --checkpoint_name mcorec_finetuning \
# --model_name_or_path ./model-bin/avsr_cocktail \
# --output_dir ./model-bin


def load_avsr_dataset(cache_dir='data-bin/cache', include_mcorec=False, streaming=False):
    # streaming=True to avoid downloading all dataset at once, but it can be crash if network is unstable
    # streaming=False to download all dataset at once, it take time and around 1.5TB disk space. More stable.

    def format_sample(sample):
        sample['label'] = str(sample['label'], encoding='utf-8')
        sample['length'] = int(sample['length'])
        sample['sample_id'] = str(sample['sample_id'], encoding='utf-8')
        return sample

    # Load dataset
    finished_loading = False
    try_times = 0
    max_try_times = 5

    while not finished_loading:
        try:
            # Load dataset. It's quite bigdataset and sometime downloading can break. You can simple retry.
            lrs2 = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", streaming=streaming,
                                         cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            vox2 = datasets.load_dataset("nguyenvulebinh/AVYT", "vox2", streaming=streaming,
                                         cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            avyt = datasets.load_dataset("nguyenvulebinh/AVYT", "avyt", streaming=streaming,
                                         cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            avyt_mix = datasets.load_dataset("nguyenvulebinh/AVYT", "avyt-mix", streaming=streaming,
                                             cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            # Load mcorec dataset. Ensure you have permission to use this dataset.
            if include_mcorec:
                print("Loading MCoRec dataset")
                mcorec_dataset = datasets.load_dataset("MCoRecChallenge/MCoRec", streaming=streaming,
                                                       cache_dir=cache_dir).remove_columns(['__key__', '__url__'])
            finished_loading = True
        except Exception as e:
            try_times += 1
            if try_times >= max_try_times:
                raise e
            time.sleep(10)

    if not streaming:
        # That mean above datasets are already downloaded and cached
        list_datasets = [lrs2, vox2, avyt, avyt_mix]
        if include_mcorec:
            list_datasets.append(mcorec_dataset)
        for ds in list_datasets:
            for split in ds.keys():
                split_size = len(ds[split])
                if split_size > 10000:
                    num_shards = max(20, split_size // 10000)
                else:
                    num_shards = 1
                ds[split] = ds[split].to_iterable_dataset(num_shards=num_shards)
                print(f"Split {split} has {split_size} samples and {ds[split].num_shards} shards")

    if include_mcorec:
        map_dataset_probabilities = {
            "lrs2": 0.25,
            "vox2": 0.10,
            "avyt": 0.20,
            "avyt-mix": 0.25,
            "mcorec": 0.2,
        }
    else:
        map_dataset_probabilities = {
            "lrs2": 0.3,
            "vox2": 0.2,
            "avyt": 0.25,
            "avyt-mix": 0.25,
        }

    map_datasets = {
        "lrs2": {
            "probabilities": map_dataset_probabilities["lrs2"],
            "dataset": {
                "train": datasets.concatenate_datasets([
                    lrs2["train"],
                    lrs2["pretrain"]
                ]),
                "valid": datasets.concatenate_datasets([
                    lrs2["valid"],
                    lrs2["test_snr_0_interferer_2"]
                ]) if not include_mcorec else None
            },
        },
        "vox2": {
            "probabilities": map_dataset_probabilities["vox2"],
            "dataset": {
                "train": vox2["dev"],
                "valid": None,
            },
        },
        "avyt": {
            "probabilities": map_dataset_probabilities["avyt"],
            "dataset": {
                "train": datasets.concatenate_datasets([
                    avyt['talking'],
                    avyt['silent']
                ]),
                "valid": None,
            },
        },
        "avyt-mix": {
            "probabilities": map_dataset_probabilities["avyt-mix"],
            "dataset": {
                "train": avyt_mix["train"],
                "valid": avyt_mix["test"] if not include_mcorec else None,
            },
        },
        "mcorec": {
            "probabilities": map_dataset_probabilities["mcorec"] if include_mcorec else 0,
            "dataset": {
                "train": mcorec_dataset["train"] if include_mcorec else None,
                "valid": mcorec_dataset["valid"] if include_mcorec else None,
            },
        }
    }
    print("map_datasets\n", map_datasets)

    train_dataset = datasets.interleave_datasets(
        [item['dataset']['train'] for item in map_datasets.values() if item['dataset']['train'] is not None],
        seed=11,
        probabilities=[item['probabilities'] for item in map_datasets.values() if item['dataset']['train'] is not None],
        stopping_strategy='all_exhausted')
    valid_dataset = datasets.interleave_datasets(
        [item['dataset']['valid'] for item in map_datasets.values() if item['dataset']['valid'] is not None],
        stopping_strategy='first_exhausted')

    train_dataset = train_dataset.map(format_sample)
    valid_dataset = valid_dataset.map(format_sample)

    # load lrs2 for interference speech
    # interference_speech = None
    print("Loading interference speech dataset. Actual file around 10GB need to download. This may take a while...")
    interference_speech = datasets.load_dataset("nguyenvulebinh/AVYT", "lrs2", cache_dir=cache_dir,
                                                data_files='lrs2/lrs2-train-*.tar').remove_columns(
        ['__key__', '__url__'])['train']
    return train_dataset, valid_dataset, interference_speech


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--streaming_dataset", action="store_true", default=False)
    parser.add_argument("--include_mcorec", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=400000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--checkpoint_name", type=str, default="avsr_avhubert_ctcattn")
    parser.add_argument("--model_name_or_path", type=str,
                        default="./model-bin/avsr_cocktail")  # Or None to train from scratch
    parser.add_argument("--report_to", type=str, default="none")  # wandb or none
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), f"model-bin"))

    parser.add_argument("--cache_dir", type=str,
                        default="data-bin/cache")  # Or None to train from scratch

    parser.add_argument('--no_progress_bar', action='store_true',
                        help="Disable progress bar (useful for SLURM runs)")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--bf16", action="store_true", default=False)

    args = parser.parse_args()

    if args.attn_implementation == _FLASH_ATTN:
        modeling_utils.PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(skip_check_and_enable_flash_attn_2)

    streaming_dataset = True if args.streaming_dataset else False
    include_mcorec = True if args.include_mcorec else False
    batch_size = args.batch_size
    max_steps = args.max_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    log_interval = args.log_interval
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    resume_from_checkpoint = True if args.resume_from_checkpoint else False
    checkpoint_name = args.checkpoint_name
    model_name_or_path = args.model_name_or_path  # Or None to train from scratch
    output_dir = os.path.join(args.output_dir, checkpoint_name)
    report_to = args.report_to

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load text transform
    sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "src/tokenizer/spm/unigram/unigram5000.model")
    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "src/tokenizer/spm/unigram/unigram5000_units.txt")
    text_transform = TextTransform(
        sp_model_path=sp_model_path,
        dict_path=dict_path,
    )

    # Load from pretrained checkpoint
    if model_name_or_path is not None and os.path.exists(model_name_or_path):
        print("Loading pretrained model from", model_name_or_path)
        avsr_model = AVHubertAVSR.from_pretrained(model_name_or_path)
    else:
        # Load from scratch
        print("Loading model from scratch")
        avsr_config = AVHubertAVSRConfig(odim=len(text_transform.token_list),
                                         attn_implementation=args.attn_implementation)
        avsr_model = AVHubertAVSR(avsr_config)

        # Load pretrained encoder checkpoint
        encoder_pretrained_checkpoint = "nguyenvulebinh/avhubert_encoder_large_noise_pt_noise_ft_433h"  # AVHubert encoder original (https://facebookresearch.github.io/av_hubert/)
        print("Loading pretrained encoder from", encoder_pretrained_checkpoint)
        encoder_pretrained = avsr_model.avsr.encoder.from_pretrained(
            encoder_pretrained_checkpoint,
            cache_dir=args.cache_dir,
            # cache_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model-bin")
        )
        avsr_model.avsr.encoder.load_state_dict(encoder_pretrained.state_dict())

    # TODO: add extension installation script
    replace_layernorm_with_memory_efficient(avsr_model)

    # Load dataset
    train_dataset, valid_dataset, interference_dataset = load_avsr_dataset(cache_dir=args.cache_dir,
                                                                           streaming=streaming_dataset,
                                                                           include_mcorec=include_mcorec)

    train_av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=AudioTransform(subset="train", speech_dataset=interference_dataset),
        video_transform=VideoTransform(subset="train"),
    )
    valid_av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=AudioTransform(subset="test"),
        video_transform=VideoTransform(subset="test"),
    )

    print("train_dataset\n", train_dataset)
    print("valid_dataset\n", valid_dataset)
    summary(avsr_model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "log"),
        # group_by_length=True,
        # length_column_name='length',
        label_names=["labels"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # auto_find_batch_size = True,
        # max_grad_norm=0.1,
        eval_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        metric_for_best_model='loss',
        greater_is_better=False,
        fp16=not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataloader_num_workers=10,
        # save_only_model=True, # WARNING: this will save only model and not optimizer, scheduler, etc.
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=log_interval,
        learning_rate=learning_rate,
        weight_decay=0.005,
        warmup_steps=warmup_steps,
        save_total_limit=500,
        ignore_data_skip=True,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        # save_safetensors=False,
        report_to=report_to,  # enable logging to W&B,
        # report_to="none",
        run_name=checkpoint_name,  # name of the W&B run (optional)
        accelerator_config={
            "dispatch_batches": False
        },
        disable_tqdm=args.no_progress_bar
        # dispatch_batches=False
        # ddp_find_unused_parameters=True
    )

    trainer = AVSRTrainer(
        model=avsr_model,
        data_collator=train_av_data_collator,
        valid_data_collator=valid_av_data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    if not resume_from_checkpoint:
        trainer.train()
    else:
        print("Resuming from checkpoint")
        trainer.train(resume_from_checkpoint=True)
