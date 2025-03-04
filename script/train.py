import os
import sys
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from datasets import load_from_disk
from src.dataset.avhubert_dataset import load_audio, load_video, cut_or_pad, AudioTransform, VideoTransform, DataCollator
from src.tokenizer.spm_tokenizer import TextTransform
from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from transformers import TrainingArguments
from src.custom_trainer import AVSRTrainer
from transformers.trainer_utils import IntervalStrategy
from torchsummary import summary
import safetensors.torch

# NCCL_DEBUG=WARN OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3,5 torchrun --nproc_per_node 2 train.py
# os.environ['WANDB_PROJECT'] = 'avsr'

if __name__ == "__main__":
    # Load text transform
    sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram_vn/unigram2048.model")
    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram_vn/unigram2048_units.txt")
    text_transform = TextTransform(
        sp_model_path=sp_model_path,
        dict_path=dict_path,
    )
    
    avsr_config = AVHubertAVSRConfig(odim=len(text_transform.token_list))
    avsr_model = AVHubertAVSR(avsr_config)
    
    
    # Load pretrained checkpoint
    # encoder_pretrained_checkpoint = "nguyenvulebinh/avhubert_encoder_large_muavic_en" # AVHubert encoder from muavic-en (https://github.com/facebookresearch/muavic?tab=readme-ov-file#models)
    # encoder_pretrained_checkpoint = "nguyenvulebinh/avhubert_encoder_large_muavic_ar_de_el_es_fr_it_pt_ru" # AVHubert encoder from muavic multilingual (https://github.com/facebookresearch/muavic?tab=readme-ov-file#models)
    encoder_pretrained_checkpoint = "nguyenvulebinh/avhubert_encoder_large_noise_pt_noise_ft_433h" # AVHubert encoder original (https://facebookresearch.github.io/av_hubert/)
        
    encoder_pretrained = avsr_model.avsr.encoder.from_pretrained(
        encoder_pretrained_checkpoint, 
        cache_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model-bin")
    )
    avsr_model.avsr.encoder.load_state_dict(encoder_pretrained.state_dict())
    
    
    
    # Load dataset
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/dummy")
    metadata_path = os.path.join(raw_data_path, "metadata")
    dataset = load_from_disk(metadata_path).map(lambda x: {"video": os.path.join(raw_data_path, x["video"])})
    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]
    
    
    train_av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=AudioTransform(subset="train", speech_dataset=train_dataset),
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
    
    ############ Debugging ############
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # avsr_model.eval().cuda()
    # batch_size = 2
    # batch_samples = []
    # for i in range(len(train_dataset)):
    #     batch_samples.append(train_dataset[i])
    #     if len(batch_samples) == batch_size:
    #         break
    # features = valid_av_data_collator(batch_samples)
    # for key in features:
    #     if isinstance(features[key], torch.Tensor):
    #         if key in ["videos", "audios"]:
    #             features[key] = features[key]
    #         features[key] = features[key].cuda()
    # output = avsr_model(**features)
    # print(output)
    # exit()
    ##################################

    
    batch_size = 2
    max_steps = 200000
    gradient_accumulation_steps = 2
    save_steps = 2000
    eval_steps = 2000
    log_interval = 25
    learning_rate = 1e-4
    warmup_steps = 4000
    checkpoint_name = "avhubert_avvn_noisy"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"model-bin/{checkpoint_name}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "log"),
        group_by_length=True,
        length_column_name='length',
        label_names = ["labels"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # auto_find_batch_size = True,
        # max_grad_norm=0.1,
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        max_steps = max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        metric_for_best_model='loss',
        greater_is_better=False,
        # bf16=True,
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
        save_total_limit=50,
        ignore_data_skip=True,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        # save_safetensors=False,
        # report_to="wandb",  # enable logging to W&B,
        report_to="none",
        run_name=checkpoint_name,  # name of the W&B run (optional)
        accelerator_config={
            "dispatch_batches": False
        }
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
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    
    