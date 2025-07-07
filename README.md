# AVSRCocktail: Audio-Visual Speech Recognition for Cocktail Party Scenarios

**Official implementation** of "[Cocktail-Party Audio-Visual Speech Recognition](https://arxiv.org/abs/2506.02178)" (Interspeech 2025).

A robust audio-visual speech recognition system designed for multi-speaker environments and noisy cocktail party scenarios. The model combines lip reading and audio processing to achieve superior performance in challenging acoustic conditions with background noise and speaker interference.

## Getting Started

### Sections
1. <a href="#install">Installation</a>
2. <a href="#evaluation">Evaluation</a>
3. <a href="#training">Training</a>

## <a id="install">1. Installation </a>

Following this steps:

```sh
# Clone the baseline code repo
git clone https://github.com/nguyenvulebinh/AVSRCocktail.git
cd AVSRCocktail

# Create Conda environment
conda create --name AVSRCocktail python=3.11
conda activate AVSRCocktail

# Install FFmpeg, if it's not already installed.
conda install ffmpeg

# Install dependencies
pip install -r requirements.txt
```

## <a id="evaluation">2. Evaluation</a>

The evaluation script `script/evaluation.py` provides comprehensive evaluation capabilities for the AVSR Cocktail model on multiple datasets with various noise conditions and interference scenarios.

### Quick Start

**Basic evaluation on LRS2 test set:**
```sh
python script/evaluation.py --model_type avsr_cocktail --dataset_name lrs2 --set_id test
```

**Evaluation on AVCocktail dataset:**
```sh
python script/evaluation.py --model_type avsr_cocktail --dataset_name AVCocktail --set_id video_0
```

### Supported Datasets

#### 1. LRS2 Dataset
Evaluate on the LRS2 dataset with various noise conditions:

**Available test sets:**
- `test`: Clean test set
- `test_snr_n5_interferer_1`: SNR -5dB with 1 interferer
- `test_snr_n5_interferer_2`: SNR -5dB with 2 interferers  
- `test_snr_0_interferer_1`: SNR 0dB with 1 interferer
- `test_snr_0_interferer_2`: SNR 0dB with 2 interferers
- `test_snr_5_interferer_1`: SNR 5dB with 1 interferer
- `test_snr_5_interferer_2`: SNR 5dB with 2 interferers
- `test_snr_10_interferer_1`: SNR 10dB with 1 interferer
- `test_snr_10_interferer_2`: SNR 10dB with 2 interferers
- `*`: Evaluate on all test sets and report average WER

**Example:**
```sh
# Evaluate on clean test set
python script/evaluation.py --model_type avsr_cocktail --dataset_name lrs2 --set_id test

# Evaluate on noisy conditions
python script/evaluation.py --model_type avsr_cocktail --dataset_name lrs2 --set_id test_snr_0_interferer_1

# Evaluate on all conditions
python script/evaluation.py --model_type avsr_cocktail --dataset_name lrs2 --set_id "*"
```

#### 2. AVCocktail Dataset
Evaluate on the AVCocktail cocktail party dataset:

**Available video sets:**
- `video_0` to `video_50`: Individual video sessions
- `*`: Evaluate on all video sessions and report average WER

The evaluation reports WER for three different chunking strategies:
- `asd_chunk`: Chunks based on Active Speaker Detection
- `fixed_chunk`: Fixed-duration chunks
- `gold_chunk`: Ground truth optimal chunks

**Example:**
```sh
# Evaluate on specific video
python script/evaluation.py --model_type avsr_cocktail --dataset_name AVCocktail --set_id video_0

# Evaluate on all videos
python script/evaluation.py --model_type avsr_cocktail --dataset_name AVCocktail --set_id "*"
```

### Configuration Options

#### Model Configuration
- `--model_type`: Model architecture to use (use `avsr_cocktail` for the AVSR Cocktail model)
- `--checkpoint_path`: Path to custom model checkpoint (default: uses pretrained `nguyenvulebinh/AVSRCocktail`)
- `--cache_dir`: Directory to cache downloaded models (default: `./model-bin`)

#### Processing Parameters  
- `--max_length`: Maximum length of video segments in seconds (default: 15)
- `--beam_size`: Beam size for beam search decoding (default: 3)

#### Dataset Parameters
- `--dataset_name`: Dataset to evaluate on (`lrs2` or `AVCocktail`)
- `--set_id`: Specific subset to evaluate (see dataset-specific options above)

#### Output Options
- `--verbose`: Enable verbose output during processing
- `--output_dir_name`: Name of output directory for session processing (default: `output`)

### Advanced Usage

**Custom model checkpoint:**
```sh
python script/evaluation.py \
    --model_type avsr_cocktail \
    --dataset_name lrs2 \
    --set_id test \
    --checkpoint_path ./model-bin/my_custom_model \
    --cache_dir ./custom_cache
```

**Optimized inference settings:**
```sh
python script/evaluation.py \
    --model_type avsr_cocktail \
    --dataset_name AVCocktail \
    --set_id "*" \
    --max_length 10 \
    --beam_size 5 \
    --verbose
```

### Output Format

The evaluation script outputs Word Error Rate (WER) scores:

**LRS2 evaluation output:**
```
WER test: 0.1234
```

**AVCocktail evaluation output:**
```
WER video_0 asd_chunk: 0.1234
WER video_0 fixed_chunk: 0.1456  
WER video_0 gold_chunk: 0.1123
```

When using `--set_id "*"`, the script reports both individual and average WER scores across all test conditions.

## <a id="training">3. Training</a>

### Model Architecture

- **Encoder**: Pre-trained AV-HuBERT large model (`nguyenvulebinh/avhubert_encoder_large_noise_pt_noise_ft_433h`)
- **Decoder**: Transformer decoder with CTC/Attention joint training
- **Tokenization**: SentencePiece unigram tokenizer with 5000 vocabulary units
- **Input**: Video frames are cropped to the mouth region of interest using a 96 × 96 bounding box, while the audio is sampled at a 16 kHz rate

### Training Data

The model is trained on multiple large-scale datasets that have been preprocessed and are ready for the training pipeline. All datasets are hosted on Hugging Face at [nguyenvulebinh/AVYT](https://huggingface.co/datasets/nguyenvulebinh/AVYT) and include:

| Dataset | Size |
|---------|------|
| **LRS2** | ~145k samples |
| **VoxCeleb2** | ~540k samples |
| **AVYT** | ~717k samples |
| **AVYT-mix** | ~483k samples |

The information about these datasets can be found in the [Cocktail-Party Audio-Visual Speech Recognition](https://arxiv.org/abs/2506.02178) paper.

**Dataset Features:**
- **Preprocessed**: All audio-visual data is pre-processed and ready for direct input to the training pipeline
- **Multi-modal**: Each sample contains synchronized audio and video (mouth crop) data
- **Labeled**: Text transcriptions for supervised learning

The training pipeline automatically handles dataset loading and loads data in [streaming mode](https://huggingface.co/docs/datasets/stream). However, to make training faster and more stable, it's recommended to download all datasets before running the training pipeline. The storage needed to save all datasets is approximately 1.46 TB.

### Training Process

The training script is available at `script/train.py`.

**Multi-GPU Distributed Training:**
```sh
# Set environment variables for distributed training
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run with torchrun for multi-GPU training (using default parameters)
torchrun --nproc_per_node 4 script/train.py

# Run with custom parameters
torchrun --nproc_per_node 4 script/train.py \
    --streaming_dataset \
    --batch_size 6 \
    --max_steps 400000 \
    --gradient_accumulation_steps 2 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --learning_rate 1e-4 \
    --warmup_steps 4000 \
    --checkpoint_name avsr_avhubert_ctcattn \
    --model_name_or_path ./model-bin/avsr_cocktail \
    --output_dir ./model-bin
```

**Model Output:**
The trained model will be saved by default in `model-bin/{checkpoint_name}/` (default: `model-bin/avsr_avhubert_ctcattn/`).

#### Configuration Options

You can customize training parameters using command line arguments:

**Dataset Options:**
- `--streaming_dataset`: Use streaming mode for datasets (default: False)

**Training Parameters:**
- `--batch_size`: Batch size per device (default: 6)
- `--max_steps`: Total training steps (default: 400000)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--warmup_steps`: Learning rate warmup steps (default: 4000)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 2)

**Checkpoint and Logging:**
- `--save_steps`: Checkpoint saving frequency (default: 2000)
- `--eval_steps`: Evaluation frequency (default: 2000)
- `--log_interval`: Logging frequency (default: 25)
- `--checkpoint_name`: Name for the checkpoint directory (default: "avsr_avhubert_ctcattn")
- `--resume_from_checkpoint`: Resume training from last checkpoint (default: False)

**Model and Output:**
- `--model_name_or_path`: Path to pretrained model (default: "./model-bin/avsr_cocktail")
- `--output_dir`: Output directory for checkpoints (default: "./model-bin")
- `--report_to`: Logging backend, "wandb" or "none" (default: "none")

**Hardware Requirements:**
- **GPU Memory**: The default training configuration is designed to fit within **24GB GPU memory**
- **Training Time**: With 2x NVIDIA Titan RTX 24GB GPUs, training takes approximately **56 hours per epoch**
- **Convergence**: **200,000 steps** (total batch size 24) is typically sufficient for model convergence


## Acknowledgement

This repository is built using the [auto_avsr](https://github.com/mpc001/auto_avsr), [espnet](https://github.com/espnet/espnet), and [avhubert](https://github.com/facebookresearch/av_hubert) repositories.

## Contact 

nguyenvulebinh@gmail.com