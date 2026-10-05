# UniAE-MoE: A Unified Audio Encoder via Mixture of Experts

This repository contains the official implementation for the work "UniAE-MoE: A Unified Audio Encoder via Mixture of Experts".

<p align="center">
  <a href="https://arxiv.org/abs/2609.39199"><img src="https://img.shields.io/badge/arXiv-2609.39199-b31b1b.svg" alt="arXiv"></a>
  <a href="#whats-new"><img src="https://img.shields.io/badge/Venue-NCMMSC%202026%20%28Oral%29-4b8bbe.svg" alt="NCMMSC 2026 Oral"></a>
  <a href="https://huggingface.co/Syclus/UniAE-MoE/tree/main"><img src="https://img.shields.io/badge/Model-Hugging%20Face-yellow.svg" alt="Hugging Face model"></a>
</p>

<img width="1432" alt="abs" src="assets/UniAE-MoE.png">

## Abstract
> Large Audio Language Models (LALMs) rely on effective audio encoders for multi-task performance. We introduce **UniAE-MoE**, a unified audio encoder designed to model cross-domain audio representations and achieve outstanding downstream understanding performance via a Mixture-of-Experts (MoE) architecture. Specifically, we explore mainstream audio encoders and integrate those from Qwen2-Audio and Audio-Flamingo 3, which demonstrate superior downstream capabilities. To facilitate effective model fusion, we improve our encoder using SwiGLU with shared experts to decouple encoder networks, and we further introduce a two-stage instruction-tuning strategy to better adapt the model to diverse downstream tasks. Moreover, we propose the task-specific data scaling (TSDS) technique to enhance UniAE-MoE's understanding capabilities. On the XARES-LLM benchmark, UniAE-MoE attains a score of **0.802**, achieving state-of-the-art performance. It also delivers top-tier performance in the official Interspeech 2026 Audio Encoder Capability Challenge, further demonstrating robust generalization across diverse audio tasks. Together, these results validate the effectiveness of UniAE-MoE for unified audio understanding across speech, music, and general audio domains.

## 🔥What's new

- 🎉 **[2026/09] Our paper has been accepted for an oral presentation at NCMMSC 2026!**
- 🎉 **[2026/03] We won first place 🏆 in Track A and fourth place in Track B of the Interspeech 2026 Audio Encoder Capability Challenge.**


## 🚀 Get Started

### 🔧 Download pretrained models

Our code can automatically download the required pretrained models. If the automatic download fails (e.g., due to network restrictions), you can download the models manually from Hugging Face and place them under the `UniAE-MoE/` directory.

Required models:
- **Qwen2-Audio** (Hugging Face repo: `Qwen/Qwen2-Audio-7B`)  
  https://huggingface.co/Qwen/Qwen2-Audio-7B

- **Flamingo3** (Hugging Face repo: `nvidia/audio-flamingo-3-hf`)  
  https://huggingface.co/nvidia/audio-flamingo-3-hf

- **UniAE-MoE** (Hugging Face repo: `Syclus/UniAE-MoE`)  
  https://huggingface.co/Syclus/UniAE-MoE/tree/main

After downloading, please ensure your local directory structure looks like this:

```text
.../
└── UniAE-MoE/
    ├── Qwen2-Audio-7B/          # from Qwen/Qwen2-Audio-7B
    ├── audio-flamingo-3-hf/     # from nvidia/audio-flamingo-3-hf
    ├── checkpoints/             # from Syclus/UniAE-MoE
    └── ...
```

Alternatively, you can skip downloading the models in advance. The code will automatically download the models to the specified path during evaluation.

### 👀 Environment

### Install ffmpeg

Install ffmpeg to ensure dataset unpacking works properly
```bash
conda install -y -c conda-forge ffmpeg
```

On Ubuntu, you can also use:
```bash
apt install -y ffmpeg
```

### Set up the uv environment

```bash
cd UniAE-MoE
uv sync
source .venv/bin/activate
```

### 🤖 training

Single GPU Training:

```bash
python -m xares_llm.run_encoder_tuning \
    models/uniae/moe_fusion.py --stage full \
    --model_args '{"fusion_mode": "moe_swiglu"}' \
    --config config/train.yaml
```

Multi-GPU Training:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=src accelerate launch \
    --mixed_precision="bf16" \
    --num_processes=3 \
    --main_process_port="29500" \
    -m xares_llm.run_encoder_tuning \
    models/uniae/moe_fusion.py \
    --stage full \
    --model_args '{"fusion_mode": "moe_swiglu"}' \
    --config config/train.yaml
```
### ✏️ Evaluation

```bash
python -m xares_llm.run models/uniae/moe_fusion.py all all \
  --model_args '{
    "fusion_mode": "moe_swiglu",
    "checkpoint_path": "./checkpoints"
    }'
```

The evaluation outputs are saved under:
./experiments/all_train_config/moe_fusion/.

⚠️ **Compatibility Notice**  
The audio-flamingo-3 requires a newer version of the `transformers` library that has not fully compatible with `xares-llm` framework. Due to this incompatibility, training and evaluation cannot be reliably resumed from intermediate checkpoints. If the process is interrupted, please delete all checkpoint files in the output directory and restart from scratch to avoid inconsistent or corrupted states.

⭐️ The `result/` directory stores the results from our own tests for reference.