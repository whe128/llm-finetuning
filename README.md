# Qwen2.5-1.5B Fine-tuning Pipeline

A complete parameter-efficient LLM fine-tuning pipeline for adapting **Qwen2.5-1.5B** to domain-specific technical question answering.

The project focuses specifically on **LLM fine-tuning**, including supervised fine-tuning, LoRA, QLoRA, model inference, automated evaluation, and training resource profiling.

The QA dataset is based on **RAG-related technical concepts**, but the retrieval components themselves are outside the scope of this project.

---

## Overview

This project implements an end-to-end fine-tuning pipeline for adapting a general-purpose language model to a domain-specific QA task.

The pipeline covers:

- Dataset preparation
- Train/test splitting
- Supervised Fine-Tuning (SFT)
- Parameter-Efficient Fine-Tuning (PEFT)
- LoRA
- QLoRA
- 4-bit quantization
- Adapter saving and loading
- Qwen chat-template based inference
- LLM-as-a-Judge evaluation
- Accuracy calculation
- Training time profiling
- GPU memory profiling

The project focuses on the **fine-tuning layer** rather than implementing a complete RAG system.

---

# Pipeline Architecture

```text
                         80 QA Samples
                              │
                    Train / Test Split
                              │
              ┌───────────────┴───────────────┐
              │                               │
          40 Train                         40 Test
              │                               │
              ▼                    ┌──────────┴──────────┐
         SFT Training              │                     │
              │                    │                     │
        ┌─────┴─────┐              ▼                     ▼
        │           │          Base Model          Fine-tuned
        ▼           ▼          Baseline              Models
      LoRA        QLoRA             │              ┌──────┴──────┐
        │           │               │              │             │
        │      4-bit Base           │              ▼             ▼
        │           │               │            LoRA          QLoRA
        └─────┬─────┘               │              │             │
              │                     │              └──────┬──────┘
              └─────────────────────┴─────────────────────┘
                                    │
                                    ▼
                                Inference
                                    │
                                    ▼
                              Generated Answer
                                    │
                                    ▼
                            LLM-as-a-Judge
                                    │
                                    ▼
                                 Accuracy
```

---

# Dataset

The dataset contains **80 technical QA samples**.

The questions are related to concepts encountered in RAG-based LLM applications, making the dataset suitable for domain-specific instruction tuning.

The dataset is split into:

| Split    | Samples | Purpose                |
| -------- | ------: | ---------------------- |
| Training |      40 | Supervised fine-tuning |
| Test     |      40 | Held-out evaluation    |
|          |         |                        |

The 40 test samples are kept separate from training and are used to evaluate all three model configurations.

---

# Fine-Tuning Strategies

## Base Model

The original:

```text
Qwen/Qwen2.5-1.5B
```

is used directly without any fine-tuning.

It provides the baseline performance before domain adaptation.

```text
Qwen2.5-1.5B
      │
      ▼
  Inference
      │
      ▼
  Test Set
      │
      ▼
Evaluation
```

No training time or training GPU memory is reported for the Base Model because it is not trained.

---

## LoRA

LoRA is used for supervised fine-tuning while keeping the original model weights frozen.

Instead of directly updating the original weight matrix:

```text
W
```

LoRA learns a low-rank update:

```text
W' = W + BA
```

where `A` and `B` are trainable low-rank matrices.

This reduces the number of trainable parameters and allows the model to be adapted with significantly lower resource requirements than full fine-tuning.

### Target Modules

The current implementation applies LoRA to:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

### Configuration

```text
r = 16
lora_alpha = 32
lora_dropout = 0.05
learning_rate = 2e-4
epochs = 3
```

The resulting LoRA adapter is saved separately from the base model.

---

## QLoRA

QLoRA combines:

```text
4-bit Quantized Base Model
+
LoRA Adapter
```

The base model is loaded using 4-bit quantization while the LoRA parameters remain trainable.

### Configuration

```text
Quantization: 4-bit
Quantization Type: NF4
Double Quantization: Enabled
Compute Type: BF16
```

The key difference is:

```text
LoRA:

FP16/BF16 Base Model
        +
Trainable LoRA
```

and:

```text
QLoRA:

4-bit Quantized Base Model
        +
Trainable LoRA
```

This reduces the GPU memory required during fine-tuning.

---

# Training Pipeline

The training workflow is:

```text
        40 Training Samples
                │
                ▼
        Dataset Preparation
                │
                ▼
        Qwen Chat Template
                │
                ▼
        Tokenization
                │
                ▼
        Base Model Loading
                │
                ▼
        ┌──────────────┐
        │              │
        ▼              ▼
      LoRA           QLoRA
        │              │
        │        4-bit Quantization
        │              │
        └──────┬───────┘
               ▼
              SFT
               │
               ▼
        Model Training
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
   Training Time   Peak GPU Memory
        │             │
        └──────┬──────┘
               ▼
        Save LoRA Adapter
```

Only the LoRA adapter parameters are updated during training.

The base model remains frozen.

---

# Data Format

Training samples are converted into a conversational format compatible with Qwen's chat template.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is RAG?"
    },
    {
      "role": "assistant",
      "content": "RAG is a system architecture that combines information retrieval with large language model generation."
    }
  ]
}
```

The Qwen tokenizer's chat template is used to construct the training sequence.

The same conversational format is used during inference.

---

# Inference

After training, the LoRA adapter is loaded on top of the original Qwen2.5-1.5B model.

```text
Qwen2.5-1.5B
      +
LoRA Adapter
      │
      ▼
Qwen Chat Template
      │
      ▼
Generation
      │
      ▼
Generated Answer
```

For QLoRA, the quantized base model and LoRA adapter are loaded together.

The Base Model follows the same inference interface without an adapter.

Therefore, the three configurations share the same downstream inference pipeline:

```text
Base
LoRA
QLoRA
```

---

# Evaluation

The evaluation uses the **40 held-out test samples**.

The evaluation pipeline is:

```text
Test Question
      │
      ▼
Model Inference
      │
      ▼
Generated Answer
      │
      ▼
Qwen2.5-1.5B Judge
      │
      ▼
Correct / Incorrect
      │
      ▼
Accuracy
```

A separate `qwen2.5:1.5b` model is used as the LLM judge.

The judge evaluates whether the generated answer correctly addresses the reference answer.

The evaluation records:

- Correct answers
- Incorrect answers
- Judge errors
- Accuracy

---

# Evaluation Results

All models are evaluated on the same **40 held-out test samples**.

| Configuration            | Training | Test Samples | Accuracy | Training Time | Peak GPU Memory |
| ------------------------ | -------- | -----------: | -------: | ------------: | --------------: |
| **Base Qwen2.5-1.5B**    | No       |           40 |  **65%** |           N/A |             N/A |
| **Qwen2.5-1.5B + LoRA**  | Yes      |           40 |  **85%** |      285.06 s |         4.28 GB |
| **Qwen2.5-1.5B + QLoRA** | Yes      |           40 |  **85%** |      151.33 s |         2.29 GB |

### Accuracy Improvement

Fine-tuning improved the test accuracy from:

```text
65% → 85%
```

This corresponds to a **20 percentage-point improvement** over the Base Model.

Both LoRA and QLoRA achieved the same test accuracy under the current configuration.

---

# Resource Profiling

The training pipeline automatically records training time and peak GPU memory.

## Training Time

```text
LoRA       285.06 s
QLoRA      151.33 s
```

Under the current configuration, QLoRA completed training approximately **46.9% faster** than LoRA.

Training time is dependent on the hardware, software environment, sequence length, batch configuration, and other training settings.

---

## Peak GPU Memory

```text
LoRA       4.28 GB
QLoRA      2.29 GB
```

QLoRA reduced measured peak GPU memory by approximately **46.5%** compared with LoRA.

This demonstrates the practical memory savings obtained by quantizing the frozen base model.

---

# Project Structure

```text
qwen-finetune/
│
├── data/
│   ├── train.json
│   └── test.json
│
├── models/
│   ├── lora/
│   └── qlora/
│
├── outputs/
│   ├── lora/
│   └── qlora/
│
├── results/
│   ├── base_generation.json
│   ├── lora_generation.json
│   ├── qlora_generation.json
│   ├── base_evaluation.json
│   ├── lora_evaluation.json
│   └── qlora_evaluation.json
│
├── src/
│   ├── prediction.py
│   ├── evaluation.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

---

# Technology Stack

### Model

- Qwen2.5-1.5B

### Fine-Tuning

- PyTorch
- Hugging Face Transformers
- TRL
- PEFT
- LoRA
- QLoRA
- BitsAndBytes

### Evaluation

- LLM-as-a-Judge
- Qwen2.5-1.5B

### Hardware Acceleration

- CUDA
- NVIDIA GPU

### Language

- Python

---

# Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
torch
transformers
datasets
trl
peft
bitsandbytes
accelerate
```

---

# Usage

## 1. Prepare Dataset

```bash
python src/prepare_data.py
```

This generates:

```text
data/train.json
data/test.json
```

with:

```text
40 training samples
40 test samples
```

---

## 2. Train

```bash
# set the NEED_QUANT: True or False
python src/train.py
```

The LoRA adapter is saved under:

```text
models/lora/  or models/qlora/
```

---

## 3. Model Inference and save result

```bash
# set mode 0: base  1:lora  2:qlora
python src/prediction.py
```

The prediction is saved under:

```text
results/base_generation.json
results/lora_generation.json
results/qlora_generation.json
```

---

## 3. Evaluate the result

```bash
# set mode 0: base  1:lora  2:qlora
python src/evaluation.py
```

The prediction is saved under:

```text
results/base_evaluation.json
results/lora_evaluation.json
results/qlora_evaluation.json
```

---

# Engineering Design

## Parameter-Efficient Fine-Tuning

The project uses PEFT rather than updating all parameters of Qwen2.5-1.5B.

The base model remains frozen, while lightweight LoRA adapters are trained.

This provides several practical advantages:

- Lower GPU memory requirements
- Smaller trainable parameter set
- Smaller adapter checkpoints
- Faster experimentation and iteration
- Ability to maintain multiple domain-specific adapters

---

## Adapter-Based Model Loading

The fine-tuned model is represented as:

```text
Base Model
     +
LoRA Adapter
```

rather than storing a completely independent copy of the model.

This allows the same base model to be reused with different adapters.

For example:

```text
                 Qwen2.5-1.5B
                      │
             ┌────────┼────────┐
             │        │        │
             ▼        ▼        ▼
          No Adapter  LoRA   QLoRA
             │        │        │
             ▼        ▼        ▼
           Base     Domain   Domain
                    Adapter  Adapter
```

---

## Unified Inference Interface

Base, LoRA, and QLoRA use the same high-level inference flow.

This reduces duplicated inference logic and makes it easier to switch between model configurations.

---

## Automated Resource Monitoring

Training time and peak GPU memory are measured automatically during training.

Peak GPU memory is collected through CUDA memory statistics, while training duration is measured using a high-resolution timer.

This allows resource usage to be recorded together with the model's evaluation results.

---

# Results Summary

The current implementation demonstrates that supervised fine-tuning can effectively adapt Qwen2.5-1.5B to the target technical QA domain.

```text
                    Base       LoRA       QLoRA
------------------------------------------------
Training             No         Yes        Yes
Test Accuracy       65%        85%        85%
Training Time        -       285.06s    151.33s
Peak GPU Memory      -        4.28GB     2.29GB
```

Key results:

- Test accuracy increased from **65% to 85%**
- LoRA achieved **85%** test accuracy
- QLoRA achieved **85%** test accuracy
- QLoRA reduced measured peak GPU memory by approximately **46.5%**
- QLoRA training was approximately **46.9% faster under the current configuration**
- The Base Model provides a pre-fine-tuning baseline
- The pipeline automates dataset preparation, SFT, adapter management, inference, evaluation, and resource profiling

---

# Limitations

The current dataset contains 80 QA samples, with 40 samples used for training and 40 held out for testing.

Therefore, the current accuracy results demonstrate the effectiveness of the implemented pipeline on this specific technical QA dataset, rather than serving as a general benchmark for Qwen2.5-1.5B, LoRA, or QLoRA.

The evaluation also uses an LLM-as-a-Judge approach, which introduces dependence on the judge model's evaluation behavior.

A larger production-oriented evaluation could use a larger and more diverse dataset together with additional evaluation methods.

---

# Future Extensions

Potential extensions include:

- Larger domain-specific datasets
- More diverse instruction formats
- More difficult QA samples
- Hyperparameter configuration management
- Multiple LoRA adapter configurations
- Adapter merging
- Quantized inference
- Larger judge models
- Human evaluation
- Additional automatic evaluation metrics
- Integration with an external RAG application

The last item represents **integration with a separate RAG system**, rather than adding retrieval components to this fine-tuning repository.

---
