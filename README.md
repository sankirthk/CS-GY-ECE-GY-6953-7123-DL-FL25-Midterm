# Llama-3.1-8B Fine-Tuning for Math Solution Verification
**NYU Deep Learning Fall 2025 — Kaggle Math Verifier Competition**

---

## Overview
This repository documents a production-ready workflow for fine-tuning **Llama-3.1-8B** with **Unsloth** to verify mathematical solutions. The objective is to classify whether a student-provided answer is correct. The pipeline supports incremental fine-tuning, memory-efficient quantization, LoRA adapters, error analysis, and Kaggle submission generation.

## Table of Contents
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Loading and Configuration](#model-loading-and-configuration)
- [Incremental Fine-Tuning](#incremental-fine-tuning)
- [Training Configuration](#training-configuration)
- [Validation and Error Analysis](#validation-and-error-analysis)
- [Kaggle Submission](#kaggle-submission)
- [Saving and Resuming](#saving-and-resuming)
- [Results and Observations](#results-and-observations)
- [Key Takeaways](#key-takeaways)
- [Citation](#citation)
- [Contributors](#contributors)

---

## Setup

### Prerequisites
- Google Colab with a T4 or A100 GPU
- Google Drive for checkpoint and artifact storage
- Weights & Biases account for experiment tracking

### Installation
Run the following cells in Colab to install dependencies:

```bash
pip install uv
uv pip install unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets pandas tqdm wandb
```

Configure Weights & Biases:

```python
import os
import wandb

wandb.login()

WANDB_PROJECT = "nyu_math_eval_colab_experiment5"
WANDB_ENTITY = "KachraSweep-Colab"
USER_NAME = "Sankirth"

os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_RUN_GROUP"] = "incremental_training"
```

---

## Project Structure
All paths are organized under Google Drive for persistence.

```text
DL_Fall_2025_Kaggle/
├── checkpoints/
│   └── Sankirth/
│       ├── Sankirth_run_0_to_20000/
│       └── Sankirth_run_20000_to_40000/
├── dataset/
│   ├── balanced_dataset/
│   └── train_val_split/
└── results/
    ├── submission_Sankirth_0_to_20000.csv
    ├── prediction_details_Sankirth_0_to_20000.csv
    └── wandb_logs/
```

---

## Data Preparation
1. **Load and balance the dataset**
   - Source dataset: `ad6398/nyu-dl-teach-maths-comp`
   - Compute class distribution and create a balanced subset of 400k samples per class.
   - Persist the balanced dataset to Google Drive.
2. **Create a train/validation split**
   - Reserve 5,000 stratified samples for validation.
   - Use the remaining balanced data for training.
   - Reuse the same validation split across runs for consistent tracking.

---

## Model Loading and Configuration

```python
from unsloth import FastLanguageModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Resume from a checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    "/content/drive/MyDrive/DL_Fall_2025_Kaggle/checkpoints/Sankirth/Sankirth_run_20000_to_40000/final_model",
    load_in_4bit=True,
    device_map="auto",
)
```

---

## Incremental Fine-Tuning
Fine-tuning proceeds in sequential chunks of data to manage memory and track incremental progress. Each run consumes a new slice of the balanced training pool while keeping the validation subset fixed.

Key steps:
- Format prompts and responses for supervised fine-tuning.
- Initialize LoRA adapters and 4-bit quantization to fit training within Colab resources.
- Log each run to Weights & Biases for comparison.

---

## Training Configuration

```python
from trl import SFTTrainer
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir=f"{CHECKPOINT_BASE}/{USER_NAME}_run_{train_start_idx}_to_{train_end_idx}",
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_steps=100,
    weight_decay=0.01,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=True,
    report_to="wandb",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    eval_dataset=formatted_validation_dataset,
    dataset_text_field="text",
    args=args,
)

trainer.train()
```

---

## Validation and Error Analysis
- Evaluate on the fixed validation set using deterministic decoding.
- Track accuracy, false positives/negatives, malformed outputs, and sample-level error analyses.

```python
validation_accuracy = evaluate_accuracy(model, tokenizer, validation_dataset, n=500)
errors = detailed_error_analysis(model, tokenizer, validation_dataset, n=200)
```

---

## Kaggle Submission

```python
from datasets import load_dataset
import pandas as pd

test_dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="test")
predictions = []

for example in test_dataset:
    prompt = inference_prompt.format(example["question"], str(example["solution"]))
    outputs = model.generate(
        **tokenizer([prompt], return_tensors="pt").to("cuda"),
        max_new_tokens=5,
    )
    predictions.append(parse_prediction(outputs, tokenizer))

submission = pd.DataFrame({
    "ID": range(len(predictions)),
    "is_correct": predictions,
})

submission.to_csv("submission_Sankirth_0_to_20000.csv", index=False)
```

Upload the generated CSV to Kaggle for leaderboard evaluation.

---

## Saving and Resuming
After each run:
- Save the final model and tokenizer.
- Record metadata (data range, hyperparameters, metrics) for reproducibility.

```python
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")
```

Resume training by loading the desired checkpoint with `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`.

---

## Results and Observations

| Run | Samples       | Learning Rate       | Validation Accuracy | Notes                      |
| --- | ------------- | ------------------- | ------------------- | -------------------------- |
| 1   | 0–20k         | 1e-4                | ~83%                | Stable initial convergence |
| 2   | 20k–40k       | 1e-4                | ~85%                | Gradual improvement        |
| 3   | 40k–60k       | 1e-4 (cosine decay) | ~87%                | Smoother convergence       |
| …   | …             | …                   | …                   | …                          |

---

## Key Takeaways
- 4-bit quantization enables large-model fine-tuning on Colab GPUs.
- LoRA adapters reduce trainable parameters by ~99% while preserving quality.
- Balanced sampling prevents collapse into predicting a single class.
- Incremental training improves stability and resource utilization.
- Cosine learning-rate scheduling delivers smoother training curves.

---

## Contributors
- Sankirth Kalahasti — sk11617
- Milind Kaushal - mk9694 
- Team KachraSweep — Collaborative development and W&B integration
