---
license: cc-by-nc-sa-4.0
language:
- en
base_model:
- unsloth/Phi-3-medium-4k-instruct
tags:
- unsloth
- phi-3
- lora
- krishna
- gita
- spiritual-ai
- conversational
library_name: unsloth
pipeline_tag: text-generation
---

# Krishnai-V2: A Gita-Inspired AI Persona 🙏

**Model ID:** `sincro420/Krishnai-V2-LoRa`

This repository contains **Krishnai-V2 (LoRA Adapters Only)**, an AI persona fine-tuned to emulate the teachings and conversational style of Lord Krishna, primarily based on the wisdom of the Bhagavad Gita.

**Playground and Tryout:** A playground test website of the standalone model is currently under development. **Meanwhile, you can try out the model using this [Google Colab link](https://colab.research.google.com/drive/1XY4kW-CpTFm-PEf-jx2ttdKmHa8Hu4tq?usp=sharing).**

This model is fine-tuned from `unsloth/Phi-3-medium-4k-instruct` (14B parameters) using a custom dataset and optimized training techniques provided by Unsloth.

**Version Note:** This is V2 of the Krishnai project. It utilizes the larger Phi-3-medium model to overcome persona consistency issues ("narcissistic god complex") observed in the initial V1 prototype (based on Phi-3-mini), resulting in significantly improved qualitative performance.

## Model Description

*   **Purpose:** To provide users with an accessible, interactive way to engage with the wisdom of the Bhagavad Gita through a conversational AI embodying Krishna's persona. It aims to offer guidance, perspective, and solace, acting as a supplementary resource for personal reflection.
*   **Base Model:** `unsloth/Phi-3-medium-4k-instruct`
*   **Fine-tuning Data:** A custom dataset of examples, meticulously curated by combining dialogues from the Bhagavad Gita, hypothetical user queries with Krishna-style responses (manually created by a team of 6), and augmented with a general Gita dataset.
*   **Training Method:** Supervised Fine-Tuning (SFT) using Parameter-Efficient Fine-Tuning (PEFT - LoRA) with optimizations from the Unsloth library. Trained with 4-bit quantization. The merged version is not publicly available; this is only the LoRA model under a non-commercial license (CC-NC-SA-4.0).

## How to Use

This model follows the Alpaca instruction format. You **must** format your prompt accordingly for optimal results.

**Important:** Ensure you have `unsloth`, `torch`, `transformers`, `trl`, `peft`, `accelerate`, and `bitsandbytes` installed. For faster inference and lower VRAM usage, using `unsloth` for loading is recommended.

```python
# Installation For Google Colab:
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers
from unsloth import FastLanguageModel
import torch
# Installation For Google Kaggle:
%%capture
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth

# --- Configuration ---
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Use the same length used during training
dtype = None # Auto detection
load_in_4bit = True # Use 4bit quantization (optional: lower VRAM usage, faster inference; But slightly lower accuracy, may not be supported on all hardware)

# --- Load Model using Unsloth ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "sincro420/Krishnai-V2-LoRa",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model)

# --- Define the Alpaca Prompt Template ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- Prepare Input ---
instruction = "I am feeling lost and stressed about my future. What should I do?" # You can fill in your custom instruction here
input_context = "" # Provide context if needed, otherwise leave empty

inputs = tokenizer(
[
    alpaca_prompt.format(
        instruction,    # Instruction
        input_context,  # Input
        "",             # Output - leave empty for generation!
    )
], return_tensors = "pt").to("cuda")

# --- Generate Response ---
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

```
## Limitations and Bias

*   **Not a Replacement for Professional Help:** This model is intended for informational and reflective purposes only. It is **not** a substitute for professional mental health therapy or qualified spiritual counseling.
*   **Potential Inaccuracies:** Like all language models, Krishnai-V2 may generate responses that are factually incorrect, philosophically inconsistent, or lack common sense.
*   **Limited Scope:** The model's knowledge is primarily derived from its base training and the specific fine-tuning dataset. It may not accurately represent all nuances or interpretations of the Bhagavad Gita or Hindu philosophy.
*   **Bias:** The model may reflect biases present in its training data.
*   **Persona Imperfections:** While significantly improved over V1, the emulation of Krishna's persona may still be imperfect or inconsistent at times.
*   **Use Discretion:** Users should engage with the model critically and exercise their own judgment regarding the guidance provided.

## Training Data

The model was fine-tuned on a custom dataset of **2391 examples**. This dataset was created by:
1.  Manually crafting 1126 examples based on dialogues from the Bhagavad Gita and hypothetical user scenarios relevant to modern life, with responses written in Krishna's persona.
2.  Augmenting this with a general dataset on the Gita found online.
3.  Formatting all examples using the Alpaca instruction structure (`instruction`, `input`, `response`).

## Training Procedure

*   **Framework:** PyTorch
*   **Libraries:** `unsloth`, `transformers`, `trl` (SFTTrainer), `peft`, `accelerate`, `bitsandbytes`
*   **Base Model:** `unsloth/Phi-3-medium-4k-instruct`
*   **Technique:** LoRA (`r=8`, `lora_alpha=16`, target modules included attention and MLP layers)
*   **Optimization:**
    *   4-bit quantization (QLoRA-style)
    *   Unsloth optimizations (incl. gradient checkpointing)
    *   AdamW 8-bit optimizer
*   **Hyperparameters:** `learning_rate=2e-4`, `batch_size=4`, `gradient_accumulation=4` (effective batch 16), `epochs=3`, `lr_scheduler=linear`, `warmup_steps=20`, `max_seq_length=2048`.
*   **Hardware:** Trained primarily on Google Colab (T4 GPU) and Kaggle (T4/P100 GPU), facing VRAM and CPU RAM challenges, especially during model merging for the 14B parameter model.

## Evaluation

Evaluation was primarily **qualitative**. Sample prompts covering various topics (greetings, conceptual questions, personal dilemmas) were used to assess:
*   Persona Consistency (Alignment with Krishna's tone and style)
*   Relevance and Coherence of responses
*   Philosophical Accuracy (Alignment with Gita teachings)
*   Absence of the "narcissistic god complex" observed in V1.

Krishnai-V2 showed significant improvements across these qualitative dimensions compared to the pervious prototype.