# Model Weights Information

## Overview

This repository contains fine-tuned weights for the **Qwen2.5-1.5B-Instruct** model using LoRA (Low-Rank Adaptation) technique.

## Base Model

- **Name:** Qwen/Qwen2.5-1.5B-Instruct
- **Size:** 1.5 Billion parameters
- **Architecture:** Transformer-based causal language model
- **Developer:** Alibaba Cloud (Qwen Team)
- **Base Capabilities:** Instruction-following, chat, Q&A

## Fine-tuned Model Weights

### 1. qwen_lora_adapter
**Type:** Basic LoRA SFT Adapter
**Training Method:** Supervised Fine-Tuning (SFT)
**Weight File:** `adapter_model.safetensors` (32.8 MB)

**LoRA Configuration:**
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target Modules: q_proj, v_proj (attention layers)

**Use Case:** General purpose fine-tuned model for improved instruction following.

---

### 2. qwen_lora_sft (Checkpoint 39)
**Type:** LoRA SFT Training Checkpoint
**Training Method:** Supervised Fine-Tuning (SFT)
**Weight File:** `checkpoint-39/adapter_model.safetensors`

**Additional Files:**
- Training state (optimizer, scheduler)
- Trainer state (metrics, training progress)
- RNG state (for reproducibility)

**Use Case:** More advanced SFT model, likely trained longer or with different data than the basic adapter.

---

### 3. dpo_model (Checkpoint 3)
**Type:** DPO Fine-tuned Model
**Training Method:** Direct Preference Optimization (DPO)
**Weight File:** `checkpoint-3/adapter_model.safetensors`

**DPO Advantages:**
- Trained on preference data (chosen vs rejected responses)
- Better alignment with human preferences
- Improved response quality without explicit reward model

**Use Case:** Model optimized for generating preferred responses based on human feedback.

---

### 4. qwen_lora_dpo_adapter
**Type:** LoRA DPO Adapter
**Training Method:** Direct Preference Optimization with LoRA
**Weight File:** `adapter_model.safetensors`

**Use Case:** Lightweight DPO-optimized adapter for preference-aligned responses.

---

## LoRA Technical Details

### What is LoRA?
Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that:
- Freezes the original model weights
- Adds small trainable matrices to attention layers
- Reduces trainable parameters by >99%
- Maintains model quality while being memory-efficient

### LoRA Formula
For a weight matrix W, LoRA adds: ΔW = BA
Where:
- B is dimension d × r (rank r)
- A is dimension r × k
- r << min(d, k) (r=16 in this case)

### Weight Files Structure

Each adapter contains:
```
adapter_model.safetensors  # LoRA weight matrices (A and B)
adapter_config.json        # LoRA configuration
tokenizer files           # Tokenizer for text processing
```

### Memory Footprint

| Component | Size |
|-----------|------|
| Base Model (full) | ~3 GB |
| LoRA Adapter | ~33 MB |
| **Total** | **~3 GB** |

The LoRA adapters are extremely small compared to full fine-tuning, which would require storing an entire 3GB copy of the model for each variant.

---

## Training Framework

**Library:** PEFT (Parameter-Efficient Fine-Tuning) v0.18.0
**Training Framework:** TRL (Transformer Reinforcement Learning) v0.26.2
**Base Framework:** Transformers v4.57.3, PyTorch v2.9.0

---

## How the Weights Work

### Loading Process
1. Load base Qwen2.5-1.5B-Instruct model
2. Load LoRA adapter weights (A and B matrices)
3. Option A: Keep separate (faster switching)
4. Option B: Merge into base model (faster inference)

### Inference
During inference, the model computes:
```
output = BaseModel(input) + LoRA_Adapter(input)
```

Or if merged:
```
output = MergedModel(input)
```

---

## Choosing Which Model to Use

| Model | Best For |
|-------|----------|
| **qwen_lora_adapter** | General purpose, basic fine-tuning |
| **qwen_lora_sft** | More refined instruction following |
| **dpo_model** | Human-aligned, preferred responses |
| **qwen_lora_dpo_adapter** | Lightweight human-aligned model |
| **Base Model** | Original capabilities without fine-tuning |

---

## Technical Specifications

### Supported Data Types
- FP32 (CPU)
- FP16 (GPU, recommended)
- BF16 (if supported by hardware)

### Inference Requirements
- **Minimum:** 8GB RAM (CPU)
- **Recommended:** 6GB+ VRAM (GPU)
- **Optimal:** 8GB+ VRAM for batch processing

### Context Length
- Maximum sequence length: 32,768 tokens (inherited from base model)
- Practical limit depends on available memory

---

## File Formats

### SafeTensors (.safetensors)
- Safe, fast serialization format for ML tensors
- More secure than pickle-based formats
- Faster loading times
- Memory-mapped for efficient access

### Configuration (.json)
- Human-readable JSON format
- Contains hyperparameters and metadata
- Required for proper model loading

---

## Version Compatibility

Ensure compatibility between:
- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.57.0
- PEFT >= 0.18.0

---

## Additional Resources

### Papers
- **Qwen2.5:** https://qwenlm.github.io/
- **LoRA:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **DPO:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)

### Documentation
- Transformers: https://huggingface.co/docs/transformers
- PEFT: https://huggingface.co/docs/peft
- TRL: https://huggingface.co/docs/trl
