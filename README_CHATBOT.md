# Fine-tuned Qwen Chatbot

This chatbot application uses the fine-tuned Qwen2.5-1.5B-Instruct model with LoRA adapters.

## Model Information

**Base Model:** Qwen/Qwen2.5-1.5B-Instruct

**Available Fine-tuned Adapters:**
1. **qwen_lora_adapter** - Basic LoRA SFT (Supervised Fine-Tuning) Adapter
2. **qwen_lora_sft/checkpoint-39** - LoRA SFT Checkpoint 39
3. **dpo_model/checkpoint-3** - DPO (Direct Preference Optimization) Model Checkpoint 3
4. **qwen_lora_dpo_adapter** - LoRA DPO Adapter

**LoRA Configuration:**
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target Modules: q_proj, v_proj
- PEFT Type: LORA

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch transformers peft accelerate
```

## Usage

### Running the Chatbot

```bash
python chatbot.py
```

### Model Selection

When you run the chatbot, you'll be prompted to select which fine-tuned model to use:

```
Available Fine-tuned Models:
--------------------------------------------------
1. Basic LoRA SFT Adapter
2. LoRA SFT Checkpoint 39
3. DPO Model Checkpoint 3
4. LoRA DPO Adapter
5. Base Model (No Adapter)
--------------------------------------------------

Select a model (1-5) [default: 2]:
```

### Chatbot Commands

Once the chatbot is running:
- Type your questions naturally
- **quit** or **exit** - Close the chatbot
- **reset** - Clear conversation history and start fresh
- **help** - Display available commands

### Example Session

```
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence...

You: Can you explain it more simply?
Assistant: Sure! Think of machine learning as teaching computers to learn...

You: reset
Conversation history cleared.

You: quit
Goodbye!
```

## Model Details

### Training Information

These models were fine-tuned using:
- **Framework:** TRL (Transformer Reinforcement Learning)
- **Techniques:** SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization)
- **Library:** PEFT (Parameter-Efficient Fine-Tuning)

### Hardware Requirements

- **Minimum:** 8GB RAM, CPU-only (slower)
- **Recommended:** 8GB+ GPU VRAM with CUDA support
- **Storage:** ~3GB for base model + adapters

## Customization

You can modify the chatbot behavior by adjusting parameters in `chatbot.py`:

```python
response = self.generate_response(
    user_input,
    max_new_tokens=256,  # Maximum length of response
    temperature=0.7,     # Creativity (0.0-1.0, higher = more creative)
    top_p=0.9           # Nucleus sampling parameter
)
```

## Troubleshooting

### CUDA Out of Memory
If you encounter GPU memory errors:
1. The chatbot will automatically fall back to CPU
2. Reduce `max_new_tokens` parameter
3. Clear conversation history with `reset` command

### Model Loading Issues
Ensure all model files are present:
- Adapter weights (`.safetensors` files)
- Configuration files (`.json` files)
- Tokenizer files

### Dependencies
If you encounter import errors:
```bash
pip install --upgrade torch transformers peft accelerate
```

## License

Please refer to the Qwen model license and ensure compliance with all usage terms.
