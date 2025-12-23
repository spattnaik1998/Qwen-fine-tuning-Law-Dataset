"""
Simple Chatbot Application using Fine-tuned Qwen Model
This chatbot loads the fine-tuned LoRA weights and provides an interactive interface.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os

class FineTunedChatbot:
    def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct", adapter_path=None):
        """
        Initialize the chatbot with base model and optional LoRA adapter.

        Args:
            base_model_name: HuggingFace model name
            adapter_path: Path to the LoRA adapter weights
        """
        print("Loading model and tokenizer...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path if adapter_path else base_model_name,
            trust_remote_code=True
        )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        # Load LoRA adapter if provided
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # Merge adapter with base model

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Model loaded successfully!\n")

        # Conversation history with system message
        system_message = {
            "role": "system",
            "content": """You are an AI assistant fine-tuned on the academic paper "When Fairness Isn't Statistical: The Limits of Machine Learning in Evaluating Legal Reasoning" by Claire Barale, Michael Rovatsos, and Nehal Bhuta.

This paper examines whether machine learning methods can meaningfully evaluate fairness in refugee adjudication using the ASYLEX dataset of 59,000+ Canadian refugee decisions. The research evaluates three ML approaches:
1. Feature-based statistical analysis
2. Semantic clustering using embeddings
3. Predictive modeling (Random Forest and Neural Networks)

Key findings include:
- ML methods produce divergent and sometimes contradictory signals
- Predictive models depend heavily on contextual/procedural features rather than legal reasoning
- Semantic clustering fails to capture substantive legal arguments
- Statistical fairness metrics cannot distinguish between unjust bias and legitimate legal discretion
- None of the methods can effectively assess legal reasoning or justification

The paper argues that evaluating fairness in law requires methods grounded in legal reasoning and institutional context, not just statistical patterns.

When answering questions, focus on these key topics: refugee law, asylum adjudication, fairness in ML, the ASYLEX dataset, limitations of computational fairness evaluation, and the distinction between distributive and procedural fairness."""
        }
        self.conversation_history = [system_message]

    def generate_response(self, user_input, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        Generate a response to user input.

        Args:
            user_input: User's message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated response text
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Format conversation for the model
        text = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Add assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def reset_conversation(self):
        """Reset the conversation history with system message."""
        system_message = {
            "role": "system",
            "content": """You are an AI assistant fine-tuned on the academic paper "When Fairness Isn't Statistical: The Limits of Machine Learning in Evaluating Legal Reasoning" by Claire Barale, Michael Rovatsos, and Nehal Bhuta.

This paper examines whether machine learning methods can meaningfully evaluate fairness in refugee adjudication using the ASYLEX dataset of 59,000+ Canadian refugee decisions. The research evaluates three ML approaches:
1. Feature-based statistical analysis
2. Semantic clustering using embeddings
3. Predictive modeling (Random Forest and Neural Networks)

Key findings include:
- ML methods produce divergent and sometimes contradictory signals
- Predictive models depend heavily on contextual/procedural features rather than legal reasoning
- Semantic clustering fails to capture substantive legal arguments
- Statistical fairness metrics cannot distinguish between unjust bias and legitimate legal discretion
- None of the methods can effectively assess legal reasoning or justification

The paper argues that evaluating fairness in law requires methods grounded in legal reasoning and institutional context, not just statistical patterns.

When answering questions, focus on these key topics: refugee law, asylum adjudication, fairness in ML, the ASYLEX dataset, limitations of computational fairness evaluation, and the distinction between distributive and procedural fairness."""
        }
        self.conversation_history = [system_message]
        print("Conversation history cleared.\n")

    def chat(self):
        """Start an interactive chat session."""
        print("=" * 70)
        print("Fine-tuned Qwen Chatbot")
        print("=" * 70)
        print("Type your questions below. Commands:")
        print("  - 'quit' or 'exit': Exit the chatbot")
        print("  - 'reset': Clear conversation history")
        print("  - 'help': Show this help message")
        print("=" * 70)
        print()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'reset':
                    self.reset_conversation()
                    continue
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  - 'quit' or 'exit': Exit the chatbot")
                    print("  - 'reset': Clear conversation history")
                    print("  - 'help': Show this help message\n")
                    continue
                elif not user_input:
                    continue

                # Generate and display response
                print("Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")


def main():
    """Main function to run the chatbot."""
    # Available model paths
    available_models = {
        "1": ("qwen_lora_adapter", "Basic LoRA SFT Adapter"),
        "2": ("qwen_lora_sft/checkpoint-39", "LoRA SFT Checkpoint 39"),
        "3": ("dpo_model/checkpoint-3", "DPO Model Checkpoint 3"),
        "4": ("qwen_lora_dpo_adapter", "LoRA DPO Adapter"),
        "5": (None, "Base Model (No Adapter)")
    }

    print("Available Fine-tuned Models:")
    print("-" * 50)
    for key, (path, desc) in available_models.items():
        print(f"{key}. {desc}")
    print("-" * 50)

    # Get user choice
    choice = input("\nSelect a model (1-5) [default: 2]: ").strip() or "2"

    if choice not in available_models:
        print("Invalid choice. Using default (LoRA SFT Checkpoint 39).")
        choice = "2"

    adapter_path, model_desc = available_models[choice]

    if adapter_path:
        # Convert to absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        adapter_path = os.path.join(script_dir, adapter_path)

    print(f"\nLoading: {model_desc}\n")

    # Initialize and run chatbot
    try:
        chatbot = FineTunedChatbot(adapter_path=adapter_path)
        chatbot.chat()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("\nMake sure you have the required dependencies installed:")
        print("  pip install torch transformers peft accelerate")
        sys.exit(1)


if __name__ == "__main__":
    main()
