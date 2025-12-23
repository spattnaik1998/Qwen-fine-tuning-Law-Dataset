"""
Quick test script for the fine-tuned chatbot
Tests basic functionality without interactive mode
"""

import os
import sys

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    required_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'peft': 'peft',
        'accelerate': 'accelerate'
    }

    missing = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\nAll dependencies installed!\n")
    return True


def check_model_files():
    """Check if model files exist."""
    print("Checking model files...")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    models = {
        "qwen_lora_adapter": "qwen_lora_adapter/adapter_config.json",
        "qwen_lora_sft": "qwen_lora_sft/checkpoint-39/adapter_config.json",
        "dpo_model": "dpo_model/checkpoint-3/adapter_config.json",
        "qwen_lora_dpo_adapter": "qwen_lora_dpo_adapter/adapter_config.json"
    }

    for name, path in models.items():
        full_path = os.path.join(script_dir, path)
        if os.path.exists(full_path):
            print(f"✓ {name}")
        else:
            print(f"✗ {name} - NOT FOUND at {path}")

    print()


def test_basic_functionality():
    """Test basic chatbot initialization."""
    print("Testing chatbot initialization...")
    print("This will download the base model from HuggingFace (~3GB)...")
    print("Press Ctrl+C to cancel if you don't want to download now.\n")

    try:
        from chatbot import FineTunedChatbot

        # Test with just base model (no adapter) to keep it simple
        print("Initializing chatbot with base model...")
        chatbot = FineTunedChatbot(adapter_path=None)

        # Test a simple question
        print("\nTesting response generation...")
        test_question = "What is 2+2?"
        print(f"Question: {test_question}")
        response = chatbot.generate_response(test_question, max_new_tokens=50)
        print(f"Response: {response}\n")

        print("✓ Chatbot is working correctly!")
        print("\nYou can now run: python chatbot.py")
        return True

    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        return False
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection (to download base model)")
        print("2. Check if you have enough disk space (~3GB)")
        print("3. Verify all dependencies are installed")
        return False


def main():
    print("=" * 70)
    print("Fine-tuned Qwen Chatbot - Test Script")
    print("=" * 70)
    print()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check model files
    check_model_files()

    # Ask user if they want to run the full test
    response = input("Run initialization test? This will download ~3GB base model (y/n): ").strip().lower()

    if response in ['y', 'yes']:
        test_basic_functionality()
    else:
        print("\nSkipping initialization test.")
        print("Run 'python chatbot.py' when ready to use the chatbot.")


if __name__ == "__main__":
    main()
