"""
Streamlit Web Application for Fine-tuned Qwen Chatbot
A beautiful, user-friendly interface for interacting with the fine-tuned models
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time

# Page configuration
st.set_page_config(
    page_title="Legal AI Research Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(adapter_path=None, base_model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Load the model and tokenizer with caching.
    This function only runs once and caches the result.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer (always from base model for consistency)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Load LoRA adapter if provided
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    if device == "cpu":
        model = model.to(device)

    model.eval()

    return model, tokenizer, device


def generate_response(model, tokenizer, device, conversation_history, max_tokens, temperature, top_p):
    """Generate a response from the model."""
    try:
        # Format conversation for the model
        text = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Handle empty responses
        if not response or response.strip() == "":
            response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

        return response

    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")


def main():
    # Header
    st.markdown('<div class="chat-header">🤖 Legal AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-top: -1rem;">Fine-tuned on "When Fairness Isn\'t Statistical" - ML in Refugee Law</p>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        st.subheader("Model Selection")
        model_options = {
            "LoRA SFT Checkpoint 39 (Recommended)": "qwen_lora_sft/checkpoint-39",
            "Basic LoRA SFT Adapter": "qwen_lora_adapter",
            "DPO Model Checkpoint 3": "dpo_model/checkpoint-3",
            "LoRA DPO Adapter": "qwen_lora_dpo_adapter",
            "Base Model (No Fine-tuning)": None
        }

        selected_model = st.selectbox(
            "Choose a model:",
            options=list(model_options.keys()),
            index=0,
            help="Select which fine-tuned model to use for generation"
        )

        adapter_path = model_options[selected_model]
        if adapter_path:
            # Use current working directory for better cross-platform compatibility
            adapter_path = os.path.join(os.getcwd(), adapter_path)
            # Verify adapter path exists
            if not os.path.exists(adapter_path):
                st.error(f"⚠️ Adapter path not found: {adapter_path}")
                st.info("Make sure the model directories are in the same folder as app.py")
                adapter_path = None

        # Model info
        with st.expander("ℹ️ Model Information"):
            st.markdown(f"""
            **Selected:** {selected_model}

            **Base Model:** Qwen2.5-1.5B-Instruct

            **Fine-tuning Topic:** Fairness in ML for Refugee Law

            **Paper:** "When Fairness Isn't Statistical: The Limits of Machine Learning in Evaluating Legal Reasoning"

            **Authors:** Claire Barale, Michael Rovatsos, Nehal Bhuta (University of Edinburgh)

            **LoRA Config:**
            - Rank: 16
            - Alpha: 32
            - Dropout: 0.05
            """)

        st.divider()

        # Generation parameters
        st.subheader("Generation Parameters")

        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=1000,
            value=256,
            step=50,
            help="Maximum number of tokens to generate"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )

        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling parameter"
        )

        st.divider()

        # Action buttons
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            # Reset conversation history with system message
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
            st.session_state.conversation_history = [system_message]
            st.rerun()

        if st.button("🔄 Reload Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.divider()

        # System info
        with st.expander("💻 System Information"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.write(f"**Device:** {device.upper()}")
            if device == "cuda":
                st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
                st.write(f"**VRAM:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversation_history" not in st.session_state:
        # Add system message to guide the model about the paper
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
        st.session_state.conversation_history = [system_message]

    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    # Load model
    try:
        with st.spinner("Loading model... This may take a minute on first run."):
            model, tokenizer, device = load_model(adapter_path)

        if not st.session_state.model_loaded:
            st.success(f"✅ Model loaded successfully on {device.upper()}!")
            st.session_state.model_loaded = True

    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {str(e)}")
        st.info("💡 Make sure you have internet connection to download the base model (~3GB)")
        st.info("💡 Check that adapter directories exist in the repository")
        st.stop()
    except ImportError as e:
        st.error(f"❌ Missing required library: {str(e)}")
        st.code("pip install -r requirements.txt", language="bash")
        st.stop()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error(f"❌ GPU Out of Memory")
            st.info("💡 Try reducing max tokens or the app will fallback to CPU")
            st.info("💡 Close other GPU-intensive applications")
        else:
            st.error(f"❌ Runtime error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Make sure you have internet connection to download the base model (~3GB)")
        st.info("💡 Try clicking 'Reload Model' button in the sidebar")
        with st.expander("See detailed error"):
            st.code(str(e))
        st.stop()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        conversation_history=st.session_state.conversation_history,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )

                    st.markdown(response)

                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        error_msg = "⚠️ GPU ran out of memory. Try reducing max tokens or the model will use CPU."
                    else:
                        error_msg = f"⚠️ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"⚠️ Unexpected error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Welcome message if no messages
    if len(st.session_state.messages) == 0:
        st.info("👋 Welcome! I'm trained on the paper about ML fairness in refugee law. Ask me anything about the research!")

        # Example prompts
        st.subheader("Try these example questions:")
        col1, col2, col3 = st.columns(3)

        example_prompts = [
            "What are the main findings of the paper?",
            "What is the ASYLEX dataset?",
            "Why do ML methods fail to assess legal fairness?"
        ]

        for col, prompt_text in zip([col1, col2, col3], example_prompts):
            with col:
                if st.button(prompt_text, key=prompt_text):
                    # Simulate user input
                    st.session_state.messages.append({"role": "user", "content": prompt_text})
                    st.session_state.conversation_history.append({"role": "user", "content": prompt_text})
                    st.rerun()

    # Footer with tips
    st.divider()
    with st.expander("💡 Pro Tips"):
        st.markdown("""
        **What you can ask:**
        - Questions about the paper's methodology, findings, and conclusions
        - Details about the ASYLEX dataset (59,000+ Canadian refugee decisions)
        - Explanations of the three ML methods tested (feature-based, clustering, predictive)
        - Discussion of fairness metrics and their limitations in legal contexts
        - Information about refugee law and asylum adjudication

        **For better responses:**
        - Be specific about which aspect of the paper interests you
        - Adjust temperature: lower (0.3-0.6) for factual answers, higher (0.7-0.9) for explanations
        - Use the Clear Chat button to start fresh conversations
        - Try different fine-tuned models to compare their understanding

        **Troubleshooting:**
        - If responses are slow, check System Information to confirm GPU is being used
        - Click "Reload Model" if you encounter issues
        - Reduce max tokens if you experience memory errors
        """)


if __name__ == "__main__":
    main()
