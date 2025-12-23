# Streamlit Web Application Guide

## Overview

This Streamlit application provides a beautiful, user-friendly web interface for interacting with your fine-tuned Qwen models. No command-line knowledge required!

## Features

- **Beautiful Chat Interface** - Modern, intuitive chat UI similar to ChatGPT
- **Model Selection** - Easy dropdown to switch between fine-tuned models
- **Real-time Parameters** - Adjust temperature, max tokens, and top-p on the fly
- **Conversation Management** - Chat history with clear button
- **System Monitoring** - See GPU/CPU usage and model status
- **Example Prompts** - Quick-start buttons for common questions
- **Responsive Design** - Works on desktop and mobile

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- Transformers (HuggingFace library)
- PEFT (LoRA adapter support)
- Accelerate (GPU optimization)
- Streamlit (web framework)

### 2. Run the Application

```bash
streamlit run app.py
```

The app will automatically:
1. Open in your default browser at `http://localhost:8501`
2. Download the base model (~3GB) on first run
3. Load your selected fine-tuned adapter
4. Be ready to chat!

### 3. Start Chatting

- Type your message in the chat input at the bottom
- Press Enter or click Send
- Watch the AI generate a response in real-time

## Interface Guide

### Left Sidebar

#### Configuration Section

**Model Selection**
- Choose from 5 available models
- Default: LoRA SFT Checkpoint 39 (recommended)
- Models can be switched at any time

**Model Information**
- Expandable panel showing current model details
- LoRA configuration parameters
- Base model information

#### Generation Parameters

**Max Tokens** (50-1000)
- Controls the maximum length of responses
- Higher = longer responses
- Default: 256

**Temperature** (0.1-2.0)
- Controls randomness/creativity
- Lower (0.1-0.5): More focused, deterministic
- Medium (0.6-0.9): Balanced
- Higher (1.0-2.0): More creative, random
- Default: 0.7

**Top P** (0.1-1.0)
- Nucleus sampling parameter
- Controls diversity of word choices
- Lower: More conservative
- Higher: More diverse
- Default: 0.9

#### Action Buttons

**Clear Chat**
- Removes all messages from the current conversation
- Resets conversation history
- Starts fresh chat

**Reload Model**
- Clears model cache
- Reloads model from scratch
- Use if you encounter issues

#### System Information

- Shows current device (CPU/GPU)
- GPU name and VRAM if available
- Useful for troubleshooting

### Main Chat Area

**Chat Messages**
- User messages appear on the right
- AI responses appear on the left
- Scrollable history of the entire conversation

**Chat Input**
- Text box at the bottom
- Type your message and press Enter
- Supports multi-line input (Shift+Enter)

**Example Prompts** (on welcome screen)
- Quick-start buttons for common questions
- Click to auto-populate and send

## Available Models

### 1. LoRA SFT Checkpoint 39 (Recommended)
- **Best for:** General purpose conversations
- **Training:** Supervised Fine-Tuning
- **Quality:** High-quality instruction following

### 2. Basic LoRA SFT Adapter
- **Best for:** Lightweight deployment
- **Training:** Basic SFT
- **Quality:** Good general performance

### 3. DPO Model Checkpoint 3
- **Best for:** Human-aligned responses
- **Training:** Direct Preference Optimization
- **Quality:** Preferred response style

### 4. LoRA DPO Adapter
- **Best for:** Preference-aligned, lightweight
- **Training:** DPO with LoRA
- **Quality:** Good human alignment

### 5. Base Model (No Fine-tuning)
- **Best for:** Testing original capabilities
- **Training:** None (original Qwen model)
- **Quality:** Base model performance

## Tips for Best Results

### Getting Quality Responses

1. **Be Specific**
   - Bad: "Tell me about AI"
   - Good: "Explain how neural networks work in 3 simple steps"

2. **Provide Context**
   - Include relevant background information
   - Reference previous messages in the conversation

3. **Adjust Parameters**
   - For creative writing: Higher temperature (0.8-1.2)
   - For factual answers: Lower temperature (0.3-0.6)
   - For code generation: Medium temperature (0.6-0.8)

### Model Selection

- Start with **LoRA SFT Checkpoint 39** (default)
- Try **DPO Model** for more natural conversations
- Use **Base Model** to compare improvements

### Performance Optimization

**If responses are slow:**
- Reduce max tokens to 128-150
- Ensure you're using GPU (check System Information)
- Close other GPU-intensive applications

**If model doesn't load:**
- Check internet connection (first run downloads ~3GB)
- Verify disk space (need ~5GB free)
- Try "Reload Model" button

**If responses are repetitive:**
- Increase temperature to 0.8-1.0
- Try a different model variant
- Clear chat and start fresh

## Advanced Usage

### Running on Custom Port

```bash
streamlit run app.py --server.port 8080
```

### Running on Network (Allow Other Devices)

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then access from other devices at: `http://YOUR_IP:8501`

### Headless Mode (No Auto-Browser)

```bash
streamlit run app.py --server.headless true
```

### Custom Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Troubleshooting

### "Model download failed"
**Cause:** No internet connection or HuggingFace down
**Solution:**
- Check internet connection
- Try again later
- Use VPN if HuggingFace is blocked

### "CUDA out of memory"
**Cause:** GPU doesn't have enough VRAM
**Solution:**
- App will automatically fall back to CPU
- Reduce max tokens
- Close other programs

### "Module not found" errors
**Cause:** Missing dependencies
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### "Port already in use"
**Cause:** Another Streamlit app running
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

### App is slow on CPU
**Cause:** No GPU available
**Solution:**
- This is expected - CPU inference is slower
- First response may take 30-60 seconds
- Consider using a GPU-enabled environment

### Responses seem poor quality
**Cause:** Wrong parameters or model
**Solution:**
- Reset temperature to 0.7
- Try LoRA SFT Checkpoint 39 model
- Clear chat history and try again

## Keyboard Shortcuts

- **Enter:** Send message
- **Shift+Enter:** New line in message
- **Ctrl+C:** Stop the Streamlit server (in terminal)
- **F5:** Refresh page (clears chat)

## Comparison: CLI vs Streamlit

| Feature | CLI (chatbot.py) | Streamlit (app.py) |
|---------|------------------|-------------------|
| Interface | Terminal | Web Browser |
| Ease of Use | Medium | Easy |
| Visual Appeal | Basic | Beautiful |
| Parameters | Code editing | Live sliders |
| Multi-user | No | Yes (network mode) |
| Setup | Instant | ~1 minute |
| Resource Usage | Lower | Slightly higher |

## Security Notes

- App runs locally by default (localhost:8501)
- No data is sent to external servers (except model download)
- Conversations are not saved automatically
- To share: Use network mode cautiously on trusted networks

## Next Steps

1. Experiment with different models
2. Try adjusting generation parameters
3. Test with various types of questions
4. Compare model outputs side-by-side
5. Customize the UI (see Advanced Usage)

## Support

For issues:
1. Check this guide's Troubleshooting section
2. Review `MODEL_WEIGHTS_INFO.md` for technical details
3. Check Streamlit docs: https://docs.streamlit.io
4. Verify model files are intact

## Credits

- **Base Model:** Qwen Team (Alibaba Cloud)
- **Framework:** HuggingFace Transformers
- **UI:** Streamlit
- **Fine-tuning:** TRL (Transformer Reinforcement Learning)
