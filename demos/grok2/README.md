# Grok 2 Demo with tt-torch

This demo showcases how to run xAI's Grok 2 model using the tt-torch compiler backend for Tenstorrent hardware acceleration.

## Overview

Grok 2 is a large language model developed by xAI that can engage in conversational AI, answer questions, and perform various text generation tasks. This demo integrates Grok 2 with the tt-torch compiler to leverage Tenstorrent's AI acceleration capabilities.

## Model Information

- **Model**: `xai-org/grok-2` from Hugging Face
- **Size**: Approximately 500GB
- **Requirements**: 8 GPUs with >40GB VRAM each (as recommended by xAI)
- **License**: Grok 2 Community License Agreement

## Prerequisites

### Hardware Requirements
- **Minimum**: 8 GPUs with 40GB+ VRAM each
- **Storage**: 500GB+ free disk space for model weights
- **Memory**: 64GB+ system RAM recommended

### Software Requirements
- Python 3.10+
- PyTorch 2.7.0+
- transformers >= 4.50.0
- tt-torch (this repository)
- All dependencies from `requirements.txt`

### Authentication
You may need to authenticate with Hugging Face to access the Grok 2 model:

```bash
huggingface-cli login
```

Or set your token as an environment variable:
```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

## Installation

1. **Activate the tt-torch environment**:
   ```bash
   source .tt-torch-venv/bin/activate
   ```

2. **Install additional dependencies**:
   ```bash
   pip install transformers>=4.50.0 tabulate
   ```

3. **Download the model** (optional - will auto-download on first run):
   ```bash
   huggingface-cli download xai-org/grok-2 --local-dir ./grok-2-local
   ```

## Usage

### Basic Usage (Non-interactive)
Run the demo with default prompts:

```bash
source .tt-torch-venv/bin/activate
python grok2_demo.py
```

This will:
- Load the Grok 2 model
- Compile it with tt-torch backend
- Run several example prompts
- Display the generated responses

### Interactive Mode
For a conversational experience:

```bash
source .tt-torch-venv/bin/activate
python grok2_demo.py --run_interactive
```

This allows you to:
- Enter custom prompts
- Have a conversation with Grok 2
- Type "quit", "exit", or "stop" to end the session

### Testing Mode
For testing the tt-torch compilation pipeline with a smaller model:

```bash
source .tt-torch-venv/bin/activate
python grok2_demo.py --test_model
```

This uses Microsoft DialoGPT-small (~500MB) instead of Grok 2 (~500GB) to test the compilation process.

## Expected Output

The demo will show:
1. Model loading progress
2. Compilation status
3. Generated responses to prompts
4. Performance information (if available)

Example:
```
Loading Grok 2 model: xai-org/grok-2
Compiling model with tt-torch backend...
Model compilation successful!

Prompt: What is the meaning of life?
Grok response: The meaning of life is a profound philosophical question...
```

## Configuration Options

The demo uses these default settings:
- **Temperature**: 0.7 (controls randomness)
- **Top-p**: 0.9 (nucleus sampling)
- **Max new tokens**: 100 (response length)
- **Device mesh**: [1, 8] (8 GPUs)

You can modify these in the `grok2_demo.py` file as needed.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Ensure you have 8 GPUs with 40GB+ VRAM each
   - Try reducing batch size or sequence length
   - Consider using gradient checkpointing

2. **Model Download Fails**
   - Check your internet connection
   - Verify Hugging Face authentication
   - Ensure sufficient disk space (500GB+)

3. **Compilation Errors**
   - Check tt-torch installation
   - Verify device availability
   - Try falling back to single device mode

4. **Performance Issues**
   - Model is very large - first run will be slow
   - Subsequent runs should be faster due to caching
   - Consider using smaller models for testing

### Error Messages

- **"Insufficient memory/storage"**: Need more VRAM or disk space
- **"Missing authentication"**: Run `huggingface-cli login`
- **"Network connectivity issues"**: Check internet connection
- **"Could not create 8-device mesh"**: Falls back to single device automatically

## Performance Notes

- **First run**: Expect significant setup time for model download and compilation
- **Subsequent runs**: Should be faster due to caching
- **Memory usage**: Very high - monitor system resources
- **Generation speed**: Will vary based on hardware and prompt complexity

## Model Behavior

Grok 2 is designed to be:
- Conversational and engaging
- Factual and informative
- Creative when appropriate
- Respectful of content policies

The model follows xAI's guidelines and may refuse certain types of requests.

## Files

- `grok2_demo.py`: Main demo script
- `README.md`: This documentation
- Model weights: Downloaded to Hugging Face cache or local directory

## References

- [Grok 2 on Hugging Face](https://huggingface.co/xai-org/grok-2)
- [xAI Official Website](https://x.ai/)
- [Grok 2 Community License](https://huggingface.co/xai-org/grok-2/blob/main/LICENSE)
- [tt-torch Documentation](../../docs/)

## Support

For issues specific to:
- **tt-torch integration**: Open an issue in this repository
- **Grok 2 model**: Refer to Hugging Face model page
- **Hardware requirements**: Consult Tenstorrent documentation
