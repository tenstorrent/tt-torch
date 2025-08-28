# Single Layer Demo for tt-torch

This demo demonstrates how to extract and compile a single transformer layer using the tt-torch compiler backend. This is useful for:

- **Testing compilation**: Verify tt-torch works on individual layers
- **Performance analysis**: Measure single layer execution times
- **Debugging**: Isolate issues to specific layers
- **Memory optimization**: Test with minimal memory requirements

## Overview

The `grok_single_layer_demo.py` script extracts a single transformer layer from either:
- **Microsoft DialoGPT-small** (default - ~351MB)
- **xAI Grok 2** (with `--full_model` flag - ~500GB)

It then compiles just that layer with tt-torch and runs performance tests.

## Files

- `grok_single_layer_demo.py` - Main single layer demo script
- `SINGLE_LAYER_README.md` - This documentation

## Prerequisites

- tt-torch environment activated (`.tt-torch-venv`)
- N150 or compatible Tenstorrent hardware
- Required Python packages (transformers, torch, tt_torch)

## Usage

### Basic Single Layer Test
Test layer 0 (first layer) of DialoGPT-small:

```bash
source ../.tt-torch-venv/bin/activate
python grok_single_layer_demo.py
```

### Test Different Layers
Test layer 5 of the model:

```bash
python grok_single_layer_demo.py --layer 5
```

### Use Full Grok 2 Model
⚠️ **Warning**: Requires ~500GB RAM and significant resources:

```bash
python grok_single_layer_demo.py --full_model --layer 0
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--layer N` | Extract layer N (0-based indexing) | 0 |
| `--full_model` | Use Grok 2 instead of DialoGPT-small | False |

## What the Demo Does

1. **Load Model**: Downloads and loads the specified transformer model
2. **Extract Layer**: Isolates the specified transformer layer
3. **Compile Layer**: Compiles the single layer with tt-torch backend
4. **Test Execution**: Runs the layer multiple times with test input
5. **Performance Analysis**: Reports execution times and output statistics

## Expected Output

```
Using test model: microsoft/DialoGPT-small
Note: Using smaller model for testing single layer compilation.
Loading full model...
Model loaded successfully. Total layers: 12
Extracted layer 0 of 12
Layer type: <class 'transformers.models.gpt2.modeling_gpt2.GPT2Block'>
Using single device for single layer...
Compiling single layer with tt-torch backend...
Single layer compilation successful!

Testing compiled layer...
Input shape: torch.Size([1, 4, 768])
Input prompt: 'Testing single layer:'
Run 1: 0.0234s, Output shape: torch.Size([1, 4, 768])
Run 2: 0.0187s, Output shape: torch.Size([1, 4, 768])
Run 3: 0.0191s, Output shape: torch.Size([1, 4, 768])

Average execution time: 0.0204s
Layer 0 successfully processed on tt-torch backend!
Output statistics:
  Mean: 0.001234
  Std: 0.567890
  Min: -2.345678
  Max: 3.456789

Cleaning up...
Single layer demo completed!
```

## Model Architecture Support

The demo supports multiple transformer architectures:

### GPT-Style Models (DialoGPT, GPT-2)
- **Layers**: `model.transformer.h[N]`
- **Total layers**: Usually 12-24 layers
- **Layer type**: GPT2Block

### LLaMA-Style Models (Grok 2, LLaMA)
- **Layers**: `model.model.layers[N]`
- **Total layers**: Usually 24-96+ layers
- **Layer type**: Various (depends on model)

## Performance Analysis

The demo provides several metrics:

- **Execution Time**: Average time per layer execution
- **Output Statistics**: Mean, standard deviation, min/max values
- **Shape Verification**: Ensures output maintains expected dimensions
- **Memory Usage**: Minimal footprint for single layer

## Troubleshooting

### Common Issues

1. **"Layer index X >= total layers Y"**
   - Solution: Use `--layer` with valid index (0 to Y-1)

2. **"Unknown model architecture"**
   - Solution: Model may not be supported, try DialoGPT-small

3. **Compilation failures**
   - Solution: Check tt-torch environment and hardware setup

4. **Memory errors with `--full_model`**
   - Solution: Ensure sufficient RAM (500GB+) or use test model

### Layer Index Reference

For **DialoGPT-small** (12 layers):
- Valid layers: 0-11
- Layer 0: First transformer block
- Layer 11: Last transformer block

For **Grok 2** (varies):
- Valid layers: 0-(N-1) where N is total layers
- Check output for total layer count

## Comparison with Full Model Demo

| Feature | Single Layer Demo | Full Model Demo |
|---------|-------------------|-----------------|
| **Memory Usage** | Minimal | Full model size |
| **Compile Time** | Fast | Slower |
| **Testing Scope** | One layer | End-to-end |
| **Use Case** | Development/Debug | Production |
| **Hardware Reqs** | Low | High |

## Next Steps

1. **Try Different Layers**: Test various layer indices
2. **Performance Profiling**: Compare layer execution times
3. **Memory Analysis**: Monitor memory usage per layer
4. **Custom Models**: Adapt for other transformer architectures

## Advanced Usage

### Extract Multiple Layers
Run the script multiple times for different layers:

```bash
for i in {0..5}; do
    echo "Testing layer $i"
    python grok_single_layer_demo.py --layer $i
done
```

### Benchmark All Layers
Create a performance profile of all layers in a model.

## Technical Details

- **Input Processing**: Uses model embeddings as layer input
- **Layer Wrapping**: `SingleTransformerLayer` class for isolation
- **Compilation**: Standard tt-torch compilation process
- **Device Management**: Single device mesh for simplicity
