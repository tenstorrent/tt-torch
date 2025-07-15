# Forge Agent: Hugging Face Model Compatibility Pipeline

This package implements the Hugging Face Model Compatibility Pipeline as specified in the architecture document.

## Components

1. **Model Selection and Metadata Service** - Selects and extracts metadata from Hugging Face models
2. **Agentic Test Pipeline** - Performs automated testing and adaptation of models
3. **Result Tracking Database** - Stores test results and metadata
4. **Analysis and Reporting Dashboard** - Visualizes test results and statistics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m forge_agent.main
```
