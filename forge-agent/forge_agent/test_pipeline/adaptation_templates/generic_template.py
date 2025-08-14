"""
Generic adaptation template for models without specific templates.
"""
from typing import Dict, Any, Optional
import torch

# Level of adaptation required for this template
# Generic template performs no real changes; mark as level 0 (none)
ADAPTATION_LEVEL = "none"

def adapt(model_data: Dict[str, Any], test_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply generic adaptations to make the model compatible with tt-torch.
    
    Args:
        model_data: Dictionary containing model, config, and other model data
        test_config: Configuration for the model test
        
    Returns:
        Dictionary with adapted model data
    """
    # Get the model
    model = model_data.get("model")
    
    # Set model to eval mode for inference
    if model is not None:
        model.eval()
    
    # Make a copy of the input data
    adapted_data = {**model_data}
    
    return adapted_data
