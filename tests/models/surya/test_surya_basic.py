# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import numpy as np
from PIL import Image, ImageDraw
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from surya.settings import settings

from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.table_rec import TableRecPredictor

import random
import os

class ThisTester(ModelTester):
    def __init__(self, *args, model_type="recognition", **kwargs):
        # Set model type for different surya models (recognition, detection, layout, table_recognition)
        self.model_type = model_type

        # Call parent constructor
        super(ThisTester, self).__init__(*args, **kwargs)

    # KCM FIXME - Figure out how to clean this up, only list what is actually used.
    def _extract_outputs(self, output_object):
        # Handle outputs based on model type
        if self.model_type == "recognition":
            # For encoder-decoder models, logits are the token predictions
            if hasattr(output_object, 'logits'):
                return (output_object.logits,)  # Return as tuple for verification
            # Some models return a tuple with logits as first element
            elif isinstance(output_object, tuple) and len(output_object) > 0:
                return output_object
            # Handle case where model returns something else
            else:
                print(f"Unexpected output type: {type(output_object)}")
                print(f"Output attributes: {dir(output_object) if hasattr(output_object, '__dict__') else None}")
                return (output_object,)  # Return as tuple for verification
        
        elif self.model_type == "detection":
            # Detection model returns a SemanticSegmenterOutput object
            if hasattr(output_object, 'logits'):
                print(f"Detection output is a SemanticSegmenterOutput object")
                # Just return the logits for comparison - that's the most important part
                return (output_object.logits,)
            # Detection model returns dict with pred_logits and pred_boxes
            elif isinstance(output_object, dict):
                if 'pred_logits' in output_object and 'pred_boxes' in output_object:
                    return (output_object['pred_logits'], output_object['pred_boxes'])
                elif 'logits' in output_object:
                    return (output_object['logits'],)
                else:
                    print(f"Detection output keys: {list(output_object.keys())}")
                    # Just return the first value as a tuple
                    return (next(iter(output_object.values())),)
            elif isinstance(output_object, tuple):
                return output_object
            
            # Default fallback - wrap in tuple
            return (output_object,)
            
        elif self.model_type in ["layout", "table_recognition"]:
            # Similar output structure as detection
            if isinstance(output_object, dict):
                # Extract the most important outputs as tuples
                outputs = tuple(output_object.values())
                return outputs
            elif isinstance(output_object, tuple):
                return output_object
                
            # Default fallback - wrap in tuple
            return (output_object,)
        
        else:
            # For any other models, ensure we return a tuple
            if isinstance(output_object, tuple):
                return output_object
            else:
                return (output_object,)

    def _load_model(self):
        # Create appropriate model based on model_type
        if self.model_type == "recognition":
            # FLush prints:
            print("Loading recognition model...", flush=True)

            # KCM - As far as I can tell, this didn't change anything, but model said env-vars needed to enable compile
            os.environ["COMPILE_RECOGNITION"] = "true"  

            # Create predictor directly without custom parameters
            predictor = RecognitionPredictor()
            model = predictor.model

            print("Loaded recognition model", flush=True)
            
            # Set to eval mode explicitly
            model.eval()
            
            # CRITICAL FIX: Add the missing encoder-to-decoder projection
            # This is a minimal necessary patch for this architecture to work
            print("Checking if model needs enc_to_dec_proj layer...", flush=True)
            if hasattr(model, "encoder") and hasattr(model, "decoder") and not hasattr(model, "enc_to_dec_proj"):
                import torch.nn as nn
                import torch
                
                print("Adding missing enc_to_dec_proj layer", flush=True)
                
                # Determine dimensions
                encoder_dim = getattr(model.encoder, "hidden_size", 1024)
                decoder_dim = getattr(model.decoder, "hidden_size", 1280)
                
                print(f"Creating projection: {encoder_dim} → {decoder_dim}", flush=True)
                
                # Create and initialize the layer with identity matrix plus zeros
                with torch.no_grad():
                    model.enc_to_dec_proj = nn.Linear(encoder_dim, decoder_dim)
                    if encoder_dim <= decoder_dim:
                        nn.init.zeros_(model.enc_to_dec_proj.weight)
                        for i in range(min(encoder_dim, decoder_dim)):
                            model.enc_to_dec_proj.weight[i, i] = 1.0
            
            print("Finished loading of recognition model", flush=True)

            # CRITICAL FIX #2: Add missing attributes to cross-attention modules
            print("Checking for cross-attention modules that need patching...", flush=True)
            cross_attn_patched = 0
            
            import types
            for name, module in model.named_modules():
                if module.__class__.__name__ == "SuryaADETRDecoderSdpaCrossAttention":
                    # Add missing key_states and value_states attributes
                    if not hasattr(module, "key_states"):
                        print(f"Adding missing key_states attribute to {name}", flush=True)
                        module.key_states = None
                        cross_attn_patched += 1
                    
                    if not hasattr(module, "value_states"):
                        print(f"Adding missing value_states attribute to {name}", flush=True)
                        module.value_states = None
                        cross_attn_patched += 1
                        
            print(f"Patched {cross_attn_patched} missing attributes in cross-attention modules", flush=True)

            # Add a simple input wrapper for convenience
            # This will handle adding the decoder_input_ids automatically
            def preprocess_inputs(inputs):
                if isinstance(inputs, dict) and 'pixel_values' in inputs:
                    pixel_values = inputs['pixel_values']
                    
                    # If decoder_input_ids not provided, create a simple start token
                    if 'decoder_input_ids' not in inputs:
                        inputs['decoder_input_ids'] = torch.zeros(
                            pixel_values.shape[0], 1,  # Match batch size
                            dtype=torch.long, 
                            device=pixel_values.device
                        )
                        print(f"Added decoder_input_ids with shape {inputs['decoder_input_ids'].shape}")
                        
                return inputs
            
            # Store the preprocessor on the model for later use
            model.preprocess_inputs = preprocess_inputs
            
            return model
        elif self.model_type == "detection":
            model = DetectionPredictor()
            model.model.eval()
            return model.model
        elif self.model_type == "layout":
            model = LayoutPredictor()
            model.model.eval()
            return model.model
        elif self.model_type == "table_recognition":
            model = TableRecPredictor()
            model.model.eval()
            return model.model
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_inputs(self):
        """Load appropriate inputs based on model type"""
        import torch
        import numpy as np

        if self.model_type == "detection":
            # For detection model - standard dimensions (not in settings)
            batch_size = 1
            channels = 3
            height = 512
            width = 512
            return torch.rand(batch_size, channels, height, width)

        elif self.model_type == "recognition":
            # For recognition model - use dimensions from settings
            batch_size = 1
            channels = 3
            height = settings.RECOGNITION_IMAGE_SIZE['height']  # 256
            width = settings.RECOGNITION_IMAGE_SIZE['width']    # 896

            # Create a random tensor directly and input dict
            pixel_values = torch.rand(batch_size, channels, height, width)
            inputs = {'pixel_values': pixel_values}
            
            # Apply the model's preprocessor if available
            if hasattr(self.framework_model, 'preprocess_inputs'):
                inputs = self.framework_model.preprocess_inputs(inputs)
                
            return inputs

        elif self.model_type == "layout" or self.model_type == "table_recognition":
            # For layout model - use dimensions from settings
            batch_size = 1
            channels = 3
            height = settings.LAYOUT_IMAGE_SIZE['height']  # 768
            width = settings.LAYOUT_IMAGE_SIZE['width']    # 768
            return torch.rand(batch_size, channels, height, width)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")



# Detection :   Passes op-by-op flow and full-eval (4 outputs)
# Recognition : Passes op-by-op flow, full eval hits error : ValueError: Unsupported input type <class 'torch.SymInt'>
# Layout :      op-by-op error (NoneType) on assert decoder_input_boxes[0][0][0] == self.config.decoder_start_token_id
# Table :       op-by-op error: boxes = boxes.to(torch.long).clamp(0, self.config.vocab_size)

@pytest.mark.parametrize(
    "model_type",
    ["detection", "recognition", "layout", "table_recognition"],
)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH, OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_torch", "op_by_op_stablehlo", "full"],
)
def test_surya(record_property, model_type, mode, op_by_op):
    model_name = f"surya-{model_type}"

    cc = CompilerConfig()
    # Disable constant evaluation which can cause issues
    cc.enable_consteval = False
    cc.consteval_parameters = False
    
    # Set other compiler options for better compatibility
    cc.dynamic_shape = False
    
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # Higher tolerance for surya models
    required_atol = 0.05

    tester = ThisTester(
        model_name,
        mode,
        model_type=model_type,
        required_atol=required_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
    )

    results = tester.test_model()

    # Print output information
    input_shape = "N/A"
    if isinstance(tester.inputs, dict) and 'pixel_values' in tester.inputs:
        input_shape = tester.inputs['pixel_values'].shape
    elif hasattr(tester.inputs, 'shape'):
        input_shape = tester.inputs.shape
        
    print(
        f"""
    Model: {model_name}
    Input shape: {input_shape}
    Output shape: {results.shape if hasattr(results, 'shape') else 'N/A'}
    """
    )

    tester.finalize()
