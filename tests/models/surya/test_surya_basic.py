# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

from surya.settings import settings
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.table_rec import TableRecPredictor

class ThisTester(ModelTester):

    def _extract_outputs(self, output_object):
        if hasattr(output_object, 'logits'):
            return (output_object.logits,)
        else:
            model_type = self.model_name.split('-')[-1]
            print(f"DEBUG: Model {model_type} returned output of type {type(output_object)}")
            if isinstance(output_object, tuple):
                return output_object
            return (output_object,)

    # Workaround: Add the missing encoder-to-decoder projection
    def _apply_encoder_decoder_projection_fix(self, model):
        """Add missing encoder-to-decoder projection if needed"""
        if hasattr(model, "encoder") and hasattr(model, "decoder") and not hasattr(model, "enc_to_dec_proj"):
            # Determine dimensions
            encoder_dim = getattr(model.encoder, "hidden_size", 1024)
            decoder_dim = getattr(model.decoder, "hidden_size", 1280)
            print(f"enc_to_dec_proj workaround. Creating projection: {encoder_dim} → {decoder_dim}", flush=True)
            
            # Create and initialize the layer with identity matrix plus zeros
            with torch.no_grad():
                model.enc_to_dec_proj = nn.Linear(encoder_dim, decoder_dim)
                if encoder_dim <= decoder_dim:
                    nn.init.zeros_(model.enc_to_dec_proj.weight)
                    for i in range(min(encoder_dim, decoder_dim)):
                        model.enc_to_dec_proj.weight[i, i] = 1.0
        return model
    
    # Workaround: Add missing attributes to cross-attention modules
    def _apply_cross_attention_fix(self, model):
        """Add missing attributes to cross-attention modules"""
        cross_attn_patched = 0
        
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
                    
        if (cross_attn_patched > 0):
            print(f"cross_attn workaround. Patched {cross_attn_patched} missing attributes.", flush=True)
        return model

    def _load_model(self):
        model_type = self.model_name.split('-')[-1]

        if model_type == "recognition":
            predictor = RecognitionPredictor()
            model = predictor.model
            model = self._apply_encoder_decoder_projection_fix(model)
            model = self._apply_cross_attention_fix(model)
            return model
        elif model_type == "detection":
            predictor = DetectionPredictor()
            return predictor.model
        elif model_type == "layout":
            predictor = LayoutPredictor()
            model = predictor.model
            model = self._apply_cross_attention_fix(model)
            return model
        elif model_type == "table_recognition":
            predictor = TableRecPredictor()
            return predictor.model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _load_inputs(self):
        model_type = self.model_name.split('-')[-1]

        batch_size = 1
        channels = 3

        if model_type == "detection":
            height = 512
            width = 512
            pixel_values = torch.rand(batch_size, channels, height, width)
            return pixel_values

        elif model_type == "recognition":
            height = settings.RECOGNITION_IMAGE_SIZE['height']  # 256
            width = settings.RECOGNITION_IMAGE_SIZE['width']    # 896

            # Create simple tensor with correct dimensions
            pixel_values = torch.rand(batch_size, channels, height, width)
            decoder_input_ids = torch.zeros(batch_size, 1, dtype=torch.long)
            
            return {
                'pixel_values': pixel_values,
                'decoder_input_ids': decoder_input_ids
            }

        elif model_type == "layout":
            height = settings.LAYOUT_IMAGE_SIZE['height']
            width = settings.LAYOUT_IMAGE_SIZE['width']
            pixel_values = torch.rand(batch_size, channels, height, width)

            # Get model-specific configuration values
            decoder_start_token_id = self.framework_model.config.decoder_start_token_id
            
            # Create box inputs - Shape: (batch_size, num_boxes, 7)
            # First dimension must contain decoder_start_token_id
            num_boxes = 3  # Small number for testing
            safe_coord_value = 5  # Small value to avoid embedding index errors
            decoder_input_boxes = torch.ones(batch_size, num_boxes, 7, dtype=torch.long) * safe_coord_value
            decoder_input_boxes[0, 0, 0] = decoder_start_token_id
            
            # Box counts tensor required by model
            decoder_input_boxes_counts = torch.tensor([num_boxes], dtype=torch.long)
            
            return {
                'pixel_values': pixel_values,
                'decoder_input_boxes': decoder_input_boxes,
                'decoder_input_boxes_counts': decoder_input_boxes_counts
            }

        elif model_type == "table_recognition":
            height = settings.TABLE_REC_IMAGE_SIZE['height']  # 768
            width = settings.TABLE_REC_IMAGE_SIZE['width']    # 768
            pixel_values = torch.rand(batch_size, channels, height, width)
            return pixel_values

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


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
    cc.enable_consteval = False
    cc.consteval_parameters = False
    
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # Higher tolerance for surya models
    required_atol = 0.05

    tester = ThisTester(
        model_name,
        mode,
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
