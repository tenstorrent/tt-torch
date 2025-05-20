# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend, CompileMode


class ThisTester(ModelTester):
    def _load_model(self):
        # Download model from cloud
        model_name = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", torch_dtype=torch.bfloat16
        )
        m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        return m

    def _load_inputs(self):
        # Set up sample input
        self.test_input = "This is a sample text from "
        inputs = self.tokenizer(self.test_input, return_tensors="pt")
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
#@pytest.mark.parametrize(
#    "op_by_op",
#    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
#    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
#)
@pytest.mark.parametrize(
<<<<<<< HEAD
    "compile_mode",
    ["STABLEHLO", "TTNN_IR", "COMPILE_OP_BY_OP_TORCH",
        "COMPILE_OP_BY_OP_SHLO", "EXECUTE_OP_BY_OP_TORCH", "EXECUTE_OP_BY_OP_SHLO", "EXECUTE"],
    ids=["stablehlo", "ttnn_ir", "compile_op_by_op_torch", 
        "compile_op_by_op_shlo", "execute_op_by_op_torch", "execute_op_by_op_shlo", "full"],
=======
     "compile_mode",
     [CompileMode.TORCH_FX, CompileMode.STABLEHLO, CompileMode.TTNN_IR, CompileMode.COMPILE_OP_BY_OP_TORCH, 
       CompileMode.EXECUTE_OP_BY_OP_TORCH, CompileMode.COMPILE_OP_BY_OP_SHLO, CompileMode.EXECUTE_OP_BY_OP_SHLO,  None],
     ids=["torch_fx", "stablehlo", "ttnn_ir", "compile_op_by_op_torch", 
          "execute_op_by_op_torch", "compile_op_by_op_shlo", "execute_op_by_op_shlo", "full"],
>>>>>>> df56077 (made new enum class that aggragates CompileDepth and OpByOpBackend and uses that are the parameter for testing. Also added a decomposing method under that enum to decompose CompileMode into CompileDepth and OpByOpBackend. Currently still only on falcon tests.)
)

def test_falcon(record_property, mode, compile_mode):
    model_name = "Falcon"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    
<<<<<<< HEAD
    if compile_mode in ("STABLEHLO", "TTNN_IR", "COMPILE_OP_BY_OP_TORCH", "EXECUTE_OP_BY_OP_TORCH"):
        # cc.op_by_op_backend = OpByOpBackend.TORCH # Default case
        if compile_mode == "STABLEHLO":
            cc.compile_depth = CompileDepth.STABLEHLO
        elif compile_mode == "TTNN_IR":
            cc.compile_depth = CompileDepth.TTNN_IR
        elif compile_mode == "EXECUTE":
            cc.compile_depth = CompileDepth.EXECUTE
        elif compile_mode == "COMPILE_OP_BY_OP_TORCH":
            cc.compile_depth = CompileDepth.COMPILE_OP_BY_OP
        elif compile_mode == "EXECUTE_OP_BY_OP_TORCH":
            cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
    elif compile_mode in ("COMPILE_OP_BY_OP_SHLO", "EXECUTE_OP_BY_OP_SHLO"):
        cc.op_by_op_backend = OpByOpBackend.STABLEHLO
        if compile_mode == "COMPILE_OP_BY_OP_SHLO":
            cc.compile_depth = CompileDepth.COMPILE_OP_BY_OP
        elif compile_mode == "EXECUTE_OP_BY_OP_SHLO":
            cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
=======
    compile_depth, op_by_op_backend = compile_mode.decompose_CompileMode()

    cc.compile_depth = compile_depth
    cc.op_by_op_backend = op_by_op_backend
>>>>>>> df56077 (made new enum class that aggragates CompileDepth and OpByOpBackend and uses that are the parameter for testing. Also added a decomposing method under that enum to decompose CompileMode into CompileDepth and OpByOpBackend. Currently still only on falcon tests.)

    tester = ThisTester(
        model_name,
        mode,
        relative_atol=0.015,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=False,
        assert_atol=False,
    )
    results = tester.test_model()

    if mode == "eval":
        # Helper function to decode output to human-readable text
        def decode_output(outputs):
            next_token_logits = outputs.logits[:, -1]
            next_token = next_token_logits.softmax(dim=-1).argmax()
            return tester.tokenizer.decode([next_token])

        decoded_output = decode_output(results)

        print(
            f"""
        model_name: {model_name}
        input: {tester.test_input}
        output before: {decoded_output}
        """
        )

    tester.finalize()
