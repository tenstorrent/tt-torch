from tests.utils import ModelTester
from transformers import AutoModelForCausalLM, AutoTokenizer
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

import torch
import pprint

class ThisTester(ModelTester):
    def _load_model(self):
        checkpoint = "Salesforce/codegen-350M-mono"
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return model.generate

    def _load_inputs(self):
        text = "def hello_world():"
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs

    def set_model_eval(self, model):
        return model

def test_inputs():
    cc = CompilerConfig()

    tester = ThisTester(
        "foo",
        "eval",
        compiler_config=cc,
        is_token_output=True,
    )

    tester._load_model()
    inputs = tester._load_inputs()

    print("Inputs:")
    pprint.pprint(inputs)