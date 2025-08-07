# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tt_torch.dynamo.backend import BackendOptions


def main():
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    config.quantization_config["quant_method"] = "none"
    config.num_hidden_layers = 1
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    options = BackendOptions()
    options.compiler_config = cc

    with torch.no_grad():
        tt_model = torch.compile(model, backend="tt", dynamic=False, options=options)
        outputs = tt_model(**inputs)


if __name__ == "__main__":
    main()
