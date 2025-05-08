# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from bi_lstm_crf import BiRnnCrf
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def __init__(self, *args, rnn_type="lstm", **kwargs):

        # Some arbitrary sanity test values, and configurable rnn.
        self.model_config = {
            "rnn_type": rnn_type.lower(),
            "vocab_size": 30000,
            "tagset_size": 20,
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_rnn_layers": 2,
        }

        super(ThisTester, self).__init__(*args, **kwargs)

    def _load_model(self):
        # Create the model with random weights
        model = BiRnnCrf(
            vocab_size=self.model_config["vocab_size"],
            tagset_size=self.model_config["tagset_size"],
            embedding_dim=self.model_config["embedding_dim"],
            hidden_dim=self.model_config["hidden_dim"],
            num_rnn_layers=self.model_config["num_rnn_layers"],
            rnn=self.model_config["rnn_type"],
        )

        return model

    def _load_inputs(self):
        # Generate random token ids within vocabulary range
        batch_size = 4
        seq_length = 16
        input_ids = torch.randint(
            0, self.model_config["vocab_size"], (batch_size, seq_length)
        )
        return input_ids


@pytest.mark.parametrize(
    "rnn_type",
    ["lstm", "gru"],
    ids=["lstm", "gru"],
)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_bi_lstm_crf(record_property, rnn_type, mode, op_by_op):
    model_name = f"BiRnnCrf-{rnn_type.upper()}"
    model_group = "red"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_FE_COMPILATION",
        reason="need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.bool' - https://github.com/tenstorrent/tt-torch/issues/724",
        model_group=model_group,
    )

    tester = ThisTester(
        model_name,
        mode,
        rnn_type=rnn_type,
        relative_atol=0.01,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )

    results = tester.test_model()
    emissions, best_tag_sequence = results

    print(
        f"""
        Model: {model_name}
        Input shape: {tester.inputs.shape}
        Output:
          - Emissions shape: {emissions.shape}
          - Best tag sequence length: {len(best_tag_sequence[0])}
        """
    )

    tester.finalize()
