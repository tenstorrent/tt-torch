# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.d_fine.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)


# Print available variants for reference
available_variants = ModelLoader.query_available_variants()
print("Available variants: ", [str(k) for k in available_variants.keys()])


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "variant,variant_config",
    available_variants.items(),
    ids=[str(k) for k in available_variants.keys()],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_d_fine(record_property, variant, variant_config, mode, op_by_op):

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info(variant=variant)
    model_name = model_info.name

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_FE_COMPILATION",
        reason="need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.int' - https://github.com/tenstorrent/tt-torch/issues/724",
    )

    tester = ThisTester(
        model_name,
        mode,
        loader=loader,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    if mode == "eval":
        results = tester.processor.post_process_object_detection(
            results,
            target_sizes=[(tester.image.height, tester.image.width)],
            threshold=0.5,
        )
        for result in results:
            for score, label_id, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                score, label = score.item(), label_id.item()
                box = [round(i, 2) for i in box.tolist()]
                print(f"{tester.model.config.id2label[label]}: {score:.2f} {box}")

    tester.finalize()
