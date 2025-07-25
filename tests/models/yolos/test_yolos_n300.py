# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from PIL import Image

# Load model directly
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        # Download model from cloud
        model_name = "hustvl/yolos-tiny"
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
        )
        m = AutoModelForObjectDetection.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        return m

    def _load_inputs(self):
        # Set up sample input
        self.test_input = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(str(get_file(self.test_input)))
        inputs = self.image_processor(images=[self.image] * 4, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_yolos(record_property, mode, op_by_op):
    model_name = "YOLOS"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        required_pcc=0.98,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        # FIXME fails with tt-experimental - https://github.com/tenstorrent/tt-torch/issues/1105
        backend="tt",
    )
    results = tester.test_model()
    if mode == "eval":
        # Helper function to decode output to human-readable text
        def decode_output(outputs):
            batch_size = outputs["logits"].shape[0]
            target_sizes = torch.tensor([tester.image.size[::-1]] * batch_size)
            results = tester.image_processor.post_process_object_detection(
                outputs, threshold=0.9, target_sizes=target_sizes
            )
            return results

        decoded_outputs = decode_output(results)

        def interpret_results(decoded_output):
            strings = []
            for score, label, box in zip(
                decoded_output["scores"],
                decoded_output["labels"],
                decoded_output["boxes"],
            ):
                box = [round(i, 2) for i in box.tolist()]
                string = (
                    f"Detected {tester.framework_model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                strings.append(string)
            return strings

        for i, decoded_output in enumerate(decoded_outputs):
            print(
                f"""
            model_name: {model_name}
            input_url: {tester.test_input}
            sample: {i}
            answers: {interpret_results(decoded_output)}
            """
            )

    tester.finalize()
