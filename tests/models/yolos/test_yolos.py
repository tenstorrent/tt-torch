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
        # Local cache of http://images.cocodataset.org/val2017/000000039769.jpg
        image_file = get_file("test_images/coco_two_cats_000000039769_640x480.jpg")
        self.image = Image.open(str(image_file))
        inputs = self.image_processor(images=self.image, return_tensors="pt")
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
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_yolos(record_property, mode, op_by_op, data_parallel_mode):
    model_name = "YOLOS"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
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
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()
    if mode == "eval":
        # Helper function to decode output to human-readable text
        def decode_output(outputs):
            target_sizes = torch.tensor([tester.image.size[::-1]])
            results = tester.image_processor.post_process_object_detection(
                outputs, threshold=0.9, target_sizes=target_sizes
            )[0]
            return results

        def interpret_results(decoded_output):
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
                return string

        def print_result(result):
            decoded_output = decode_output(result)
            print(
                f"""
            model_name: {model_name}
            input_url: {tester.test_input}
            answer before: {interpret_results(decoded_output)}
            """
            )

        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
