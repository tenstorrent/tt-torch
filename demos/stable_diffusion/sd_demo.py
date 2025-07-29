# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import argparse
import tt_torch
from pathlib import Path
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend, BackendOptions
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny
from transformers import CLIPTextModel, CLIPTokenizer
from tt_torch.tools.device_manager import DeviceManager

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)


def main(run_interactive):

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = BackendOptions()
    options.compiler_config = cc

    pipe.enable_attention_slicing()
    import torch_xla.core.xla_model as xm

    pipe.transformer = pipe.transformer.eval().to(torch.bfloat16)

    # Currently, the transformer is the only model in the pipeline which can be compiled.
    compiled_fwd = torch.compile(
        pipe.transformer.forward, backend="tt-experimental", options=options
    )

    # We convert to bfloat16 to save memory, but we convert back to float32 for the output.
    # We do this because the rest of the pipeline is run on the host CPU, and float32 math
    # is much faster than bfloat16 on CPU.
    def transformer_wrapper(*args, **kwargs):
        kwargs["hidden_states"] = kwargs["hidden_states"].to(torch.bfloat16)
        kwargs["timestep"] = kwargs["timestep"].to(torch.bfloat16)
        kwargs["encoder_hidden_states"] = kwargs["encoder_hidden_states"].to(
            torch.bfloat16
        )
        kwargs["pooled_projections"] = kwargs["pooled_projections"].to(torch.bfloat16)
        out = compiled_fwd(*args, **kwargs)
        return tuple(t.to(torch.float32) for t in out)

    pipe.transformer.forward = transformer_wrapper

    def generate_image(prompt, output_path):
        img = pipe(
            prompt, num_inference_steps=50, height=512, width=512, guidance_scale=7.0
        ).images[0]
        img.save(output_path)

    if not run_interactive:
        prompt = "An astronaut riding a horse on mars"
        output_path = f"{Path.cwd()}/{prompt.replace(' ', '_')}.png"
        print(f"File will be saved to {output_path}")
        generate_image(prompt, output_path)
    else:
        cmd_prompt = 'Enter the prompt for the image (type "stop" to exit): '
        prompt = input(cmd_prompt)
        while prompt != "stop":
            output_path = f"{Path.cwd()}/{prompt.replace(' ', '_')}.png"
            print(f"File will be saved to {output_path}")
            generate_image(prompt, output_path)
            prompt = input(cmd_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_interactive",
        action="store_true",
        help="Run the demo interactively as opposed to once with the default image.",
    )
    args = parser.parse_args()
    main(args.run_interactive)
