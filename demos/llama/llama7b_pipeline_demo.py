# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import argparse
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from tt_mlir import (
    open_mesh_device,
    MeshDeviceOptions,
)


def get_model_and_tokenizer():
    # Download model from cloud
    model_name = "huggyllama/llama-7b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    for param in m.parameters():
        param.requires_grad = False
    m.eval()
    return m, tokenizer


def get_sample_input(tokenizer, test_input):
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )
    return inputs


def decode_output(outputs, tokenizer):
    next_token_logits = outputs.logits[:, -1]
    next_token = next_token_logits.softmax(dim=-1).argmax()
    return tokenizer.decode([next_token])


class LlamaFirstHalf(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        assert self.original_model.model.config.num_hidden_layers % 2 == 0
        self.num_layers = self.original_model.model.config.num_hidden_layers // 2

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.original_model.model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.original_model.model.config.output_hidden_states
        )

        use_cache = (
            use_cache
            if use_cache is not None
            else self.original_model.model.config.use_cache
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if (
            self.original_model.model.gradient_checkpointing
            and self.original_model.model.training
            and use_cache
        ):
            use_cache = False

        assert not output_attentions
        assert not output_hidden_states

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            inputs_embeds = self.original_model.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.original_model.model._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.original_model.model.rotary_emb(
            hidden_states, position_ids
        )

        for decoder_layer in self.original_model.model.layers[: self.num_layers]:
            assert not (
                self.original_model.model.gradient_checkpointing
                and self.original_model.model.training
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

        return hidden_states


class LlamaSecondHalf(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        assert self.original_model.model.config.num_hidden_layers % 2 == 0
        self.num_layers = self.original_model.model.config.num_hidden_layers // 2

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        assert (input_ids is not None) and (hidden_states is not None)
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.original_model.model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.original_model.model.config.output_hidden_states
        )

        use_cache = (
            use_cache
            if use_cache is not None
            else self.original_model.model.config.use_cache
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if (
            self.original_model.model.gradient_checkpointing
            and self.original_model.model.training
            and use_cache
        ):
            use_cache = False

        assert not output_attentions
        assert not output_hidden_states

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            inputs_embeds = self.original_model.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.original_model.model._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.original_model.model.rotary_emb(
            inputs_embeds, position_ids
        )

        # Process through the second half of layers
        for decoder_layer in self.original_model.model.layers[self.num_layers :]:
            assert not (
                self.original_model.model.gradient_checkpointing
                and self.original_model.model.training
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

        # Apply the final normalization
        hidden_states = self.original_model.model.norm(hidden_states)

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(),
            attentions=tuple(),
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.original_model.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# This should throw an error at runtime
# Since it shouldn't fit on a single device
def main_single_device(prompt):
    cc = CompilerConfig()
    original_model, tokenizer = get_model_and_tokenizer()
    test_input = get_sample_input(tokenizer, prompt)
    tt_model = torch.compile(original_model, backend=backend, dynamic=False, options=cc)
    results = tt_model(**test_input)
    output = decode_output(results, tokenizer)
    return output


def main_split_single_device(prompt):
    cc = CompilerConfig()
    original_model, tokenizer = get_model_and_tokenizer()
    # # Create the two model halves
    first_half = LlamaFirstHalf(original_model)
    test_input = get_sample_input(tokenizer, prompt)
    tt_first_half = torch.compile(
        first_half, backend=backend, dynamic=False, options=cc
    )
    hidden_states = tt_first_half(**test_input)

    second_half = LlamaSecondHalf(original_model)
    tt_second_half = torch.compile(
        second_half, backend=backend, dynamic=False, options=cc
    )
    results = tt_second_half(**test_input, hidden_states=hidden_states)
    cc.cleanup_device()
    output = decode_output(results, tokenizer)
    return output


def main_split_multi_device(prompt):
    original_model, tokenizer = get_model_and_tokenizer()
    test_input = get_sample_input(tokenizer, prompt)
    mesh_device_options = MeshDeviceOptions()
    mesh_device_options.enable_async_ttnn = True
    parent_device = open_mesh_device([1, 2], mesh_device_options)

    cc1 = CompilerConfig()
    cc1.initialize_sub_device(parent_device, mesh_shape=[1, 1], mesh_offset=[0, 0])
    first_half = LlamaFirstHalf(original_model)
    tt_first_half = torch.compile(
        first_half, backend=backend, dynamic=False, options=cc1
    )

    cc2 = CompilerConfig()
    cc2.initialize_sub_device(parent_device, mesh_shape=[1, 1], mesh_offset=[0, 1])
    second_half = LlamaSecondHalf(original_model)
    tt_second_half = torch.compile(
        second_half, backend=backend, dynamic=False, options=cc2
    )

    hidden_states = tt_first_half(**test_input)
    results = tt_second_half(**test_input, hidden_states=hidden_states)

    output = decode_output(results, tokenizer)
    cc1.cleanup_sub_device()
    cc2.cleanup_sub_device()
    return output


def main_golden(prompt):
    original_model, tokenizer = get_model_and_tokenizer()
    test_input = get_sample_input(tokenizer, prompt)
    results = original_model(**test_input)
    output = decode_output(results, tokenizer)
    return output


def main_split_cpu(prompt):
    original_model, tokenizer = get_model_and_tokenizer()
    # # Create the two model halves
    first_half = LlamaFirstHalf(original_model)
    second_half = LlamaSecondHalf(original_model)

    # Move both halves to CPU explicitly for testing
    first_half = first_half.cpu()
    second_half = second_half.cpu()

    test_input = get_sample_input(tokenizer, prompt)
    hidden_states = first_half(**test_input)
    results = second_half(**test_input, hidden_states=hidden_states)
    output = decode_output(results, tokenizer)
    return output


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run different model execution modes")

    # Create mode argument group
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "single_device",
            "split_single_device",
            "split_multi_device",
            "golden",
            "split_cpu",
        ],
        default="golden",
        help="Execution mode for the model",
    )

    # Add prompt argument
    parser.add_argument(
        "--prompt",
        type=str,
        default="I enjoy walking in the",
        help="Input prompt for text generation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    print(f"Executing with mode {args.mode} and prompt `{args.prompt}`")
    # Dictionary mapping mode names to their respective functions
    mode_functions = {
        "single_device": main_single_device,
        "split_single_device": main_split_single_device,
        "split_multi_device": main_split_multi_device,
        "golden": main_golden,
        "split_cpu": main_split_cpu,
    }

    selected_function = mode_functions[args.mode]
    output = selected_function(args.prompt)

    print("Prompt: ", args.prompt)
    print("Generated Text:", output)
