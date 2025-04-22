# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import time
from tt_torch.tools.verify import calculate_pcc


class CompilerTransformInverter:
    print_debug = True

    def predicate(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        pass

    def invert(self, a: torch.Tensor, b: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        pass


class IdentityInverter(CompilerTransformInverter):
    def predicate(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return True

    def invert(self, a: torch.Tensor, b: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return (a, b)


class ConvStandardInverter(CompilerTransformInverter):
    # Channels Last TTNN
    def predicate(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return (
            a.shape != b.shape
            and a.dim() == 4
            and b.dim() == 4
            and a.shape[0] == b.shape[0]
            and a.shape[2] * a.shape[3] == b.shape[2]
            and a.shape[1] == b.shape[3]
        )

    def invert(self, a: torch.Tensor, b: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        inv_a = a.permute(0, 2, 3, 1)
        inv_a = inv_a.reshape(1, 1, -1, inv_a.shape[-1])
        return (inv_a, b)


class ConvSmallInverter(CompilerTransformInverter):
    def predicate(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return (a.dim() == 4 and b.dim() == 3) and (
            a.shape[0] == b.shape[0] and b.shape[2] != a.shape[1]
        )

    def invert(self, a: torch.Tensor, b: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        inv_a = a.permute(0, 2, 3, 1)
        inv_a = inv_a.reshape(1, 1, -1, b.shape[2])  # sometimes 4, sometimes 91?
        return (inv_a, b)


def calculate_pcc_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return calculate_pcc(a.flatten(), b.flatten())


def try_invert_compiler_transform(
    a: torch.Tensor, b: torch.Tensor, verbose: bool = True
):
    inversion_passes = [IdentityInverter(), ConvStandardInverter(), ConvSmallInverter()]

    best_pcc = -float("inf")
    best_pass = "None"

    for inverter_pass in inversion_passes:
        inv_a, inv_b, _, final_pcc = apply_inverter(
            inverter_pass, a, b, verbose=verbose
        )
        if final_pcc > best_pcc:
            best_pcc = final_pcc
            best_pass = inverter_pass.__class__.__name__

    return best_pcc, best_pass


def main_add():
    demo_tensor = "add_1.pt"
    try_invert_compiler_transform(
        torch.load(f"tensors/gn_{demo_tensor}"), torch.load(f"tensors/rt_{demo_tensor}")
    )


def main_conv():
    demo_tensor = "convolution_75.pt"
    try_invert_compiler_transform(
        torch.load(f"tensors/gn_{demo_tensor}"), torch.load(f"tensors/rt_{demo_tensor}")
    )


def apply_inverter(
    inverter: CompilerTransformInverter,
    a: torch.Tensor,
    b: torch.Tensor,
    verbose: bool = True,
) -> tuple:
    """Apply a CompilerTransformInverter to two tensors

    Args:
        inverter: The CompilerTransformInverter instance to apply
        a: First tensor (typically ground truth)
        b: Second tensor (typically runtime)
        verbose: Whether to print debug information (default: True)

    Returns:
        tuple: (inverted_a, inverted_b, pcc_before, pcc_after)
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Applying {inverter.__class__.__name__} to tensors")
        print(f"{'='*50}")

        # Print original tensor information
        print(f"Original tensor shapes: a={a.shape}, b={b.shape}")
        print(f"Original tensor dtypes: a={a.dtype}, b={b.dtype}")

    # Calculate initial PCC
    initial_pcc = calculate_pcc_flat(a, b)

    if verbose:
        print(f"Initial PCC: {initial_pcc:.6f}")

    # Check if the inverter is applicable
    is_applicable = inverter.predicate(a, b)

    if verbose:
        print(f"Inverter applicable: {is_applicable}")

    if not is_applicable:
        if verbose:
            print(
                "Inverter not applicable to these tensors, returning original tensors"
            )
        return a, b, initial_pcc, initial_pcc

    # Apply the inversion
    if verbose:
        print("Applying inversion...")
    start_time = time.time()
    inv_a, inv_b = inverter.invert(a, b)
    invert_time = time.time() - start_time

    if verbose:
        print(f"Inversion completed in {invert_time:.4f}s")
        # Print inverted tensor information
        print(f"Inverted tensor shapes: inv_a={inv_a.shape}, inv_b={inv_b.shape}")

    # Calculate PCC after inversion
    final_pcc = calculate_pcc_flat(inv_a, inv_b)

    if verbose:
        print(f"Final PCC after inversion: {final_pcc:.6f}")

        # Print improvement
        pcc_change = final_pcc - initial_pcc
        print(
            f"PCC improvement: {pcc_change:.6f} ({pcc_change/max(abs(initial_pcc), 1e-10)*100:.2f}%)"
        )

        print(f"{'='*50}\n")

    return inv_a, inv_b, initial_pcc, final_pcc


if __name__ == "__main__":
    main_conv()
