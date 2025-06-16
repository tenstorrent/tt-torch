# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import psutil
import os
import torch


def print_memory_usage(message=""):
    """
    Print current memory usage statistics.

    Args:
        message: Optional message to include in the output for context
    """
    # Get process memory info
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Calculate memory usage
    ram_usage = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB

    # Get CUDA memory info if available
    cuda_memory_str = ""
    if torch.cuda.is_available():
        cuda_allocated = torch.cuda.memory_allocated() / (
            1024 * 1024 * 1024
        )  # Convert to GB
        cuda_reserved = torch.cuda.memory_reserved() / (
            1024 * 1024 * 1024
        )  # Convert to GB
        cuda_memory_str = f", CUDA allocated: {cuda_allocated:.2f} GB, CUDA reserved: {cuda_reserved:.2f} GB"

    # Print memory usage with message
    print(f"MEMORY [{message}] - RAM: {ram_usage:.2f} GB{cuda_memory_str}")


def print_system_memory_summary(message=""):
    mem = psutil.virtual_memory()
    print(
        f"SYS_MEMORY [{message}] - Total: {mem.total / (1024*1024):.2f} MB, Available: {mem.available / (1024*1024):.2f} MB, Used: {mem.used / (1024*1024):.2f} MB, Free: {mem.free / (1024*1024):.2f} MB"
    )
