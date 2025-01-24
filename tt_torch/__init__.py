# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Process start method must be "forkserver" for op-by-op compilation for the following reasons:
#   - The backend will hang if attempting to retrieve the device descriptor from a forked process if there are multiple chips
#   - torch tensors cannot be made contiguous in a forked process
import multiprocessing as mp

if mp.get_start_method() != "forkserver":
    mp.set_start_method("forkserver", force=True)
