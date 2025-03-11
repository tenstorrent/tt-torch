# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# run as
# python3 tt_torch/tools/dummy_orchestrator.py
import perf


def __main__():
    cvar = perf.Perf()
    cvar.setup_tracy_server()


__main__()
