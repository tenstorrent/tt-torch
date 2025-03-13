# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# run as
# python3 tt_torch/tools/dummy_orchestrator.py
import perf
import pytest
import subprocess
import os
import signal
import sys

use_test_subprocess = True


def main():
    cvar = perf.Perf()
    cvar.setup_tracy_server()

    if use_test_subprocess:
        # canonical subprocess invocation
        env_vars = os.environ.copy()
        test_command = (
            "pytest -svv tests/models/mnist/test_mnist.py::test_mnist_train[full-eval]"
        )
        # test_command = "pytest -svv tests/torch/test_basic.py::test_linear"

        testProcess = subprocess.Popen(
            [test_command], shell=True, env=env_vars, preexec_fn=os.setsid
        )

        def signal_handler(sig, frame):
            print("sig handler got invoked")
            os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
            cvar.tracy_capture_tool_process.terminate()
            cvar.tracy_capture_tool_process.communicate()
            sys.exit(3)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        testProcess.communicate()  # block until the test process exits (tracy server should then auto disconnect)

    cvar.close_capture_tool_process()
    cvar.process_csvexport()
    cvar.copy_files_to_tt_metal()
    cvar.run_ttmetal_process_ops()
    cvar.post_process_ops()
    cvar.cleanup()


if __name__ == "__main__":
    main()
