# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# run as
# python3 tt_torch/tools/dummy_orchestrator.py
from tt_torch.tools.profile_util import Profiler
import pytest
import subprocess
import os
import signal
import sys
from argparse import ArgumentParser


def profile(test_command: str, output_filename: str = Profiler.DEFAULT_OUTPUT_FILENAME):
    profiler = Profiler(output_filename)
    profiler.assert_perf_build()
    profiler.setup_tracy_server()

    env_vars = os.environ.copy()

    # Test process must not run in main process, or else tracy capture will
    # deadlock this.
    testProcess = subprocess.Popen(
        [test_command], shell=True, env=env_vars, preexec_fn=os.setsid
    )

    def signal_handler(sig, frame):
        print("sig handler got invoked")
        os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
        profiler.tracy_capture_tool_process.terminate()
        profiler.tracy_capture_tool_process.communicate()
        sys.exit(3)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    testProcess.communicate()  # block until the test process exits

    profiler.close_capture_tool_process()
    profiler.process_csvexport()
    profiler.copy_files_to_tt_metal()
    profiler.run_ttmetal_process_ops()
    profiler.post_process_ops()
    profiler.cleanup()


if __name__ == "__main__":
    """
    This script is used to wrap pytests and generate Tracy profiling data for the tests,
    gathering device-side performance data associated with individual torchfx ops.

    Usage:
        python profile.py test_command

    Examples:
        python tt_torch/tools/profile.py "pytest -svv tests/models/mnist/test_mnist.py::test_mnist_train[full-eval]"

    Notes:
        - You must provide either --excel_path or --json_path, but not both.
    """

    parser = ArgumentParser()
    parser.add_argument("test_command", type=str, help="The test command to run")
    parser.add_argument(
        "-o",
        "--output_path",
        help="Output file path",
        type=str,
        default=Profiler.DEFAULT_OUTPUT_FILENAME,
    )

    args = parser.parse_args()

    profile(args.test_command, args.output_path)
