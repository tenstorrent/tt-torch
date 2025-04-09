# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import time
import logging
import shutil


def reset_board(device_id=0, max_attempts=30, sleep_seconds=1):
    """
    Attempts to reset a hung silicon board using tt-smi-metal.

    Args:
        device_id (int): Device ID to reset (default: 0)
        max_attempts (int): Maximum number of reset attempts (default: 30)
        sleep_seconds (int): Seconds to wait between attempts (default: 1)

    Returns:
        bool: True if reset was successful, False otherwise
    """
    print(f"Attempting to reset board (device {device_id})...")

    for i in range(max_attempts):
        try:

            # TODO Check and see if this tool even exists, otherwise abort
            if not shutil.which("tt-smi-metal"):
                print(
                    "tt-smi-metal not found. Please install tt-smi-metal to reset the board."
                )
                return False

            # Execute the tt-smi-metal reset command
            result = subprocess.run(
                ["tt-smi-metal", "-r", str(device_id)],
                capture_output=True,
                text=True,
                check=False,
            )

            # Check if the reset was successful
            if (
                result.returncode != 0
                or "No chips detected" in result.stdout
                or "No chips detected" in result.stderr
            ):
                print(
                    f"Warning: Unsuccessful board reset attempt {i+1}/{max_attempts}, trying again in {sleep_seconds} second(s)..."
                )
                time.sleep(sleep_seconds)
                continue
            else:
                print("Board reset successful!")
                return True

        except Exception as e:
            print(f"Error during reset attempt: {str(e)}")
            time.sleep(sleep_seconds)

    # If we've exhausted all attempts
    print(f"Failed to reset board after {max_attempts} attempts")

    return False
