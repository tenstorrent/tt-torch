# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import time
import shutil
import argparse
import sys
import os


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

    # Look for tt-smi-metal in PATH
    tt_smi_path = shutil.which("tt-smi-metal")

    # If not found in PATH, try some common locations
    if not tt_smi_path:
        potential_paths = [
            "/usr/bin/tt-smi-metal",
            "/usr/local/bin/tt-smi-metal",
            "/opt/tenstorrent/bin/tt-smi-metal",
        ]
        for path in potential_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                tt_smi_path = path
                break

    if not tt_smi_path:
        print("tt-smi-metal not found in PATH or common locations.")
        # Try to find any tt-smi* binaries to help with debugging
        try:
            find_result = subprocess.run(
                ["find", "/", "-name", "tt-smi*", "-type", "f"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if find_result.stdout:
                print("Found potential tt-smi binaries:")
                print(find_result.stdout)
        except Exception as e:
            print(f"Error while searching for tt-smi binaries: {e}")

        return False

    print(f"Using tt-smi-metal at: {tt_smi_path}")

    for i in range(max_attempts):
        try:
            # Execute the tt-smi-metal reset command
            result = subprocess.run(
                [tt_smi_path, "-r", str(device_id)],
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
                if result.stderr:
                    print(f"Error output: {result.stderr}")
                if result.stdout:
                    print(f"Standard output: {result.stdout}")
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


def main():
    parser = argparse.ArgumentParser(description="Reset a Tenstorrent silicon board")
    parser.add_argument(
        "--device", "-d", type=int, default=0, help="Device ID to reset (default: 0)"
    )
    parser.add_argument(
        "--attempts",
        "-a",
        type=int,
        default=30,
        help="Maximum number of reset attempts (default: 30)",
    )
    parser.add_argument(
        "--sleep",
        "-s",
        type=int,
        default=1,
        help="Seconds to wait between attempts (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print additional diagnostic information",
    )

    args = parser.parse_args()

    if args.verbose:
        print("System PATH:")
        print(os.environ.get("PATH", "PATH not set"))

        print("\nCurrent working directory:")
        print(os.getcwd())

        print("\nDirectory contents:")
        try:
            print(subprocess.check_output(["ls", "-la"]).decode())
        except Exception as e:
            print(f"Error listing directory: {e}")

    success = reset_board(
        device_id=args.device, max_attempts=args.attempts, sleep_seconds=args.sleep
    )

    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
