# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_mlir
import re
import multiprocessing as mp
import os


def parse_stablehlo_ops_multiline(stablehlo_code):
    """Parses StableHLO code, handling multi-line operations.

    Args:
        stablehlo_code: The StableHLO code as a string.

    Yields:
        Each op line in the StableHLO code, potentially spanning multiple lines.
    """
    lines = stablehlo_code.splitlines()
    current_op = ""
    last_op_index = -1
    encountered_return = False

    pattern = r"^\s*%\d+ = "

    for i, line in enumerate(lines):
        if encountered_return:
            break  # Stop parsing after encountering "return"

        if re.match(pattern, line):
            # Extract op index
            try:
                op_index = int(line.split()[0][1:])
            except ValueError:
                continue  # Skip lines that don't start with "%<number>"

            # Check if op index is valid (incrementing)
            if op_index <= last_op_index:
                if current_op:
                    yield current_op
                current_op = line + "\n"
            else:
                last_op_index = op_index
                if current_op:
                    yield current_op
                current_op = line + "\n"
        elif current_op:
            current_op += line + "\n"

        if line.startswith("return"):
            encountered_return = True

    # Yield the last op if it exists
    if current_op:
        yield current_op


def compile_stablehlo_op(op_line):
    """Compiles a single StableHLO operation line to TTIR."""
    try:
        ttir = tt_mlir.compile_stable_hlo_to_ttir(op_line)
        print(ttir)
        return ttir
    except Exception as e:
        raise RuntimeError(f"Error: {e} at op_line: {op_line}")


def process_op_line(op_line):
    """Processes a single op line, handling segmentation faults."""
    try:
        result = mp.Process(target=compile_stablehlo_op, args=(op_line,))
        result.start()
        result.join()

        if result.exitcode != 0:
            raise RuntimeError(
                f"Segmentation fault or other critical error at op_line: {op_line}"
            )
    except Exception as e:
        print(f"Error: {e} at op_line: {op_line}")
        return


if __name__ == "__main__":
    shlo_path = "tests/models/mmdetection/stablehlo_output.txt"
    with open(shlo_path, "r") as f:
        shlo_code = f.read()

    for op_line in parse_stablehlo_ops_multiline(shlo_code):
        process_op_line(op_line)
