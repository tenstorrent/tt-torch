# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_torch.tools.ci_verification as ci_tools
import subprocess
from tt_torch.tools.utils import FileManager
import pandas as pd
import os


def test_intermediate_verification():
    """
    This test checks the functionality of the intermediate verification option for the following invariants:

    1. That pytests for two short tests can be run without issue with the intermediate verification option set
    2. That report dissector can generate an XLSX report file summarizing both tests
    3. That the XSLX report file is well formed
    4. That the XLSX report file contains reports for both tests
    5. That the report contents show at least one valid operation with non-NaN/inf floating-point ATOL and PCC values
    """

    model_list = [
        "tests/models/autoencoder_linear/test_autoencoder_linear.py::test_autoencoder_linear[full-eval]",
        "tests/models/mnist/test_mnist.py::test_mnist_train[full-eval]",
    ]

    log_dir = "results/intermediate_verification_logs"
    xlsx_report_path = f"{log_dir}/intermediate_verification_results.xlsx"
    try:
        FileManager.create_directory(log_dir, exist_ok=True)
        env = os.environ.copy()
        env["TT_TORCH_VERIFY_INTERMEDIATES"] = "1"
        for i, model in enumerate(model_list):
            log_file = f"{log_dir}/test_intermediate_verification_{i}.log"
            test_command = ["pytest", "-svv", model]
            print("Running command:", " ".join(test_command))
            with open(log_file, "w") as log_file:
                subprocess.run(
                    test_command, shell=False, check=True, env=env, stdout=log_file
                )

        log_parser_command = [
            "python",
            "tt_torch/tools/ci_verification.py",
            "--dissect-report",
            log_dir,
            xlsx_report_path,
        ]
        subprocess.run(log_parser_command, shell=False, check=True)

        # Validate the generated XLSX report
        assert os.path.exists(xlsx_report_path), "XLSX report file was not generated."

        # Open the XLSX file using pandas
        xlsx_data = pd.ExcelFile(xlsx_report_path)

        assert len(xlsx_data.sheet_names) == len(
            model_list
        ), "Number of worksheets does not match the number of models run."

        # Iterate through each worksheet and validate its structure
        for sheet_name in xlsx_data.sheet_names:
            df = xlsx_data.parse(sheet_name)
            print("Printing sheet:", sheet_name)
            print(
                "* note - blank cells are printed as NaN for the purposes of this test."
            )
            print(df.to_string())

            # Assert that the DataFrame has at least one row with valid floating-point ATOL and PCC values
            valid_ops = df[
                pd.to_numeric(df["PCC"], errors="coerce").notnull()
                & pd.to_numeric(  # PCC is numeric
                    df["ATOL"], errors="coerce"
                ).notnull()  # ATOL is numeric
            ]
            assert (
                valid_ops.shape[0] > 0
            ), f"No valid operations with floating-point PCC and ATOL values found in sheet '{sheet_name}'."

            # Assert that required columns exist
            required_columns = ["Node Name", "PCC", "ATOL", "Error Message"]
            for column in required_columns:
                assert (
                    column in df.columns
                ), f"Missing column '{column}' in sheet '{sheet_name}'."
            print("=" * 40)
    finally:
        # Clean up the generated files and directories
        FileManager.remove_directory(log_dir)
