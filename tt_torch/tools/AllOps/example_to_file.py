# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# import pandas lib as pd
import pandas as pd
import os
import getpass

user = getpass.getuser()
# read by default 1st sheet of an excel file
excel_path = "/localdev/" + user + "/tt-torch/tt_torch/tools/AllOps/AllOpsNov12.xlsx"
df = pd.read_excel(excel_path)
df_cleaned = df.dropna(subset=["Raw TTNNIR"])
df_final = df_cleaned[["Torch Name", "Raw TTNNIR"]]
output_file_path = (
    "/localdev/" + user + "/tt-torch/tt_torch/tools/AllOps/ParsedXLSX.xlsx"
)
df_final.to_excel(output_file_path, index=False)

output_dir = "/localdev/" + user + "/tt-torch/tt_torch/tools/AllOps/"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through the DataFrame and write to files
last_torch_name = None
for index, row in df_final.iterrows():
    torch_name = row["Torch Name"] if pd.notna(row["Torch Name"]) else last_torch_name
    raw_ttnnir = row["Raw TTNNIR"]
    if raw_ttnnir.startswith('"') and raw_ttnnir.endswith('"'):
        breakpoint()
        raw_ttnnir = raw_ttnnir[1:-1]
    last_torch_name = torch_name  # Update last_torch_name for the next iteration

    # Create the filename
    filename = os.path.join(output_dir, f"{torch_name}-{index}.txt")

    # Write the content to the file
    with open(filename, "w") as file:
        file.write(raw_ttnnir)

print("Files created successfully.")
print("Cleaned file saved as:", output_file_path)
