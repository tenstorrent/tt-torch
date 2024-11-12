# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re


class SystemDesc:
    def __init__(self, system_desc_str):
        self.arch = None
        self.grid = None
        self.l1_size = None
        self.num_dram_channels = None
        self.dram_channel_size = None
        self.noc_l1_address_align_bytes = None
        self.pcie_address_align_bytes = None
        self.noc_dram_address_align_bytes = None
        self.l1_unreserved_base = None
        self.erisc_l1_unreserved_base = None
        self.dram_unreserved_base = None
        self.dram_unreserved_end = None
        self.worker_physical_cores = None
        self.dram_physical_cores = None
        self.supported_data_types = None
        self.supported_tile_sizes = None
        self.num_cbs = None

        self.parse(system_desc_str)

    def parse(self, system_desc_str):
        # Parse the system description string and populate the class attributes
        match = re.search(r"arch = <(\w+)>", system_desc_str)
        self.arch = match.group(1) if match else None

        match = re.search(r"grid = (\d+)x(\d+)", system_desc_str)
        if match:
            self.grid = (int(match.group(1)), int(match.group(2)))

        # Add more parsing logic for other attributes...

        # Parse physical cores
        worker_cores = re.findall(r"worker = \[(.*?)\]", system_desc_str)
        if worker_cores:
            self.worker_physical_cores = [
                core.strip() for core in worker_cores[0].split(",")
            ]

        dram_cores = re.findall(r"dram = \[(.*?)\]", system_desc_str)
        if dram_cores:
            self.dram_physical_cores = [
                core.strip() for core in dram_cores[0].split(",")
            ]

        # Parse supported data types and tile sizes
        self.supported_data_types = re.findall(r"<(\w+)>", system_desc_str)
        self.supported_tile_sizes = re.findall(r"(\d+)x(\d+)", system_desc_str)

        # Parse num_cbs
        match = re.search(r"num_cbs = (\d+)", system_desc_str)
        self.num_cbs = int(match.group(1)) if match else None


class Layout:
    def __init__(self, layout_str):
        self.id = None
        self.mapping_from = None
        self.mapping_to = None
        self.undef = None
        self.memory_config = None
        self.buffer_type = None

        self.parse(layout_str)

    def parse(self, layout_str):
        id_match = re.search(r"#layout\d*", layout_str)
        if id_match:
            self.id = id_match.group(0)

        match = re.search(r"#tt\.layout<\((.*?)\) -> \((.*?)\)", layout_str)
        if match:
            self.mapping_from = tuple(arg.strip() for arg in match.group(1).split(","))
            self.mapping_to = tuple(arg.strip() for arg in match.group(2).split(","))
        else:
            self.mapping_from = None
            self.mapping_to = None

        self.undef = "undef" in layout_str

        match = re.search(r"<1x1>, memref<(\d+)x(\d+)x(\w+), #(\w+)>", layout_str)
        if match:
            self.memory_config = (
                int(match.group(1)),
                int(match.group(2)),
                match.group(3),
                match.group(4),
            )
        else:
            match = re.search(
                r"memref<(\d+)x(\d+)x!tt.tile<(\d+)x(\d+), (\w+)>, #(\w+)>", layout_str
            )
            if match:
                self.memory_config = (
                    int(match.group(1)),
                    int(match.group(2)),
                    f"tile<{match.group(3)}x{match.group(4)}, {match.group(5)}>",
                    match.group(6),
                )
            else:
                self.memory_config = None

        self.buffer_type = "interleaved" if "interleaved" in layout_str else None


class TTNNOps:
    def __init__(self, ttnn_str):
        self.system_desc = None
        self.layouts = {}
        self.ops = []

        self.parse(ttnn_str)

    def parse(self, ttnn_str):
        lines = ttnn_str.strip().split("\n")

        # Parse system description
        system_desc_lines = []
        for line in lines:
            if line.startswith("#"):
                system_desc_lines.append(line)
            else:
                break
        self.system_desc = SystemDesc("".join(system_desc_lines))

        # Parse layouts
        layout_pattern = r"#layout\d* = #tt.layout<(.*)>"
        for line in lines:
            match = re.search(layout_pattern, line)
            if match:
                myLayout = Layout(line)
                self.layouts[myLayout.id] = myLayout

        # Parse ops
        in_module = False
        current_op = {}
        for line in lines:
            if line.startswith("module"):
                in_module = True
            elif in_module:
                if line.startswith("return"):
                    in_module = False
                else:
                    id_match = re.search(r"(%\w+) =", line)
                    name_match = re.search(r'"(\w+\.\w+)"', line)
                    args_match = re.search(r"\((.*?)\) <\{", line)
                    attributes_match = re.search(r"<\{(.*?)\}>", line)
                    input_shape_match = re.search(r": \((.*?)\) ->", line)
                    output_shape_match = re.search(r"-> (.*)", line)
                    if id_match:
                        args = (
                            [arg.strip() for arg in args_match.group(1).split(", ")]
                            if args_match and args_match.group(1).strip()
                            else []
                        )
                        input_shapes = (
                            self.split_shapes(input_shape_match.group(1))
                            if input_shape_match
                            and input_shape_match.group(1).strip() != ""
                            else []
                        )
                        output_shapes = (
                            self.split_shapes(output_shape_match.group(1).strip())
                            if output_shape_match
                            and output_shape_match.group(1).strip() != ""
                            else []
                        )
                        current_op = {
                            "id": id_match.group(1) if id_match else None,
                            "name": name_match.group(1) if name_match else None,
                            "args": args,
                            "attributes": self.parse_attributes(
                                attributes_match.group(1)
                            )
                            if attributes_match
                            else {},
                            "input_shapes": input_shapes,
                            "output_shapes": output_shapes,
                        }
                        self.ops.append(current_op)
                    else:
                        print(f"Line did not match expected pattern: {line}")

    def parse_attributes(self, attr_str):
        attributes = {}
        length = len(attr_str)
        i = 0

        while i < length:
            # Find the key
            key_end = attr_str.find("=", i)
            if key_end == -1:
                break
            key = attr_str[i:key_end].strip()

            # Move to value part
            i = key_end + 1
            value_start = i
            bracket_count = {
                "<": 0,
                ">": 0,
                "[": 0,
                "]": 0,
                "(": 0,
                ")": 0,
                "{": 0,
                "}": 0,
            }

            while i < length:
                char = attr_str[i]
                if char in bracket_count:
                    if char in "<[{(":
                        bracket_count[char] += 1
                    else:
                        matching_char = {">": "<", "]": "[", ")": "(", "}": "{"}[char]
                        if bracket_count[matching_char] > 0:
                            bracket_count[matching_char] -= 1

                # Stop if we find a comma that is not inside any brackets
                if char == "," and not any(bracket_count.values()):
                    break
                i += 1

            value = attr_str[value_start:i].strip()

            # Update the index to the next attribute
            i += 1

            attributes[key] = self.parse_value(value)
        return attributes

    def parse_value(self, value):
        if value.startswith("#ttnn"):
            return value
        elif value.isdigit():
            return int(value)
        elif value in ["true", "false"]:
            return value == "true"
        else:
            return value

    def split_shapes(self, shapes_str):
        shapes = []
        current_shape = []
        brackets = 0

        for char in shapes_str:
            if char == "<":
                brackets += 1
            elif char == ">":
                brackets -= 1

            if char == "," and brackets == 0:
                shapes.append("".join(current_shape).strip())
                current_shape = []
            else:
                current_shape.append(char)

        if current_shape:
            shapes.append("".join(current_shape).strip())

        return shapes
