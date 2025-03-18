# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
import ast

crashsafe_suffix = "_crashsafe.xml"


def get_achieved_compile_depths(xml_file):
    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find all <property> elements with name="achieved_compile_depth"
        compile_depths = [
            prop.attrib["value"]
            for prop in root.findall(".//property[@name='achieved_compile_depth']")
        ]

        # Return the list of values or "UNKNOWN" if none are found
        return compile_depths if compile_depths else ["UNKNOWN"]
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return ["UNKNOWN"]


def get_max_achieved_compile_depth(xml_file):
    achieved_depth_mapping = {
        "UNKNOWN": 0,
        "STABLEHLO": 1,
        "TTNN_IR": 2,
        "EXECUTE": 3,
        "PASSED": 4,
    }
    reverse_compile_depth_mapping = {v: k for k, v in achieved_depth_mapping.items()}
    compile_depths = get_achieved_compile_depths(xml_file)
    numeric_depths = [achieved_depth_mapping.get(depth, 0) for depth in compile_depths]
    max_numeric_depth = max(numeric_depths)
    max_achieved_depth = reverse_compile_depth_mapping[max_numeric_depth]

    return max_achieved_depth


def check_valid_xml(xml_file):
    required_property_names = ["frontend", "model_name", "owner", "group"]

    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Check if each required property exists
        for property_name in required_property_names:
            xpath = f".//property[@name='{property_name}']"
            if root.find(xpath) is None:
                raise AssertionError(
                    f"Property with name='{property_name}' does not exist in the XML."
                )

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    except AssertionError as e:
        print(f"Assertion failed: {e}")


def inject_param_into_tags(xml_file, tag_name, tag_value):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the <property> element with name="tags"
    tags_property = root.find(".//property[@name='tags']")
    if tags_property is not None:
        # Parse the existing tags value as a dictionary
        tags_dict = ast.literal_eval(tags_property.attrib["value"])
    else:
        # Create a new tags dictionary if it doesn't exist
        tags_dict = {}

    # Inject the max achieved compile depth into the tags dictionary
    tags_dict[tag_name] = tag_value

    # Update or create the <property> element for tags
    if tags_property is not None:
        tags_property.set("value", str(tags_dict))
    else:
        # Find the <properties> element to add the new <property>
        properties_element = root.find(".//properties")
        ET.SubElement(properties_element, "property", name="tags", value=str(tags_dict))

    # Write the updated XML back to the file
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)


def rewrite_crashsafe_xml(xml_file):
    check_valid_xml(xml_file)
    max_achieved_compile_depth = get_max_achieved_compile_depth(xml_file)
    inject_param_into_tags(
        xml_file, "max_achieved_compile_depth", max_achieved_compile_depth
    )
