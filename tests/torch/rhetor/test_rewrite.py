# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tt_torch.tools.crashsafe_utils import *
import xml.etree.ElementTree as ET
import ast


def test_rewrite():
    rewrite_crashsafe_xml("mnist.xml_crashsafe.xml")

    tree = ET.parse("mnist.xml_crashsafe.xml")
    root = tree.getroot()

    # Find the <property> element with name="tags"
    tags_property = root.find(".//property[@name='tags']")
    tags_dict = ast.literal_eval(tags_property.attrib["value"])
    assert (
        "max_achieved_compile_depth" in tags_dict
    ), "Key 'max_achieved_compile_depth' does not exist in tags_dict"
