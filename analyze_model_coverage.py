#!/usr/bin/env python3
"""
Script to analyze model coverage between old and new XML test results.
Finds old tests that are equivalent to new model tests based on model name and variant.
"""

import xml.etree.ElementTree as ET
import re
import sys
from typing import Set, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model from test results."""
    model: str
    variant: str
    model_name: str
    test_case_name: str
    parallelism: str = "unknown"
    test_path: str = ""


def extract_covered_models(covered_file_path: str) -> Set[str]:
    """
    Extract model names that are already covered.
    
    Args:
        covered_file_path: Path to the models.covered file
        
    Returns:
        Set of model names that are already covered
    """
    covered_models = set()
    
    try:
        with open(covered_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('YES'):
                    parts = line.split()
                    if len(parts) >= 3:
                        # Extract model_name from the 3rd column
                        model_name = parts[2]
                        covered_models.add(model_name)
        
        return covered_models
    
    except FileNotFoundError:
        print(f"File not found: {covered_file_path}", file=sys.stderr)
        return set()


def parse_all_models_expected_passing(xml_file_path: str) -> List[ModelInfo]:
    """
    Parse all_models_expected_passing.xml to extract new model information.
    
    Args:
        xml_file_path: Path to the all_models_expected_passing.xml file
        
    Returns:
        List of ModelInfo objects from new tests
    """
    models = []
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Find all testcase elements
        for testcase in root.findall(".//testcase"):
            test_case_name = testcase.get("name", "Unknown")
            
            # Look for properties within each testcase
            properties = testcase.find("properties")
            if properties is not None:
                model = None
                variant = None
                model_name = None
                
                # Extract model, variant, and model_name from properties
                for prop in properties.findall("property"):
                    prop_name = prop.get("name")
                    prop_value = prop.get("value")
                    
                    if prop_name == "model":
                        model = prop_value
                    elif prop_name == "variant":
                        variant = prop_value
                    elif prop_name == "model_name":
                        model_name = prop_value
                
                if model and variant and model_name:
                    models.append(ModelInfo(
                        model=model,
                        variant=variant,
                        model_name=model_name,
                        test_case_name=test_case_name,
                        test_path="tests/runner/test_models.py"
                    ))
        
        return models
    
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print(f"File not found: {xml_file_path}", file=sys.stderr)
        return []


def parse_all_tests_xml(xml_file_path: str) -> List[ModelInfo]:
    """
    Parse all_tests.xml to extract old model information.
    
    Args:
        xml_file_path: Path to the all_tests.xml file
        
    Returns:
        List of ModelInfo objects from old tests
    """
    models = []
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Find all testcase elements
        for testcase in root.findall(".//testcase"):
            test_case_name = testcase.get("name", "Unknown")
            class_name = testcase.get("classname", "")
            
            # Extract test path from classname
            test_path = class_name.replace(".", "/") + ".py" if class_name else ""
            
            # Look for properties within each testcase
            properties = testcase.find("properties")
            if properties is not None:
                model = None
                variant = None
                model_name = None
                parallelism = "unknown"
                
                # Extract information from properties
                for prop in properties.findall("property"):
                    prop_name = prop.get("name")
                    prop_value = prop.get("value")
                    
                    if prop_name == "model":
                        model = prop_value
                    elif prop_name == "variant":
                        variant = prop_value
                    elif prop_name == "model_name":
                        model_name = prop_value
                    elif prop_name == "tags" and prop_value:
                        # Extract parallelism from tags
                        parallelism_match = re.search(r"'parallelism':\s*'([^']+)'", prop_value)
                        if parallelism_match:
                            parallelism = parallelism_match.group(1)
                
                # If we don't have model/variant from properties, try to infer from model_name or test_case_name
                if not model and model_name:
                    # Try to extract model from model_name
                    model = extract_model_from_model_name(model_name)
                
                if not variant:
                    variant = "base"  # Default variant
                
                if model and model_name:
                    models.append(ModelInfo(
                        model=model,
                        variant=variant,
                        model_name=model_name,
                        test_case_name=test_case_name,
                        parallelism=parallelism,
                        test_path=test_path
                    ))
        
        return models
    
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print(f"File not found: {xml_file_path}", file=sys.stderr)
        return []


def extract_model_from_model_name(model_name: str) -> str:
    """
    Extract the core model name from a full model_name.
    
    Args:
        model_name: Full model name like "pytorch_bert_large_nlp_qa_huggingface"
        
    Returns:
        Core model name like "bert"
    """
    # Remove common prefixes and suffixes
    clean_name = model_name.replace("pytorch_", "").replace("_huggingface", "").replace("_torch_hub", "").replace("_custom", "").replace("_timm", "")
    
    # Split by underscores and take the first meaningful part
    parts = clean_name.split("_")
    if len(parts) > 0:
        return parts[0]
    
    return clean_name


def find_model_matches(new_models: List[ModelInfo], old_models: List[ModelInfo]) -> Dict[str, List[ModelInfo]]:
    """
    Find old models that match new models based on model name and variant.
    
    Args:
        new_models: List of new model information
        old_models: List of old model information
        
    Returns:
        Dictionary mapping new model keys to lists of matching old models
    """
    matches = {}
    
    for new_model in new_models:
        key = f"{new_model.model}/{new_model.variant}"
        matches[key] = []
        
        for old_model in old_models:
            # Check for exact model match
            if new_model.model.lower() == old_model.model.lower():
                # Check variant compatibility
                if (new_model.variant.lower() == old_model.variant.lower() or
                    (new_model.variant == "base" and old_model.variant in ["base", "unknown"]) or
                    (old_model.variant == "base" and new_model.variant in ["base", "unknown"])):
                    matches[key].append(old_model)
            
            # Also check for partial model name matches in model_name field
            elif new_model.model.lower() in old_model.model_name.lower() or old_model.model.lower() in new_model.model_name.lower():
                matches[key].append(old_model)
    
    return matches


def generate_report(new_models: List[ModelInfo], old_models: List[ModelInfo], 
                   covered_models: Set[str], matches: Dict[str, List[ModelInfo]]) -> str:
    """
    Generate a comprehensive report of the analysis.
    
    Args:
        new_models: List of new model information
        old_models: List of old model information
        covered_models: Set of already covered model names
        matches: Dictionary of matches between new and old models
        
    Returns:
        Report string
    """
    report = []
    report.append("=" * 80)
    report.append("MODEL COVERAGE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics
    total_new_models = len(new_models)
    total_old_models = len(old_models)
    covered_count = len(covered_models)
    uncovered_new_models = [m for m in new_models if m.model_name not in covered_models]
    
    report.append("SUMMARY STATISTICS:")
    report.append(f"- Total models in all_models_expected_passing.xml: {total_new_models}")
    report.append(f"- Total models in all_tests.xml: {total_old_models}")
    report.append(f"- Already covered models (from models.covered): {covered_count}")
    report.append(f"- Uncovered new models to analyze: {len(uncovered_new_models)}")
    report.append("")
    
    # List uncovered new models
    report.append("UNCOVERED NEW MODELS (98 models to analyze):")
    report.append("-" * 50)
    for i, model in enumerate(uncovered_new_models, 1):
        report.append(f"{i:3d}. {model.model}/{model.variant} -> {model.model_name}")
    report.append("")
    
    # Analysis of matches
    report.append("MATCHING ANALYSIS:")
    report.append("-" * 50)
    
    match_count = 0
    no_match_count = 0
    
    for model in uncovered_new_models:
        key = f"{model.model}/{model.variant}"
        matching_old_models = matches.get(key, [])
        
        if matching_old_models:
            match_count += 1
            report.append(f"\n✓ MATCH FOUND: {key}")
            report.append(f"  New: {model.model_name}")
            report.append(f"  Old matches:")
            for old_model in matching_old_models:
                report.append(f"    - {old_model.model_name} ({old_model.test_path})")
                report.append(f"      Test: {old_model.test_case_name}")
                report.append(f"      Parallelism: {old_model.parallelism}")
        else:
            no_match_count += 1
            report.append(f"\n✗ NO MATCH: {key}")
            report.append(f"  New: {model.model_name}")
    
    report.append("")
    report.append("SUMMARY OF MATCHES:")
    report.append(f"- Models with old test matches: {match_count}")
    report.append(f"- Models without old test matches: {no_match_count}")
    report.append(f"- Coverage percentage: {(match_count / len(uncovered_new_models) * 100):.1f}%")
    
    return "\n".join(report)


def main():
    """Main function to perform the analysis."""
    print("Starting model coverage analysis...")
    
    # File paths
    covered_file = "models.covered"
    new_xml_file = "all_models_expected_passing.xml"
    old_xml_file = "all_tests.xml"
    
    # Step 1: Extract already covered models
    print("Step 1: Extracting already covered models...")
    covered_models = extract_covered_models(covered_file)
    print(f"Found {len(covered_models)} already covered models")
    
    # Step 2: Parse new models XML
    print("\nStep 2: Parsing all_models_expected_passing.xml...")
    new_models = parse_all_models_expected_passing(new_xml_file)
    print(f"Found {len(new_models)} new models")
    
    # Step 3: Filter out covered models
    print("\nStep 3: Filtering out already covered models...")
    uncovered_new_models = [m for m in new_models if m.model_name not in covered_models]
    print(f"Remaining uncovered new models: {len(uncovered_new_models)}")
    
    # Step 4: Parse old models XML
    print("\nStep 4: Parsing all_tests.xml...")
    old_models = parse_all_tests_xml(old_xml_file)
    print(f"Found {len(old_models)} old models")
    
    # Step 5: Find matches
    print("\nStep 5: Finding matches between new and old models...")
    matches = find_model_matches(uncovered_new_models, old_models)
    
    # Step 6: Generate report
    print("\nStep 6: Generating analysis report...")
    report = generate_report(new_models, old_models, covered_models, matches)
    
    # Save report to file
    report_file = "model_coverage_analysis.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis complete! Report saved to {report_file}")
    print("\nFirst few lines of the report:")
    print("-" * 40)
    print("\n".join(report.split("\n")[:20]))


if __name__ == "__main__":
    main() 