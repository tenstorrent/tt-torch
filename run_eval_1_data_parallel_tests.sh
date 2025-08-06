#!/bin/bash

# Script to run all tests from eval_1_data_parallel block and report failures
# Based on the GitHub workflow configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running eval_1_data_parallel tests...${NC}"

# Define the tests from the eval_1_data_parallel block
TESTS=(
    # "tests/models/bert/test_bert_turkish.py::test_bert_turkish[data_parallel-full-eval]"
    # "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg19]"
    # "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg19_bn]"
    # "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg16]"
    # "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg16_bn]"
    # "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg13_bn]"
    # "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg11_bn]"
    # "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg11]"
    # "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg13]"
    # "tests/models/dpr/test_dpr.py::test_dpr[data_parallel-full-eval]"
    # "tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[data_parallel-full-xlarge_v2-eval]"
    # "tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[data_parallel-full-large_v2-eval]"
    # "tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[data_parallel-full-base_v2-eval]"
    # "tests/models/albert/test_albert_sequence_classification.py::test_albert_sequence_classification[data_parallel-full-textattack/albert-base-v2-imdb-eval]"
    # "tests/models/albert/test_albert_token_classification.py::test_albert_token_classification[data_parallel-full-albert/albert-base-v2-eval]"
    # "tests/models/roberta/test_roberta.py::test_roberta[data_parallel-full-eval]"
    "tests/models/hardnet/test_hardnet.py::test_hardnet[data_parallel-full-eval]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite0.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite1.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite2.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite3.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite4.in1k]"
    "tests/models/segformer/test_segformer.py::test_segformer[data_parallel-full-eval]"
    "tests/models/beit/test_beit_image_classification.py::test_beit_image_classification[data_parallel-full-base-eval]"
    "tests/models/beit/test_beit_image_classification.py::test_beit_image_classification[data_parallel-full-large-eval]"
    "tests/models/deit/test_deit.py::test_deit[data_parallel-full-eval]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-mixer_b16_224.goog_in21k]"
    "tests/models/detr/test_detr.py::test_detr[data_parallel-full-eval]"
    "tests/models/yolos/test_yolos.py::test_yolos[data_parallel-full-eval]"
    "tests/models/whisper/test_whisper.py::test_whisper[data_parallel-full-eval]"
    # Additional tests
    "tests/models/mnist/test_mnist.py::test_mnist_train[data_parallel-full-eval]"
    "tests/models/resnet50/test_resnet50.py::test_resnet[data_parallel-full-eval]"
    "tests/models/resnet/test_resnet.py::test_resnet[data_parallel-full-eval]"
)

# Initialize counters
TOTAL_TESTS=${#TESTS[@]}
PASSED_TESTS=0
FAILED_TESTS=()

echo -e "${YELLOW}Total tests to run: $TOTAL_TESTS${NC}"
echo ""

# Activate environment if it exists
if [ -f "env/activate" ]; then
    echo -e "${YELLOW}Activating environment...${NC}"
    source env/activate
fi

# Create a temporary file to store pytest results
TEMP_RESULTS=$(mktemp)

# Run each test individually to track failures
for i in "${!TESTS[@]}"; do
    test="${TESTS[$i]}"
    test_num=$((i + 1))
    
    echo -e "${YELLOW}[$test_num/$TOTAL_TESTS] Running: $test${NC}"
    
    # Run pytest with verbose output and capture exit code
    if pytest "$test" > "$TEMP_RESULTS" 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_TESTS+=("$test")
        echo "  Error output:"
        tail -n 5 "$TEMP_RESULTS" | sed 's/^/    /'
    fi
    echo ""
done

# Clean up temporary file
rm -f "$TEMP_RESULTS"

# Print summary
echo -e "${YELLOW}================================${NC}"
echo -e "${YELLOW}TEST SUMMARY${NC}"
echo -e "${YELLOW}================================${NC}"
echo -e "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: ${#FAILED_TESTS[@]}${NC}"
echo ""

# Print failed tests
if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "${RED}FAILED TESTS:${NC}"
    echo -e "${RED}==============${NC}"
    for failed_test in "${FAILED_TESTS[@]}"; do
        echo -e "${RED}• $failed_test${NC}"
    done
    echo ""
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
