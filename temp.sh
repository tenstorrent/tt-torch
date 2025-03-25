#!/bin/bash

# Model Test Runner Script
# Runs pytest for each model and reports pass/fail status, cleaning cache between runs

# Define colors for better output readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Create a results directory
RESULTS_DIR="./model_test_results"
mkdir -p $RESULTS_DIR

# Get current timestamp for the summary file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$RESULTS_DIR/summary_$TIMESTAMP.txt"

# Initialize summary file
echo "Model Test Summary ($(date))" > $SUMMARY_FILE
echo "====================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Define model tests to run (from the GitHub workflow file)
MODEL_TESTS=(
    # run1 tests
    "stable_diffusion" "Qwen" "MobileNetV2" "clip" "flan_t5" "mlpmixer" "resnet" "vilt"
    "albert" "codegen" "glpn_kitti" "mnist" "resnet50" "RMBG" "unet_carvana" "mgp-str-base"
    "musicgen_small" "segformer" "torchvision" "yolos"
    # run2 tests
    "t5" "whisper" "autoencoder_conv" "deit" "gpt2" "mobilenet_ssd" "roberta" "timm" "xglm"
    "autoencoder_linear" "detr" "beit" "distilbert" "hand_landmark" "openpose" "segment_anything"
    "unet" "yolov3" "bert" "dpr" "hardnet" "opt" "speecht5_tts" "unet_brain" "yolov5" "bloom"
    "falcon" "llama" "perceiver_io" "squeeze_bert" "gpt_neo"
)

# Function to clean the cache
clean_cache() {
    echo -e "${YELLOW}Cleaning cache at /localdev/$USER/cache...${NC}"
    rm -rf /localdev/$USER/cache/* 2>/dev/null || true
    echo "Cache cleaned."
}

# Counter for passed/failed tests
PASSED=0
FAILED=0

# Run tests for each model
total_tests=${#MODEL_TESTS[@]}
current=0

for testname in "${MODEL_TESTS[@]}"; do
    current=$((current + 1))
    echo -e "\n${YELLOW}[$current/$total_tests] Running test for model: ${testname}${NC}"
    echo "[$current/$total_tests] Test: $testname" >> $SUMMARY_FILE

    # Run the pytest command and capture its output and exit status
    TEST_OUTPUT_FILE="$RESULTS_DIR/${testname}_output.log"

    # Run the test
    pytest -svv tests/models/$testname --nightly > $TEST_OUTPUT_FILE 2>&1
    TEST_STATUS=$?

    # Check if test passed or failed
    if [ $TEST_STATUS -eq 0 ]; then
        echo -e "${GREEN}✓ Test PASSED: $testname${NC}"
        echo "  Status: PASSED" >> $SUMMARY_FILE
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ Test FAILED: $testname${NC}"
        echo "  Status: FAILED" >> $SUMMARY_FILE
        FAILED=$((FAILED + 1))
    fi

    # Clean the cache before next test
    clean_cache

    echo "" >> $SUMMARY_FILE
done

# Write summary statistics
echo "====================================" >> $SUMMARY_FILE
echo "SUMMARY:" >> $SUMMARY_FILE
echo "Total tests: $total_tests" >> $SUMMARY_FILE
echo "Passed: $PASSED" >> $SUMMARY_FILE
echo "Failed: $FAILED" >> $SUMMARY_FILE
echo "Success rate: $(( (PASSED * 100) / total_tests ))%" >> $SUMMARY_FILE

# Print final summary to console
echo -e "\n${YELLOW}====================================${NC}"
echo -e "${YELLOW}TEST SUMMARY:${NC}"
echo -e "Total tests: $total_tests"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo -e "Success rate: $(( (PASSED * 100) / total_tests ))%"
echo -e "${YELLOW}Summary saved to: $SUMMARY_FILE${NC}"
echo -e "Detailed logs for each test are in: $RESULTS_DIR"
