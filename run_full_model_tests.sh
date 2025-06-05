#!/bin/bash

# Script to run all tests from the eval_2 block
# Each test is run with pytest -svv and results are tracked

# Initialize counters
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0

# Array of test strings from eval_2 block
TESTS=(
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg19]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg19_bn]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg16]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg16_bn]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg13_bn]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg11_bn]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg11]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vgg13]"
    "tests/models/dpr/test_dpr.py::test_dpr[data_parallel-full-eval]"
    "tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[data_parallel-full-albert/albert-xlarge-v2-eval]"
    "tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[data_parallel-full-albert/albert-large-v2-eval]"
    "tests/models/albert/test_albert_masked_lm.py::test_albert_masked_lm[data_parallel-full-albert/albert-base-v2-eval]"
    "tests/models/albert/test_albert_sequence_classification.py::test_albert_sequence_classification[data_parallel-full-textattack/albert-base-v2-imdb-eval]"
    "tests/models/albert/test_albert_token_classification.py::test_albert_token_classification[data_parallel-full-albert/albert-base-v2-eval]"
    "tests/models/roberta/test_roberta.py::test_roberta[data_parallel-full-eval]"
    "tests/models/hardnet/test_hardnet.py::test_hardnet[data_parallel-full-eval]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite0.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite1.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite2.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite3.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-tf_efficientnet_lite4.in1k]"
    "tests/models/segformer/test_segformer.py::test_segformer[data_parallel-full-eval]"
    "tests/models/torchvision/test_torchvision_object_detection.py::test_torchvision_object_detection[data_parallel-full-ssdlite320_mobilenet_v3_large-eval]"
    "tests/models/beit/test_beit_image_classification.py::test_beit_image_classification[data_parallel-full-microsoft/beit-base-patch16-224-eval]"
    "tests/models/beit/test_beit_image_classification.py::test_beit_image_classification[data_parallel-full-microsoft/beit-large-patch16-224-eval]"
    "tests/models/deit/test_deit.py::test_deit[data_parallel-full-facebook/deit-base-patch16-224-eval]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-mixer_b16_224.goog_in21k]"
    "tests/models/mgp-str-base/test_mgp_str_base.py::test_mgp_str_base[data_parallel-full-eval]"
    "tests/models/detr/test_detr.py::test_detr[data_parallel-full-eval]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-vit_h_14]"
    "tests/models/yolos/test_yolos.py::test_yolos[data_parallel-full-eval]"
    "tests/models/whisper/test_whisper.py::test_whisper[data_parallel-full-eval]"
    "tests/models/bert/test_bert_turkish.py::test_bert_turkish[data_parallel-full-eval]"
)

# Function to log results
log_result() {
    local test_name="$1"
    local status="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $status: $test_name" | tee -a test_results.log
}

# Function to run a single test
run_test() {
    local test_string="$1"
    local test_num="$2"
    local total_tests="$3"

    echo ""
    echo "=========================================="
    echo "Running test $test_num of $total_tests"
    echo "Test: $test_string"
    echo "=========================================="

    # Run the test and capture exit code
    pytest -svv "$test_string"
    local exit_code=$?

    # Update counters based on exit code
    if [ $exit_code -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        log_result "$test_string" "SUCCESS"
        echo "✅ Test PASSED"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        log_result "$test_string" "FAILED"
        echo "❌ Test FAILED (exit code: $exit_code)"
    fi

    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    echo "Current status: $SUCCESS_COUNT passed, $FAIL_COUNT failed out of $TOTAL_COUNT total"
}

# Main execution
echo "Starting eval_2 test suite execution..."
echo "Total tests to run: ${#TESTS[@]}"
echo "Results will be logged to test_results.log"
echo ""

# Clear previous log
> test_results.log

# Record start time
START_TIME=$(date)
START_TIMESTAMP=$(date +%s)

echo "Started at: $START_TIME" | tee -a test_results.log

# Run each test
for i in "${!TESTS[@]}"; do
    test_index=$((i + 1))
    run_test "${TESTS[$i]}" "$test_index" "${#TESTS[@]}"
done

# Calculate execution time
END_TIME=$(date)
END_TIMESTAMP=$(date +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
DURATION_FORMATTED=$(printf '%dh %dm %ds' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))

# Final summary
echo ""
echo "=========================================="
echo "               FINAL SUMMARY"
echo "=========================================="
echo "Started at:     $START_TIME"
echo "Ended at:       $END_TIME"
echo "Duration:       $DURATION_FORMATTED"
echo ""
echo "Total tests:    $TOTAL_COUNT"
echo "Passed:         $SUCCESS_COUNT"
echo "Failed:         $FAIL_COUNT"
echo "Success rate:   $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%"
echo "=========================================="

# Log final summary
{
    echo ""
    echo "=========================================="
    echo "FINAL SUMMARY - $END_TIME"
    echo "=========================================="
    echo "Duration: $DURATION_FORMATTED"
    echo "Total: $TOTAL_COUNT | Passed: $SUCCESS_COUNT | Failed: $FAIL_COUNT"
    echo "Success rate: $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%"
    echo "=========================================="
} >> test_results.log

# Exit with failure code if any tests failed
if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "⚠️  Some tests failed. Check test_results.log for details."
    exit 1
else
    echo ""
    echo "🎉 All tests passed!"
    exit 0
fi
