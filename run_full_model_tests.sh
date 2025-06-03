#!/bin/bash

# Script to run all tests from the eval_2 block
# Each test is run with pytest -svv and results are tracked

# Initialize counters
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0

# Array of test strings from eval_2 block
TESTS=(
    "tests/models/bloom/test_bloom.py::test_bloom[data_parallel-full-eval]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-hrnet_w18.ms_aug_in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-ghostnet_100.in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-xception71.tf_in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-mobilenetv1_100.ra4_e3600_r224_in1k]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-dla34.in1k]"
    "tests/models/yolov5/test_yolov5.py::test_yolov5[data_parallel-full-eval]"
    "tests/models/albert/test_albert_question_answering.py::test_albert_question_answering[data_parallel-full-twmkn9/albert-base-v2-squad2-eval]"
    "tests/models/phi/test_phi_1_1p5_2.py::test_phi[data_parallel-full-microsoft/phi-1-eval]"
    "tests/models/clip/test_clip.py::test_clip[data_parallel-full-eval]"
    "tests/models/gpt2/test_gpt2.py::test_gpt2[data_parallel-full-eval]"
    "tests/models/xglm/test_xglm.py::test_xglm[data_parallel-full-eval]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-swin_b]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-densenet201]"
    "tests/models/timm/test_timm_image_classification.py::test_timm_image_classification[data_parallel-full-eval-ese_vovnet19b_dw.ra_in1k]"
    "tests/models/torchvision/test_torchvision_object_detection.py::test_torchvision_object_detection[data_parallel-full-ssd300_vgg16-eval]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-swin_s]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-densenet161]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-swin_v2_t]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-swin_v2_b]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-swin_v2_s]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-densenet121]"
    "tests/models/torchvision/test_torchvision_image_classification.py::test_torchvision_image_classification[data_parallel-full-eval-densenet169]"
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
