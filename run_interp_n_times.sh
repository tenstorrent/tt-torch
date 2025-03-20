mkdir logs
mkdir logs/interp_rep

start_time=$(date +%s)  # Record the start time



for i in {1..10000}
do
    # pytest -svv tests/torch/test_interpolation.py |& tee logs/interp_rep/run_$i.log
    TT_METAL_LOGGER_LEVEL=INFO TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 TT_METAL_WATCHER_DISABLE_STATUS=1 TT_METAL_WATCHER=10 python3 interp_without_pytest.py |& tee logs/interp_rep/run_$i.log
    # pytest -svv tests/torch/test_interpolation.py -k False-2-224-50 |& tee logs/interp_rep/run_$i.log
    echo "Done with run $i" > logs/interp_rep/status.log

    current_time=$(date +%s)  # Get the current time
    elapsed_time=$((current_time - start_time))  # Calculate elapsed time
    hours=$((elapsed_time / 3600))
    minutes=$(((elapsed_time % 3600) / 60))
    seconds=$((elapsed_time % 60))
    echo "Cumulative time elapsed: ${hours}h ${minutes}m ${seconds}s" >> logs/interp_rep/status.log
done
