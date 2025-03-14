mkdir logs
mkdir logs/interp_rep
for i in {1..10000}
do
    pytest -svv tests/torch/test_interpolation.py -k False-2-224-50 |& tee logs/interp_rep/run_$i.log
    echo "Done with run $i" > logs/interp_rep/status.log
done