#!/bin/bash

counter=0
while [ $counter -lt $3 ]
do
    python3 davis_test.py test-dev configs/test_config.py --output $1 --cache $2 --gpu_num $3 --gpu $counter &
    ((counter++))
done
