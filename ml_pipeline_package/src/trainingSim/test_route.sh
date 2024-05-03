#!/bin/bash
for i in $(seq 1 115)
do
    echo "========================================================\n"
    echo "This is the $i th run\n"
    echo "========================================================\n"
    python3 /home/danielhixson/socNavProject/ml_pipeline_package/src/training_route.py
    killall -9 gzserver gzclient
done
