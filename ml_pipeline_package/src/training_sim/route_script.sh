#!/bin/bash
for i in $(seq 1 31)
do
    echo "========================================================\n"
    echo "This is the $i th run\n"
    echo "========================================================\n"
    python3 ~/FINAL_YEAR_PROJECT/ml_ros_package/ml_pipeline_package/src/training_sim/training_route.py
    killall -9 gzserver gzclient
done
