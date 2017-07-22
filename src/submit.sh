#!/bin/bash

/opt/spark/bin/spark-submit \
    --class preprocess.RunLSA \
    --master local[4] \
    --driver-memory 8g \
    --jars /home/hungdv/workspace/Babe_challenge/target/scala-2.11/Babe_challenge-assembly-0.1.0-SNAPSHOT-deps.jar \
    --executor-memory 3g \
    --executor-cores 1 \
    --num-executors 2 \
    /home/hungdv/workspace/Babe_challenge/target/scala-2.11/babe_challenge_2.11-0.1.0-SNAPSHOT.jar >> test-local.log 2>&1
