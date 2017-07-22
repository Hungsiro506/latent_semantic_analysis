
#!/bin/bash

/opt/spark/bin/spark-submit \
    --class preprocess.RunLSA \
    --master local[4] \
    --driver-memory 6g \
    --jars /home/hungvd8/projects/babe_challenge/Babe_challenge-assembly-0.1.0-SNAPSHOT-deps.jar \
    --executor-memory 3g \
    --executor-cores 2 \
    --num-executors 2 \
    /home/hungvd8/projects/babe_challenge/babe_challenge_2.11-0.1.0-SNAPSHOT.jar >> test-local.log 2>&1
