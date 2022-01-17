#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./bernoulli.py "$id" "configurations_test.csv" "test"

