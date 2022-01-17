#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./bernoulli.py "$id" "configurations_d100.csv" "d100"

