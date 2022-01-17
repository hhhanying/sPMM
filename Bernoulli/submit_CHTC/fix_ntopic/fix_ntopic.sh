#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./bernoulli.py "$id" "configurations_fix_ntopic.csv" "fix_ntopic"

