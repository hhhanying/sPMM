#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./normal.py "$id" "configurations_test.csv" "test"

