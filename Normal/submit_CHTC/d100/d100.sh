#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./normal.py "$id" "configurations_d100.csv" "d100"

