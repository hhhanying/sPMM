#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./test_un.py "cleaned_data.csv" "configurations_MOFA.csv" "test_un" "$id"  "res.txt"
