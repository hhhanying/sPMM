#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./test.py "cleaned_data.csv" "configurations_MOFA.csv" "MOFA" "$id"  
