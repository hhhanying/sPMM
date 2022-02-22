#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./MOFA_supervised.py "cleaned_data.csv" "configurations_MOFA2.csv" "MOFA_whole" "$id"  
