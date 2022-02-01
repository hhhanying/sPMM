#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./MOFA_supervised.py "cleaned_data.csv" "configurations_MOFA.csv" "MOFA" "$id"  
