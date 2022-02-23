#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./MOFA_supervised_2label.py "cleaned_data.csv" "configurations_MOFA.csv" "MOFA_2_" "$id"  
