#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./MOFA_unsupervised.py "cleaned_data.csv" "configurations_MOFA2.csv" "MOFA_un_whole" "$id"  
