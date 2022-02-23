#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./MOFA_unsupervised_2label.py "cleaned_data.csv" "configurations_MOFA.csv" "MOFA_un_2_" "$id"  
