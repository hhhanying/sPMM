#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./MOFA_unsupervised_2label.py "cleaned_data.csv" "configurations_MOFA2.csv" "MOFA_un_2_whole_" "$id"  
