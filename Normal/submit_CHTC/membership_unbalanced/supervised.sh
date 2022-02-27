#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./supervised_membership.py "$id" "configurations_membership.csv" "supervised_{}.txt" "dataset/unbalanced_{}.txt" 

