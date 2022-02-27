#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./supervised_membership.py "296" "configurations_membership.csv" "supervised_{}.txt" "dataset/dataset_{}.txt" 

