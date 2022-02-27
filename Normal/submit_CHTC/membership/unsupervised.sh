#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./unsupervised_membership.py "296" "configurations_membership.csv" "unsupervised_{}.txt" "dataset/dataset_{}.txt"

