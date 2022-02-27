#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
python ./unsupervised_membership.py "$id" "configurations_membership.csv" "unsupervised_{}.txt" "dataset/unbalanced_{}.txt"

