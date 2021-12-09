#!/bin/bash
export HOME=$_CONDOR_SCRATCH_DIR
echo $id
python ./CV.py "$id"

