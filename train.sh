#!/usr/bin/env sh
TOOLS=/mnt/vm_share2/caffe/build/tools

export PYTHONPATH=.

$TOOLS/caffe train \
            --solver=/mnt/vm_share2/caffe/examples/classify_and_regress_ped_project/xc_solver.prototxt

