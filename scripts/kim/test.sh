#!/bin/bash

export THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer_including=cudnn,warn_float64=warn,lib.cnmem=0.9'

# export THEANO_FLAGS=device=cpu,floatX=float32

python -u ./gen.py > log_test.txt 2>&1 &




