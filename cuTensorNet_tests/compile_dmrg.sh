#!/bin/bash

nvcc dmrg.cu \
    -std=c++14 \
    -I${CUQUANTUM_ROOT}/include/ \
    -L${CUQUANTUM_ROOT}/lib/ \
    -lcutensornet -lcutensor \
    -lcublas -lcusolver -lcudart \
    -I/home/digonto/anaconda3/envs/cuquantum-env/include -L/home/digonto/anaconda3/envs/cuquantum-env/lib  -lmpi \
    -o test

    
