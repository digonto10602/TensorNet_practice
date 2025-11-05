#!/bin/bash

nvcc tensornet_Sampling_MPS.cu \
    -std=c++14 \
    -I${CUQUANTUM_ROOT}/include/ \
    -L${CUQUANTUM_ROOT}/lib/ \
    -lcutensornet -lcutensor \
    -lcublas -lcusolver -lcudart \
    -o test

    
