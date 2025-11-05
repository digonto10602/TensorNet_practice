#!/bin/bash 

/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3/bin/nvcc tensornet_Sampling_MPS.cu \
    -std=c++14 \
    -I/usr/local/cutensor/include \
    -I/usr/local/cuquantum/include \
    -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/math_libs/include \
    -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3/include \
    -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/math_libs/lib64 \
    -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3/lib64 \
    -Xlinker -rpath -Xlinker /opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3/lib64 \
    -Xlinker -rpath-link -Xlinker /opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3/lib64 \
    -Xlinker -rpath -Xlinker /opt/nvidia/hpc_sdk/Linux_x86_64/24.3/math_libs/lib64 \
    -Xlinker -rpath-link -Xlinker /opt/nvidia/hpc_sdk/Linux_x86_64/24.3/math_libs/lib64 \
    -L/usr/local/cutensor/lib \
    -L/usr/local/cuquantum/lib \
    -Xlinker -rpath -Xlinker /usr/local/cuquantum/lib \
    -Xlinker -rpath -Xlinker /usr/local/cutensor/lib \
    -lcutensornet -lcutensor \
    -lcublas -lcusolver -lcudart \
    -o test

    
