#!/bin/bash 


mpic++ -m64 -std=c++17 -fconcepts -fPIC -I. \
    -I'/home/digonto/Codes/Github_repos/ITensor'  \
    -I/usr/include/hdf5/serial -DITENSOR_USE_HDF5 -DITENSOR_USE_OMP \
    -fopenmp  -O3 -DNDEBUG -Wall -Wno-unknown-pragmas \
    practice_itensor_MPI.cpp \
    -o test \
    -L'/home/digonto/Codes/Github_repos/ITensor/lib' \
    -litensor -lpthread -L/usr/lib -lblas -llapack  \
    -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 \
    -lhdf5_hl -fopenmp