#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <complex>
#include <random>
#include <functional>
#include <vector>
#include <iostream>
#include <unordered_map> 

#include <cuda_runtime.h>
#include <cutensornet.h>

#define HANDLE_ERROR(x) \
    do {   \
            const auto err = x; \
            if(err != CUTENSORNET_STATUS_SUCCESS  ) \
            { \
                printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
                fflush(stdout); \
                exit(EXIT_FAILURE); \
            } \
        } while(0) 

#define HANDLE_CUDA_ERROR(x) \
    do { \
        const auto err = x; \
        if(err != cudaSuccess) \
        { \
            printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
            fflush(stdout); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define DEV_ATTR(ENUMCONST, DID) \
    ({  int v; \
        HANDLE_CUDA_ERROR(cudaDeviceGetAttribute(&v, ENUMCONST, DID)); \
        v; })

struct GPUTimer 
{
    GPUTimer(cudaStream_t stream) : stream_(stream)
    {
        HANDLE_CUDA_ERROR(cudaEventCreate(&start_)); 
        HANDLE_CUDA_ERROR(cudaEventCreate(&stop_)); 
    }

    ~GPUTimer()
    {
        HANDLE_CUDA_ERROR(cudaEventDestroy(start_)); 
        HANDLE_CUDA_ERROR(cudaEventDestroy(stop_));
    }

    void start() { HANDLE_CUDA_ERROR(cudaEventRecord(start_, stream_)); }

    float seconds()
    {
        HANDLE_CUDA_ERROR(cudaEventRecord(stop_, stream_)) ; 
        HANDLE_CUDA_ERROR(cudaEventSynchronize(stop_)); 
        float time; 
        HANDLE_CUDA_ERROR(cudaEventElapsedTime(&time, start_, stop_)); 
        return time * 1e-3; 
    }

private: 
    cudaEvent_t start_, stop_; 
    cudaStream_t stream_; 
};

void MPS_sample()
{
    static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on 64 bit arch");

    bool verbose = true; 

    const size_t cuTensornetVersion = cutensornetGetVersion(); 
    if (verbose) printf("cuTensorNet version: %ld\n", cuTensornetVersion); 

    int numDevices{0}; 
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices)); 
    const int deviceId = 0; 
    cudaDeviceProp prop; 
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    if(verbose)
    {
        printf("=======device info=========\n");
        printf("GPU-local-id: %d\n", deviceId);
        printf("GPU-name:%s\n", prop.name); 
        printf("GPU-clock:%d\n", DEV_ATTR(cudaDevAttrClockRate, deviceId));
        printf("GPU-memoryClock:%d\n", DEV_ATTR(cudaDevAttrMemoryClockRate, deviceId));
        printf("GPU-nSM:%d\n", prop.multiProcessorCount);
        printf("GPU-major:%d\n", prop.major);
        printf("GPU-minor:%d\n", prop.minor);

        int memClockRate = DEV_ATTR(cudaDevAttrMemoryClockRate, deviceId);
        int memBusWidth = DEV_ATTR(cudaDevAttrGlobalMemoryBusWidth, deviceId); 
        int bandwidth_GBs = 2.0 * memClockRate * (memBusWidth / 8.0)/1.0e6; 
        printf("Peak Bandwidth:%d GBps\n", bandwidth_GBs);
        printf("========================\n");
    }

    typedef float floatType; 
    cudaDataType_t typeData =   CUDA_R_32F; 
    cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F; 

    if(verbose) printf("Included headers and defined data types\n"); 

    //Input Tensor
    //R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}

    constexpr int32_t numInputs = 4;

    // Create vectors of tensor modes
    std::vector<std::vector<int32_t>> tensorModes{ // for input tensors & output tensor
        // input tensors
        {'a', 'b', 'c', 'd', 'e', 'f'}, // tensor A
        {'b', 'g', 'h', 'e', 'i', 'j'}, // tensor B
        {'m', 'a', 'g', 'f', 'i', 'k'}, // tensor C
        {'l', 'c', 'h', 'd', 'j', 'm'}, // tensor D
        // output tensor
        {'k', 'l'}, // tensor R
    };

    int64_t sameExtent = 16; 
    std::unordered_map<int32_t, int64_t> extent; 
    for(auto &vec : tensorModes)
    {
        for(auto &mode : vec)
        {
            extent[mode] = sameExtent; 
            std::cout<<"mode:"<<static_cast<char>(mode)<<"\t extent:"<<extent[mode]<<std::endl; 
        }
    }

    std::vector<size_t> tensorElements(numInputs + 1); 
    std::vector<size_t> tensorSizes(numInputs + 1); 
    size_t totalSize = 0; 

    for(int32_t t = 0; t < numInputs + 1; ++t)
    {
        size_t numElements = 1; 
        for(auto &mode : tensorModes[t]) numElements *= extent[mode]; 
        tensorElements[t] = numElements; 
        tensorSizes[t] = sizeof(floatType) * numElements; 
        totalSize += tensorSizes[t]; 
    }

    if(verbose) printf("Total GPU Memory used for tensor storage: %.2f GiB\n", (totalSize)/ 1024. / 1024. / 1024);

    void* tensorData_d[numInputs + 1]; 

    for(int32_t t = 0; t < numInputs + 1; ++t)
    {
        HANDLE_CUDA_ERROR(cudaMalloc((void**) &tensorData_d[t], tensorSizes[t])); 
    }

    floatType* tensorData_h[numInputs + 1]; 
    for(int32_t t = 0; t < numInputs + 1; ++t)
    {
        tensorData_h[t] = (floatType*) malloc(tensorSizes[t]);
        if(tensorData_h[t] == NULL)
        {
            printf("Error: Host memory allocation failed\n");
            //return -1; 
        }
    }

    //initialize data

    memset(tensorData_h[numInputs], 0, tensorSizes[numInputs]); 

    for(int32_t t = 0; t < numInputs + 1; ++t)
    {
        for(size_t e = 0; e < tensorElements[t]; ++e)
        {
            tensorData_h[t][e] = ((floatType) rand()) / RAND_MAX; 
        }
    }

    for(int32_t t = 0; t < numInputs + 1; ++t)
    {
        HANDLE_CUDA_ERROR(cudaMemcpy(tensorData_d[t], tensorData_h[t], tensorSizes[t], cudaMemcpyHostToDevice));
    }

    //cuTensorNet 

    cudaStream_t stream; 
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream)); 

    cutensornetHandle_t handle; 
    HANDLE_ERROR(cutensornetCreate(&handle)); 

    if (verbose) printf("Allocated GPU memory for data, initialized data, and created library handle\n");

    cutensorNetworkDescriptor_t networkDesc; 
    HANDLE_ERROR(cutensornetCreateNetwork(handle, &networkDesc)); 

    int64_t tensorIDs[numInputs]; 

    for(int32_t t=0; t < numInputs; ++t)
    {
        HANDLE_ERROR(cutensornetNetworkAppendTensor(handle, 
                                                    networkDesc, 
                                                    tensorModes[t].size(),
                                                    tensorExtents[t].data(),
                                                    tensorModes[t].data(),
                                                    NULL, 
                                                    typeData, 
                                                    &tensorIDs[t]));
                    
    }

    HANDLE_ERROR(cutensornetNetworkSetOutputTensor( handle, 
                                                    networkDesc, 
                                                    CUTENSORNET_NETWORK_COMPUTE_TYPE, 
                                                    &typeCompute,
                                                    sizeof(typeCompute)));
    
    if (verbose) printf("Initialized the cuTensorNet library and created a tensor network descriptor\n");
    
    //optimal contraction and slicing

    size_t 






}

int main(int argc, char* argv[])
{
    MPS_sample();
    
    return 0; 
}