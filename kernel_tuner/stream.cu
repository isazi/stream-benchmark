#include <cuda_fp16.h>
using half = __half;

template<typename T>
__global__ void copy(const T* a, T* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        c[i] = a[i];
    }
}

template<typename T>
__global__ void scale(const T scalar, T* b, const T* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        b[i] = scalar * c[i];
    }
}

template<typename T>
__global__ void add(const T* a, const T* b, T* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
__global__ void triad(const T scalar, T* a, const T* b, const T* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        a[i] = b[i] + scalar * c[i];
    }
}

template<typename T>
__global__ void stream(const T scalar, T* a, T* b, T* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        c[i] = a[i];
        b[i] = scalar * c[i];
        c[i] = a[i] + b[i];
        a[i] = b[i] + scalar * c[i];
    }
}
