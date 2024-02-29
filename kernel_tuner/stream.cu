#include <cuda_fp16.h>
using half = __half;

template<typename T>
__global__ void copy(const T* a, T* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        c[i] = a[i];
    }
}

__global__ void scale(const TYPE scalar, TYPE* b, const TYPE* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        b[i] = scalar * c[i];
    }
}

__global__ void add(const TYPE* a, const TYPE* b, TYPE* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        c[i] = a[i] + b[i];
    }
}

__global__ void triad(const TYPE scalar, TYPE* a, const TYPE* b, const TYPE* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        a[i] = b[i] + scalar * c[i];
    }
}

__global__ void stream(const TYPE scalar, TYPE* a, TYPE* b, TYPE* c, const int size) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if ( i < size ) {
        c[i] = a[i];
        b[i] = scalar * c[i];
        c[i] = a[i] + b[i];
        a[i] = b[i] + scalar * c[i];
    }
}
