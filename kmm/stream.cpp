#include <random>
#include <kmm/kmm.hpp>
// CUDA kernels
#include "../cuda/stream.cu"

// constants
using real = float;
const int size = 8192;
const real scalar = 3;
const int threads = 256;
const int n_blocks = ceil((1.0 * size) / threads);

template<typename T>
void array_init(std::mt19937 &generator, T* array, const int size) {
    for (int i = 0; i < size; i++) {
        array[i] = std::generate_canonical<real, 16>(generator);
    }
}

int main(void) {
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    auto manager = kmm::build_runtime();

    // allocate and initialize
    auto a = kmm::Array<real>(size);
    auto b = kmm::Array<real>(size);
    auto c = kmm::Array<real>(size);
    manager.submit(kmm::Host(), array_init<real>, generator, write(a), size);
    // copy
    manager.submit(kmm::CudaKernel(n_blocks, threads), copy<real>, a, write(c), size);
    // scale
    manager.submit(kmm::CudaKernel(n_blocks, threads), scale<real>, scalar, write(b), c, size);
    // add
    manager.submit(kmm::CudaKernel(n_blocks, threads), add<real>, a, b, write(c), size);
    // triad
    manager.submit(kmm::CudaKernel(n_blocks, threads), triad<real>, scalar, write(a), b, c, size);

    manager.synchronize();
    return 0;
}