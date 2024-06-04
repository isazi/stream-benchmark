#include <iostream>
#include <random>
#include <kmm/kmm.hpp>
// CUDA kernels
#include "../cuda/stream.cu"

// constants
using real = float;
const real error = 1.0e-6;
const real scalar = 3;

template<typename T>
void init(T* array, T* control, const int size) {
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    for (int i = 0; i < size; i++) {
        array[i] = std::generate_canonical<real, 16>(generator);
        control[i] = array[i];
    }
}

template<typename T>
void check_final(const T* control_a, const T* a, const T* b, const T* c, const int size) {
    for (int i = 0; i < size; i++) {
        real temp = 0;
        // check a
        temp = (scalar * control_a[i]) + scalar * (control_a[i] + scalar * control_a[i]);
        if ( abs(temp - a[i]) > error ) {
            std::cerr << "Error in a index " << i << std::endl;
            std::cerr << temp << " " << a[i] << std::endl;
            break;
        }
        // check b
        temp = scalar * control_a[i];
        if ( abs(temp - b[i]) > error ) {
            std::cerr << "Error in b index " << i << std::endl;
            std::cerr << temp << " " << b[i] << std::endl;
            break;
        }
        // check c
        temp = control_a[i] + scalar * control_a[i];
        if ( abs(temp - c[i]) > error ) {
            std::cerr << "Error in c index " << i << std::endl;
            std::cerr << temp << " " << c[i] << std::endl;
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }
    const int size = atoi(argv[1]);
    const int threads = 256;
    const int n_blocks = ceil((1.0 * size) / threads);
    auto manager = kmm::build_runtime();

    // allocate and initialize
    auto a = kmm::Array<real>(size);
    auto control_a = kmm::Array<real>(size);
    auto b = kmm::Array<real>(size);
    auto c = kmm::Array<real>(size);
    manager.submit(kmm::Host(), init<real>, write(a), write(control_a), size);
    // copy
    manager.submit(kmm::CudaKernel(n_blocks, threads), copy<real>, a, write(c), size);
    // scale
    manager.submit(kmm::CudaKernel(n_blocks, threads), scale<real>, scalar, write(b), c, size);
    // add
    manager.submit(kmm::CudaKernel(n_blocks, threads), add<real>, a, b, write(c), size);
    // triad
    manager.submit(kmm::CudaKernel(n_blocks, threads), triad<real>, scalar, write(a), b, c, size);
    manager.synchronize();
    // check results
    manager.submit(kmm::Host(), check_final<real>, control_a, a, b, c, size);
    manager.synchronize();
    return 0;
}