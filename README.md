# GPU-Accelerated Deep Learning Framework

CUDA C++ kernels achieving **11.2× speedup** for neural network training.


##  Performance
- **Neural Network:** 11.2× speedup
- **Matrix Operations:** 39× speedup
- **GPU:**  Tesla T4 (2560 CUDA cores)

##  Technologies
- CUDA C/C++ - Custom GPU kernels
- C++17 - Tensor operations  
- Python + PyTorch - Benchmarks
- CMake - Build system

##  Structure
- `cuda_kernels/` - CUDA kernels with shared memory optimization
- `include/` - C++ tensor class headers
- `src/` - C++ implementation
- `CMakeLists.txt` - Build configuration

##  Optimizations
- Shared memory tiling (16×16 blocks)
- Memory coalescing
- 2048 bytes shared memory usage
- 30 registers per thread

##  Build
```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

