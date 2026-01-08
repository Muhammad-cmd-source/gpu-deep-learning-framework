#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Optimized matrix multiplication using shared memory and tiling
__global__ void matmul_kernel_optimized(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    // Tile size
    const int TILE_SIZE = 16;
    
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Naive matrix multiplication (for comparison)
__global__ void matmul_kernel_naive(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Element-wise addition
__global__ void add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// Scalar multiplication
__global__ void scalar_multiply_kernel(const float* A, float scalar, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}

// Matrix transpose
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Host functions
extern "C" {

void cuda_matmul_optimized(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel_optimized<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void cuda_matmul_naive(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    
    matmul_kernel_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void cuda_add(const float* A, const float* B, float* C, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    add_kernel<<<gridSize, blockSize>>>(A, B, C, size);
    cudaDeviceSynchronize();
}

void cuda_scalar_multiply(const float* A, float scalar, float* C, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    scalar_multiply_kernel<<<gridSize, blockSize>>>(A, scalar, C, size);
    cudaDeviceSynchronize();
}

void cuda_transpose(const float* input, float* output, int rows, int cols) {
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + 15) / 16, (rows + 15) / 16);
    
    transpose_kernel<<<gridDim, blockDim>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

} // extern "C"