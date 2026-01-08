#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ReLU activation
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU derivative
__global__ void relu_derivative_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

// Sigmoid activation
__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Sigmoid derivative
__global__ void sigmoid_derivative_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sig = 1.0f / (1.0f + expf(-input[idx]));
        output[idx] = sig * (1.0f - sig);
    }
}

// Softmax activation (numerically stable version)
__global__ void softmax_kernel(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        const float* input_batch = input + batch_idx * num_classes;
        float* output_batch = output + batch_idx * num_classes;
        
        // Find max for numerical stability
        float max_val = input_batch[0];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, input_batch[i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output_batch[i] = expf(input_batch[i] - max_val);
            sum += output_batch[i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output_batch[i] /= sum;
        }
    }
}

// Host functions
extern "C" {

void cuda_relu(const float* input, float* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    relu_kernel<<<gridSize, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

void cuda_relu_derivative(const float* input, float* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    relu_derivative_kernel<<<gridSize, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

void cuda_sigmoid(const float* input, float* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    sigmoid_kernel<<<gridSize, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

void cuda_sigmoid_derivative(const float* input, float* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    sigmoid_derivative_kernel<<<gridSize, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

void cuda_softmax(const float* input, float* output, int batch_size, int num_classes) {
    softmax_kernel<<<batch_size, 1>>>(input, output, batch_size, num_classes);
    cudaDeviceSynchronize();
}

} // extern "C"