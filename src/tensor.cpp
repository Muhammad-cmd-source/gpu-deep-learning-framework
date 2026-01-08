#include "../include/tensor.h"
#include <cstring>
#include <cmath>
#include <random>
#include <stdexcept>
#include <numeric>

// External CUDA functions
extern "C" {
    void cuda_matmul_optimized(const float* A, const float* B, float* C, int M, int N, int K);
    void cuda_add(const float* A, const float* B, float* C, int size);
    void cuda_scalar_multiply(const float* A, float scalar, float* C, int size);
    void cuda_transpose(const float* input, float* output, int rows, int cols);
    void cuda_relu(const float* input, float* output, int size);
    void cuda_relu_derivative(const float* input, float* output, int size);
    void cuda_sigmoid(const float* input, float* output, int size);
    void cuda_sigmoid_derivative(const float* input, float* output, int size);
    void cuda_softmax(const float* input, float* output, int batch_size, int num_classes);
}

Tensor::Tensor(const std::vector<int>& shape, Device device)
    : shape_(shape), device_(device), h_data_(nullptr), d_data_(nullptr) {
    total_size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    allocate_memory();
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data, Device device)
    : shape_(shape), device_(device), h_data_(nullptr), d_data_(nullptr) {
    total_size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (data.size() != total_size_) {
        throw std::runtime_error("Data size mismatch");
    }
    allocate_memory();
    std::memcpy(h_data_, data.data(), total_size_ * sizeof(float));
    if (device_ == Device::CUDA) {
        sync_to_device();
    }
}

Tensor::~Tensor() {
    free_memory();
}

void Tensor::allocate_memory() {
    h_data_ = new float[total_size_];
    std::memset(h_data_, 0, total_size_ * sizeof(float));
    
    if (device_ == Device::CUDA) {
        cudaMalloc(&d_data_, total_size_ * sizeof(float));
        cudaMemset(d_data_, 0, total_size_ * sizeof(float));
    }
}

void Tensor::free_memory() {
    if (h_data_) delete[] h_data_;
    if (d_data_) cudaFree(d_data_);
}

void Tensor::sync_to_host() {
    if (device_ == Device::CUDA && d_data_) {
        cudaMemcpy(h_data_, d_data_, total_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void Tensor::sync_to_device() {
    if (device_ == Device::CUDA && d_data_) {
        cudaMemcpy(d_data_, h_data_, total_size_ * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void Tensor::to(Device device) {
    if (device_ == device) return;
    
    if (device == Device::CUDA) {
        cudaMalloc(&d_data_, total_size_ * sizeof(float));
        cudaMemcpy(d_data_, h_data_, total_size_ * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        sync_to_host();
        cudaFree(d_data_);
        d_data_ = nullptr;
    }
    device_ = device;
}

void Tensor::fill(float value) {
    for (int i = 0; i < total_size_; i++) {
        h_data_[i] = value;
    }
    if (device_ == Device::CUDA) {
        sync_to_device();
    }
}

void Tensor::randomize(float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    
    for (int i = 0; i < total_size_; i++) {
        h_data_[i] = dist(gen);
    }
    if (device_ == Device::CUDA) {
        sync_to_device();
    }
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    if (a.shape_.size() != 2 || b.shape_.size() != 2) {
        throw std::runtime_error("Matmul requires 2D tensors");
    }
    if (a.shape_[1] != b.shape_[0]) {
        throw std::runtime_error("Incompatible dimensions for matmul");
    }
    
    int M = a.shape_[0];
    int K = a.shape_[1];
    int N = b.shape_[1];
    
    Tensor result({M, N}, a.device_);
    
    if (a.device_ == Device::CUDA) {
        cuda_matmul_optimized(a.d_data_, b.d_data_, result.d_data_, M, N, K);
    } else {
        // CPU fallback
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a.h_data_[i * K + k] * b.h_data_[k * N + j];
                }
                result.h_data_[i * N + j] = sum;
            }
        }
    }
    
    return result;
}

Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    if (a.total_size_ != b.total_size_) {
        throw std::runtime_error("Tensors must have same size for addition");
    }
    
    Tensor result(a.shape_, a.device_);
    
    if (a.device_ == Device::CUDA) {
        cuda_add(a.d_data_, b.d_data_, result.d_data_, a.total_size_);
    } else {
        for (int i = 0; i < a.total_size_; i++) {
            result.h_data_[i] = a.h_data_[i] + b.h_data_[i];
        }
    }
    
    return result;
}

Tensor Tensor::multiply(const Tensor& a, float scalar) {
    Tensor result(a.shape_, a.device_);
    
    if (a.device_ == Device::CUDA) {
        cuda_scalar_multiply(a.d_data_, scalar, result.d_data_, a.total_size_);
    } else {
        for (int i = 0; i < a.total_size_; i++) {
            result.h_data_[i] = a.h_data_[i] * scalar;
        }
    }
    
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Transpose requires 2D tensor");
    }
    
    Tensor result({shape_[1], shape_[0]}, device_);
    
    if (device_ == Device::CUDA) {
        cuda_transpose(d_data_, result.d_data_, shape_[0], shape_[1]);
    } else {
        for (int i = 0; i < shape_[0]; i++) {
            for (int j = 0; j < shape_[1]; j++) {
                result.h_data_[j * shape_[0] + i] = h_data_[i * shape_[1] + j];
            }
        }
    }
    
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_, device_);
    
    if (device_ == Device::CUDA) {
        cuda_relu(d_data_, result.d_data_, total_size_);
    } else {
        for (int i = 0; i < total_size_; i++) {
            result.h_data_[i] = std::max(0.0f, h_data_[i]);
        }
    }
    
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape_, device_);
    
    if (device_ == Device::CUDA) {
        cuda_sigmoid(d_data_, result.d_data_, total_size_);
    } else {
        for (int i = 0; i < total_size_; i++) {
            result.h_data_[i] = 1.0f / (1.0f + std::exp(-h_data_[i]));
        }
    }
    
    return result;
}

std::vector<float> Tensor::to_vector() const {
    if (device_ == Device::CUDA) {
        const_cast<Tensor*>(this)->sync_to_host();
    }
    return std::vector<float>(h_data_, h_data_ + total_size_);
}

int Tensor::size() const {
    return total_size_;
}

Tensor Tensor::copy() const {
    Tensor result(shape_, device_);
    std::memcpy(result.h_data_, h_data_, total_size_ * sizeof(float));
    if (device_ == Device::CUDA) {
        cudaMemcpy(result.d_data_, d_data_, total_size_ * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return result;
}

Tensor Tensor::relu_derivative() const {
    Tensor result(shape_, device_);
    
    if (device_ == Device::CUDA) {
        cuda_relu_derivative(d_data_, result.d_data_, total_size_);
    } else {
        for (int i = 0; i < total_size_; i++) {
            result.h_data_[i] = h_data_[i] > 0.0f ? 1.0f : 0.0f;
        }
    }
    
    return result;
}

Tensor Tensor::sigmoid_derivative() const {
    Tensor result(shape_, device_);
    
    if (device_ == Device::CUDA) {
        cuda_sigmoid_derivative(d_data_, result.d_data_, total_size_);
    } else {
        for (int i = 0; i < total_size_; i++) {
            float sig = 1.0f / (1.0f + std::exp(-h_data_[i]));
            result.h_data_[i] = sig * (1.0f - sig);
        }
    }
    
    return result;
}

Tensor Tensor::softmax() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Softmax requires 2D tensor");
    }
    
    Tensor result(shape_, device_);
    int batch_size = shape_[0];
    int num_classes = shape_[1];
    
    if (device_ == Device::CUDA) {
        cuda_softmax(d_data_, result.d_data_, batch_size, num_classes);
    } else {
        for (int b = 0; b < batch_size; b++) {
            const float* input_batch = h_data_ + b * num_classes;
            float* output_batch = result.h_data_ + b * num_classes;
            
            // Find max
            float max_val = input_batch[0];
            for (int i = 1; i < num_classes; i++) {
                max_val = std::max(max_val, input_batch[i]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < num_classes; i++) {
                output_batch[i] = std::exp(input_batch[i] - max_val);
                sum += output_batch[i];
            }
            
            // Normalize
            for (int i = 0; i < num_classes; i++) {
                output_batch[i] /= sum;
            }
        }
    }
    
    return result;
}