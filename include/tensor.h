#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>

enum class Device {
    CPU,
    CUDA
};

class Tensor {
public:
    // Constructors
    Tensor(const std::vector<int>& shape, Device device = Device::CPU);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data, Device device = Device::CPU);
    ~Tensor();

    // Core operations
    void to(Device device);
    void fill(float value);
    void randomize(float mean = 0.0f, float stddev = 1.0f);
    
    // Matrix operations (will be GPU-accelerated)
    static Tensor matmul(const Tensor& a, const Tensor& b);
    static Tensor add(const Tensor& a, const Tensor& b);
    static Tensor multiply(const Tensor& a, float scalar);
    Tensor transpose() const;
    
    // Activation functions
    Tensor relu() const;
    Tensor relu_derivative() const;
    Tensor sigmoid() const;
    Tensor sigmoid_derivative() const;
    Tensor softmax() const;
    
    // Utilities
    std::vector<float> to_vector() const;
    void from_vector(const std::vector<float>& data);
    const std::vector<int>& get_shape() const { return shape_; }
    int size() const;
    Device get_device() const { return device_; }
    float* data() { return device_ == Device::CUDA ? d_data_ : h_data_; }
    const float* data() const { return device_ == Device::CUDA ? d_data_ : h_data_; }
    
    // Copy operations
    Tensor copy() const;
    static void copy_to_device(float* dst, const float* src, int size);
    static void copy_to_host(float* dst, const float* src, int size);

private:
    std::vector<int> shape_;
    Device device_;
    float* h_data_;  // Host data
    float* d_data_;  // Device data
    int total_size_;
    
    void allocate_memory();
    void free_memory();
    void sync_to_host();
    void sync_to_device();
};

#endif // TENSOR_H