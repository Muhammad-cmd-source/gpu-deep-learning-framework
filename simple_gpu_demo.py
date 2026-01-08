import torch
import numpy as np
import time
import matplotlib.pyplot as plt

print("="*70)
print("GPU-ACCELERATED DEEP LEARNING FRAMEWORK")
print("="*70)

# Check GPU
if not torch.cuda.is_available():
    print("\n❌ CUDA not available!")
    exit()

print(f"\n✅ GPU Detected: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA Version: {torch.version.cuda}")
print(f"✅ PyTorch Version: {torch.__version__}")

# Configuration
input_size = 2048
hidden_size = 1024
output_size = 10
batch_size = 512
num_epochs = 100

print(f"\n📊 Configuration:")
print(f"   Architecture: {input_size} → {hidden_size} → {output_size}")
print(f"   Batch size: {batch_size}")
print(f"   Epochs: {num_epochs}")

# Generate data
X_np = np.random.randn(batch_size, input_size).astype(np.float32)
y_np = np.random.randn(batch_size, output_size).astype(np.float32)

# CPU Training
print("\n" + "-"*70)
print("🔵 CPU TRAINING")
print("-"*70)

X_cpu = torch.from_numpy(X_np)
y_cpu = torch.from_numpy(y_np)

W1_cpu = torch.randn(input_size, hidden_size) * 0.01
W2_cpu = torch.randn(hidden_size, output_size) * 0.01

cpu_times = []
print("Training on CPU...")

for epoch in range(num_epochs):
    start = time.time()
    
    h = torch.relu(torch.matmul(X_cpu, W1_cpu))
    output = torch.matmul(h, W2_cpu)
    loss = torch.mean(output ** 2)
    
    cpu_times.append(time.time() - start)
    
    if epoch % 20 == 0:
        print(f"  Epoch {epoch}/{num_epochs}: {cpu_times[-1]*1000:.2f}ms")

cpu_avg = np.mean(cpu_times) * 1000
cpu_total = sum(cpu_times)

print(f"\n✓ CPU Complete")
print(f"  Average: {cpu_avg:.2f} ms/epoch")
print(f"  Total: {cpu_total:.2f} seconds")

# GPU Training
print("\n" + "-"*70)
print("🟢 GPU TRAINING")
print("-"*70)

X_gpu = torch.from_numpy(X_np).cuda()
y_gpu = torch.from_numpy(y_np).cuda()

W1_gpu = torch.randn(input_size, hidden_size, device='cuda') * 0.01
W2_gpu = torch.randn(hidden_size, output_size, device='cuda') * 0.01

# Warmup
for _ in range(5):
    h = torch.relu(torch.matmul(X_gpu, W1_gpu))
    output = torch.matmul(h, W2_gpu)
    torch.cuda.synchronize()

gpu_times = []
print("Training on GPU...")

for epoch in range(num_epochs):
    torch.cuda.synchronize()
    start = time.time()
    
    h = torch.relu(torch.matmul(X_gpu, W1_gpu))
    output = torch.matmul(h, W2_gpu)
    loss = torch.mean(output ** 2)
    
    torch.cuda.synchronize()
    gpu_times.append(time.time() - start)
    
    if epoch % 20 == 0:
        print(f"  Epoch {epoch}/{num_epochs}: {gpu_times[-1]*1000:.2f}ms")

gpu_avg = np.mean(gpu_times) * 1000
gpu_total = sum(gpu_times)
speedup = cpu_avg / gpu_avg

print(f"\n✓ GPU Complete")
print(f"  Average: {gpu_avg:.2f} ms/epoch")
print(f"  Total: {gpu_total:.2f} seconds")

# Results
print("\n" + "="*70)
print("🎯 FINAL RESULTS")
print("="*70)
print(f"\n📊 Average Time per Epoch:")
print(f"   CPU: {cpu_avg:.2f} ms")
print(f"   GPU: {gpu_avg:.2f} ms")
print(f"\n⚡ SPEEDUP: {speedup:.2f}x")
print(f"\n⏱️  Total Training Time:")
print(f"   CPU: {cpu_total:.2f} seconds")
print(f"   GPU: {gpu_total:.2f} seconds")
print(f"   Time Saved: {cpu_total - gpu_total:.2f} seconds")

if speedup >= 10:
    print(f"\n🎉 EXCELLENT! Achieved {speedup:.1f}x speedup (>10x target!)")
elif speedup >= 5:
    print(f"\n✅ GREAT! Achieved {speedup:.1f}x speedup")
else:
    print(f"\n✓ Good! Achieved {speedup:.1f}x speedup on MX230")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training times
ax1 = axes[0]
epochs = range(1, num_epochs + 1)
ax1.plot(epochs, np.array(cpu_times) * 1000, label='CPU', color='#e74c3c', linewidth=2, alpha=0.7)
ax1.plot(epochs, np.array(gpu_times) * 1000, label='GPU', color='#2ecc71', linewidth=2, alpha=0.7)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Training Time per Epoch: CPU vs GPU', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup
ax2 = axes[1]
bar = ax2.bar(['GPU Speedup'], [speedup], color='#3498db', alpha=0.8, width=0.4)
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No speedup')
ax2.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10x target')
ax2.set_ylabel('Speedup Factor (x)', fontsize=12, fontweight='bold')
ax2.set_title('GPU vs CPU Speedup', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, max(speedup * 1.2, 12)])
ax2.text(0, speedup, f'{speedup:.1f}x', ha='center', va='bottom', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('gpu_speedup_results.png', dpi=300, bbox_inches='tight')
print("\n💾 Results saved as 'gpu_speedup_results.png'")
plt.show()

print("\n" + "="*70)
print("✅ BENCHMARK COMPLETE!")
print("="*70)
print("\n📋 Technologies Demonstrated:")
print("   • Python for high-level orchestration")
print("   • PyTorch for tensor operations")
print("   • CUDA for GPU acceleration")
print("   • NumPy for data generation")
print("   • Matplotlib for visualization")