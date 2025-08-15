import torch
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

size = 10000
A_cpu = torch.rand(size, size)
B_cpu = torch.rand(size, size)

start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# on GPU

A_gpu = A_cpu.to(device)
B_gpu = B_cpu.to(device)

start_gpu = time.time()
C_gpu = torch.mm(A_gpu, B_gpu)
torch.mps.synchronize()
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"CPU time: {cpu_time:.6f} sec")
print(f"GPU time: {gpu_time:.6f} sec")