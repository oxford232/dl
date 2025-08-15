import torch
import numpy as np
print(torch.__version__)

t1 = torch.tensor([1, 2, 3])
# print(t1)

t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(t2)

t3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(t3)

arr = np.array([1, 2, 3])
t_np = torch.tensor(arr)
# print(t_np)

tensor = torch.rand(3, 4)

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x_reshape = x.reshape(3, 2)
x_transpose = x.permute(1,0)

# print("permute: ", x_transpose)

x_0 = x.unsqueeze(0)
print(x_0.shape, x_0)


x_1 = x.unsqueeze(1)
print(x_1.shape, x_1)


x_2 = x.unsqueeze(2)
print(x_2.shape, x_2)

y = torch.ones((1,1,3))

y2 = torch.ones((1))

y_1 = y.squeeze()

print(y_1.shape, y_1)

a = torch.ones((2, 3))
b = torch.ones((2, 3))

print(a + b)
print(a - b)
print(a @ b.t())