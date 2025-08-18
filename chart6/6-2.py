import torch

z = torch.tensor([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])

mean = z.mean()
print("mean: ", mean)


mean = z.mean(dim = 0)
print("mean pn dim 0:", mean)

mean = z.mean(dim = 0, keepdim=True)
print("keepdim: ", mean)

t1 = torch.randn((3, 2))
t2 = t1 + 1

print(t1)
print(t2)