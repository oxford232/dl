import torch

t1 = 2 * torch.rand(5, 2) - 1
t2 = torch.normal(mean=0, std=1, size=(5, 2))

a = torch.tensor([[5, 7, 9], [3, 5, 6]], dtype=torch.float32)
b = torch.tensor([[3, 3, 3], [3, 3, 3]], dtype=torch.float32)

t3 = torch.multiply(t1, t2)
t5 = torch.mean(a, axis=1)
t6 = torch.matmul(a, torch.transpose(b, 0, 1))
t7 = torch.linalg.norm(a, ord=2, dim=1)

# print(t5)

tsp = torch.rand(6)
print(tsp)

t_splits = torch.chunk(tsp, 3)

t_splits2 = torch.split(tsp, split_size_or_sections=[3, 2, 1])

# print([item.numpy() for item in t_splits2])

aa = torch.ones(3)
bb = torch.zeros(3)
cc = torch.cat([aa, bb], axis=0)
dd = torch.stack([aa, bb], axis=1)

print(dd)