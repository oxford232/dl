import torch

t1 = torch.tensor((2, 2), dtype=torch.float32)

t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])

t3 = torch.tensor([[[1, 2], [3, 4], [5, 6]]])

# print(t2.shape, t1[0].shape)
# print(t2, t2.shape)
# print(t3, t3.shape)

x = torch.tensor([1, 2, 3, 4, 5])
mask = x > 2
# print(mask, mask.shape)

filtered_x = x[mask]
# print(filtered_x, filtered_x.shape)

x[mask] = 0
# print(x, x.shape)

shape = (2, 3)

rand_tensor = torch.rand(shape)
# print(rand_tensor, rand_tensor.shape)
randn_tensor = torch.randn(shape)
# print(randn_tensor, randn_tensor.shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
twos_tensor = torch.full(shape, 2)

# print(ones_tensor, zeros_tensor, twos_tensor)

# reshape
x2 = torch.randn(4, 4)
x3 = x2.reshape(2, 8)
# print(x2, x3)

x5 = torch.tensor([[1, 2, 3], [4, 5, 6]])
x5_reshape = x5.reshape(3, 2)
x5_transpose = x5.permute(1, 0)
# print(x5_reshape, x5_transpose)


x6 = torch.randn(2,3,2)
x7 = x6.permute(1, 0, 2)
# print(x6, x7)

x8 = torch.tensor([[1, 2, 3], [4, 5, 6]])
x8_0 = x8.unsqueeze(0)
# print(x8.shape, x8_0.shape, x8, x8_0)

x8_1 = x8.unsqueeze(1)
# print(x8.shape, x8_1.shape, x8, x8_1)

x8_2 = x8.unsqueeze(2)
# print(x8.shape, x8_2.shape, x8, x8_2)

x9 = torch.tensor(1)
# print(x9.unsqueeze(0).unsqueeze(1).unsqueeze(0))

x10 = torch.ones((1,1,3))
# print(x10.shape, x10)
x10_0 = x10.squeeze(dim=0)
# print(x10_0.shape, x10_0)

x10_2 = x10.squeeze(2)
# print(x10_2.shape, x10_2)

x11 = torch.tensor([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])

mean = x11.mean()
# print(mean.shape)

mean = x11.mean(dim=0)
# print(mean.shape, mean)

mean = x11.mean(dim=0, keepdim=True)
# print(mean.shape, mean)

x12 = torch.tensor([[1,2,3], [4,5,6]])
# print(x12[0, 1])
# print(x12[:, 1])
# print(x12[1, :])
# print(x12[:, :2])
# print(x12[1][1])

x13 = torch.randn((3, 2))
# print(x13)
x13_1 = x13 + 1
# print(x13_1)

x14_1 = torch.ones((3, 2))
x14_2 = torch.ones(2)

x14_3 = x14_1 + x14_2

print(x14_1, x14_2, x14_3)




