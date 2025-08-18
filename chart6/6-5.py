import torch
from torch.utils.tensorboard import SummaryWriter

device = device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#generate test data
inputs = torch.rand(100, 3)
weights = torch.tensor([[1.1], [2.2], [3.3]])
bias = torch.tensor(4.4)
targets = inputs @ weights + bias + 0.1*torch.randn(100, 1)

writer = SummaryWriter(log_dir="./")

w = torch.rand((3, 1), requires_grad=True, device=device)
b = torch.rand((1,), requires_grad=True, device=device)

inputs = inputs.to(device)
targets = targets.to(device)

epoch = 10000
lr = 0.003

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - targets))
    print("loss: ", loss.item())

    writer.add_scalar("loss/train", loss.item(), i)

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

print("parameter: ", w, b)




