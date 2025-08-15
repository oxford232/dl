import torch

#2∗lights+0.01∗distance+5

device = device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

inputs = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19], [31], [14], [15], [43]], dtype=torch.float, device=device)

w = torch.ones(2, 1, requires_grad=True, device=device)
b = torch.ones(1, requires_grad=True, device=device)

# inputs = inputs / torch.tensor([4, 3000], device=device)

mean = inputs.mean(dim=0)
std = inputs.std(dim=0)

inputs = (inputs - mean) / std

epoch = 2000
# lr = 0.0000001
# lr = 0.0000001

lr = 0.1

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    print("loss: ", loss.item())
    loss.backward()
    # print("w.grad: ", w.grad.tolist())

    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr

    w.grad.zero_()
    b.grad.zero_()

print("parameter: ", w, b)

new_input = torch.tensor([[3, 2500]], dtype=torch.float, device=device)

new_input = (new_input-mean)/std

predict = new_input @ w + b

print("Predict: ", predict.tolist())