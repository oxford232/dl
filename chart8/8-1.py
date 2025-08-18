import torch
from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)

    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f)
            for line in f:
                line = line.rsplit("\n")[0]
                # print(line)
                items = line.split(",")
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = torch.tensor(image)
        image = image / 255.0
        image = (image - 0.1307) / 0.3081
        label = torch.tensor(label)
        return image, label
    
    def __len__(self):
        return len(self.images)

batch_size = 128
train_dataset = MNISTDataset(r'./mnist/mnist_train.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset(r'./mnist/mnist_test.csv')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

learning_rate = 0.1
num_epochs = 100
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

layer_sizes = [28*28, 128, 128, 128, 64, 10]

weights = []
biases = []

for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
    W = torch.randn(in_size, out_size, device=device) * torch.sqrt(torch.tensor(2 / in_size))
    b = torch.zeros(out_size, device=device)
    weights.append(W)
    biases.append(b)

def relu(x):
    return torch.clamp(x, min=0)

def relu_grad(x):
    return (x > 0).float()

def softmax(x):
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return x_exp / x_exp.sum(dim=1, keepdim=True)

def cross_entropy(pred, labels):
    N = pred.shape[0]
    one_hot = torch.zeros_like(pred)
    one_hot[torch.arange(N), labels] = 1  # 生成one-hot编码
    loss = - (one_hot * torch.log(pred + 1e-8)).sum() / N  # 计算平均loss，这里加上一个很小的数1e-8，是为了防止出现log(0)时出现负无穷大的情况。
    return loss, one_hot

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        x = images.to(device)
        y = labels.to(device)
        N = x.shape[0]

        activations = [x]
        pre_acts = []
        for W, b in zip(weights[:-1], biases[:-1]):
            z = activations[-1] @ W + b
            pre_acts.append(z)
            a = relu(z)
            activations.append(a)

        z_out = activations[-1] @ weights[-1] + biases[-1]
        pre_acts.append(z_out)
        y_pred = softmax(z_out)

        loss, one_hot = cross_entropy(y_pred, y)
        total_loss += loss.item()

        #back
        grads_W = [None] * len(weights)
        grads_b = [None] * len(biases)

        dL_dz = (y_pred - one_hot) / N
        grads_W[-1] = activations[-1].t() @ dL_dz
        grads_b[-1] = dL_dz.sum(dim=0)

        for i in range(len(weights)-2, -1, -1):
            dL_dz = dL_dz @ weights[i+1].t() * relu_grad(pre_acts[i])
            grads_W[i] = activations[i].t() @ dL_dz
            grads_b[i] = dL_dz.sum(dim=0)

        with torch.no_grad():
            for i in range(len(weights)):
                weights[i] -= learning_rate * grads_W[i]
                biases[i] -= learning_rate * grads_b[i]
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.8f}")

#test
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        x = images.view(-1, layer_sizes[0]).to(device)
        y = labels.to(device)
        a = x
        for W, b in zip(weights[:-1], biases[:-1]):
            a = relu(a @ W + b)
        logits = a @ weights[-1] + biases[-1]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Test Accuracy: {correct/total*100:.2f}%")

