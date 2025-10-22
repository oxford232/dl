import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400,latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))
    

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    

def loss_function(x_recon, x, mu, logvar):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vae = VAE().to(device)
vae.load_state_dict(torch.load('vae.pt', map_location=device))
# optimizer = torch.optim.Adam(vae.parameters(), lr=1e-7)


# transform = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
# train_loader = DataLoader(MNIST('.', train=True, download=True, transform=transform), batch_size=128, shuffle=True)


# for epoch in range(1, 10000):
#     vae.train()
#     train_loss = 0
#     for x, _ in train_loader:
#         x = x.to(device)
#         x_recon, mu, logvar = vae(x)
#         loss, bce, kld = loss_function(x_recon, x, mu, logvar)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     print(f'Epoch {epoch}, Loss: {train_loss/len(train_loader.dataset):f}')
#     if epoch % 50 == 0:
#         torch.save(vae.state_dict(), "vae.pt")

# torch.save(vae.state_dict(), "vae.pt")



vae.eval()  # 切换到评估模式
num_samples = 16              # 生成16个新样本
latent_dim = 20               # 请确保与训练时一致

# 1. 从标准正态分布采样潜在向量 z
z = torch.randn(num_samples, latent_dim).to(device)

# 2. 用 Decoder 生成新样本
with torch.no_grad():
    generated = vae.decoder(z)  # 输出形状: [num_samples, 784]

# 3. 将生成结果还原成图片格式
generated = generated.cpu().view(-1, 28, 28)  # MNIST图片28x28

# 4. 可视化
fig, axes = plt.subplots(4, 4, figsize=(5, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated[i], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()