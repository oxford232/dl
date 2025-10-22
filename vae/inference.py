from vae import VAE
import torch
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
vae = VAE().to(device)
vae.load_state_dict(torch.load('vae.pt', map_location=device))

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