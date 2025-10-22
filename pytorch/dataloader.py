import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

t = torch.arange(7, dtype=torch.float32)
print(t)
data_loader = DataLoader(t, batch_size=3, drop_last=False)


torch.manual_seed(1)
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)

class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
joint_dataset = JointDataset(t_x, t_y)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)

for example in data_loader:
    print(' x:', example[0], ' y:', example[1])

