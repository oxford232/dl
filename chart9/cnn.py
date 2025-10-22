from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
from torchvision import transforms
import torch.nn as nn
import os


def verify_images(image_folder):
    classes = ["Cat", "Dog"]
    class_to_idx = {"Cat": 0, "Dog": 1}
    samples = []
    for cls_name in classes:
        cls_dir = os.path.join(image_folder, cls_name)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(cls_dir, fname)
            try:
                with Image.open(path) as img:
                    img.verify()
                samples.append((path, class_to_idx[cls_name]))
            except Exception:
                 print(f"Warning: Skipping corrupted image {path}")
    return samples


class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label
    
if __name__ == "__main__":
    BATCH_SIZE = 64
    IMG_SIZE = 224
    EPOCHS = 100
    LR = 0.001
    PRINT_STEP = 100

    all_samples = verify_images(r"../PetImages")
    train_size = int(len(all_samples) * 0.8)
    train_samples = all_samples[:train_size]
    valid_samples = all_samples[train_size:]

    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(train_samples, data_transform)
    valid_dataset = ImageDataset(valid_samples, data_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):
        print('epoch start')
        for step, (inputs, labels) in enumerate(train_dataloader):
            if len(labels) < 64:
                print(len(labels), step)
            if step % 100 == 0:
                print(step, epoch)
        print('epoch done')

