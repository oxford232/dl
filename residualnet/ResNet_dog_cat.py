from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
from torchvision import transforms
import torch.nn as nn
import os
from ResNet import ResNet, BasicBlock
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR


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
        label = torch.tensor(label, dtype=torch.long)
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label
    

def evaluate(model, test_dataloader):
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    return val_acc
    
if __name__ == "__main__":
    BATCH_SIZE = 64
    IMG_SIZE = 224
    EPOCHS = 15
    LR = 0.0005
    PRINT_STEP = 100
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    all_samples = verify_images(r"../PetImages")
    random.seed(42)
    random.shuffle(all_samples)
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

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2)

    # pretrained_model = resnet18(weights='IMAGENET1K_V1')
    # for param in model.parameters():
        # param.requires_grad = False

    # model = resnet18(weights=None)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    # model = model.to(DEVICE)

    # state_dict = pretrained_model.state_dict()
    # model.load_state_dict(state_dict)

    # model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(DEVICE)
    model.load_state_dict(torch.load('resnet18.pt', map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    


    for epoch in range(EPOCHS):
        print(f"epoch {epoch + 1}")
        model.train()
        running_loss = 0
        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            y = model(inputs)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % PRINT_STEP == 0:
                avg_loss = running_loss / PRINT_STEP
                print(f"  Step [{step + 1}] - Loss: {avg_loss:.4f}")
                running_loss = 0.0
        torch.save(model.state_dict(), "resnet18.pt")
    val_acc = evaluate(model, valid_dataloader)
    print(f"Validation Accuracy after epoch {epoch + 1}: {val_acc:.4f}")