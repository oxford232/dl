from PIL import Image
from ResNet import ResNet, BasicBlock
import os
from torchvision import transforms
import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import json
from torch.utils.data import DataLoader, Dataset
import cairosvg

class IconDataSet(Dataset):
    def __init__(self, file_path):
        self.icons, self.labels = self._read_file(file_path)

    def _read_file(self, file_path):
        icons = []
        labels = []
        with open(file_path, 'r') as file:
            data = json.load(file)
        print(type(data))
        svgByteString = data['Upload'][0].encode('utf-8')
        cairosvg.svg2jpeg(
            bytestring=svgByteString,
            write_to='icons/testconvert.jpeg',
            output_width=224,
            output_height=224
        )
        return icons, labels


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
icon_vector = []
IMG_SIZE = 224

data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# for param in model.parameters():
#     param.requires_grad = False

# model.fc = nn.Identity()
# model.eval()

model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
pretrained_model = resnet18(weights='IMAGENET1K_V1')

state_dict = pretrained_model.state_dict()
model.load_state_dict(state_dict)

model.fc = nn.Identity()
# model.to(DEVICE)

model.eval()

iconDataSet = IconDataSet(r'./icons/nameToSvg.json')




def read_icon(image_folder):
    cls_dir = os.path.join(image_folder)
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(cls_dir, fname)
        print(path)
        # try:
        with Image.open(path) as img:
            # img.verify()
            img = img.convert('RGB')
            img = data_transform(img)
            # img.to(DEVICE)
            img = img.unsqueeze(0)
            output = model.forward(img)
            icon_vector.append(output)
        # except Exception:
        #         print(f"Warning: Skipping corrupted image {path}")


# read_icon(r"./icons")


# with Image.open(r"./testdata/test.png") as img:
#     img = img.convert('RGB')
#     img = data_transform(img)
#     img = img.unsqueeze(0)
#     print(img.shape)
#     output = model.forward(img)
#     for v in icon_vector:
#         cos_sim = F.cosine_similarity(v, output, dim=1)
#         print("sim: ", cos_sim)