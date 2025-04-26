import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms

# Normalization parameters for DeepLabV3 (ImageNet)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Image transformation for input images
transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class ToTensorMask:
    """Convert PIL mask to tensor without normalization."""
    def __call__(self, pic):
        return torch.tensor(np.array(pic), dtype=torch.long)

# Transformation for segmentation masks
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    ToTensorMask()
])

class ConstructionDataset(Dataset):
    """Custom dataset for image-mask pairs used in segmentation."""
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        mask_name = os.path.join(self.mask_dir, self.img_names[idx].replace('.jpg', '_mask.png'))  # Adjust extension as needed
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask
