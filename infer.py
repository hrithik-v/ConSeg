import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import CustomDeepLabV3Plus
from dataloader import transform

# Inference configuration
img_path = 'assets/img.png'  # Path to test image
ckpt_path = 'checkpoints/ckpt_acc92_head.pth'  # Path to model checkpoint
num_classes = 13  # Should match training

def decode_segmap(mask, nc=13):
    """Map class indices to RGB colors for visualization."""
    label_colors = np.array([
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
        (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128)
    ])
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, nc):
        idx = mask == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Load and preprocess image
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Load model and weights
model = CustomDeepLabV3Plus(num_classes=num_classes, freeze_backbone=False)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()

with torch.no_grad():
    output = model(input_tensor)['out']
    pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

# Visualize segmentation result
seg_img = decode_segmap(pred, nc=num_classes)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Predicted Segmentation')
plt.imshow(seg_img)
plt.axis('off')
plt.show()
