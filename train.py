import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CustomDeepLabV3Plus
from dataloader import ConstructionDataset, transform, mask_transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
num_classes = 13  # Set according to your dataset
num_epochs = 50
batch_size = 128

# Model, optimizer, and loss
model = CustomDeepLabV3Plus(num_classes=num_classes, freeze_backbone=True).to(device)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Dataset paths
train_img_dir = 'dataset/train/images/'
train_mask_dir = 'dataset/train/masks/'
val_img_dir = 'dataset/valid/images/'
val_mask_dir = 'dataset/valid/masks/'
test_img_dir = 'dataset/test/images/'
test_mask_dir = 'dataset/test/masks/'

# Data loaders
train_dataset = ConstructionDataset(img_dir=train_img_dir, mask_dir=train_mask_dir, transform=transform, mask_transform=mask_transform)
val_dataset = ConstructionDataset(img_dir=val_img_dir, mask_dir=val_mask_dir, transform=transform, mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def compute_iou(preds, targets, num_classes):
    """Compute per-class Intersection over Union (IoU)."""
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        intersection = np.logical_and(preds == i, targets == i).sum()
        union = np.logical_or(preds == i, targets == i).sum()
        iou[i] = intersection / union if union != 0 else 0
    return iou


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    num_epochs=50,
    num_classes=13,
    device=None,
    checkpoint_path='./models/stage3/modelHead_acc',
    best_val_loss=float('inf'),
    best_val_accuracy=0
):
    """Main training loop for segmentation model."""
    train_losses = []
    val_losses = []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
        avg_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        iou = compute_iou(np.array(all_preds), np.array(all_targets), num_classes)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        print(f"IoU per class: {iou}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.squeeze(1).to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy().flatten())
                val_targets.extend(masks.cpu().numpy().flatten())
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_iou = compute_iou(np.array(val_preds), np.array(val_targets), num_classes)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
        # Save checkpoint if improved
        if avg_val_loss < best_val_loss or val_accuracy > best_val_accuracy:
            best_val_loss = min(avg_val_loss, best_val_loss)
            best_val_accuracy = max(val_accuracy, best_val_accuracy)
            torch.save(model.state_dict(), f"{checkpoint_path}_{int(accuracy*100)}.pth")
            print("Checkpoint saved!")
    print("Training complete.")
    return train_losses, val_losses


if __name__ == "__main__":
    # Start training
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        num_classes=num_classes,
        device=device
    )
