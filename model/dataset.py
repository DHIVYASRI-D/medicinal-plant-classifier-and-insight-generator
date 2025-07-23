import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from PIL import Image

# Set paths (update if needed)
train_dir = "D:/leaf_dataset/train"
val_dir = "D:/leaf_dataset/val"

# Load ViT image processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Custom transform using processor config
def transform(example):
    image = Image.open(example).convert("RGB")
    return processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

# Custom Dataset class with labels
class PlantDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        pixel_values = transform(path)
        return {
            "pixel_values": pixel_values,
            "labels": label
        }

# Create datasets
train_dataset = PlantDataset(train_dir)
val_dataset = PlantDataset(val_dir)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Map: label index <-> class name
id2label = {idx: class_name for class_name, idx in train_dataset.class_to_idx.items()}
label2id = {class_name: idx for idx, class_name in id2label.items()}
