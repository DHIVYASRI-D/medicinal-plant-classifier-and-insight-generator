import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from data_utils.dataset_preparation import PlantDataset  # your dataset.py script


# Paths
test_dir = "D:/leaf_dataset/test"
model_dir = "D:/medicinal-plant-classifier-and-insight-generator/outputs/vit-plant-classifier"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_dataset = PlantDataset(test_dir)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Create label mappings
id2label = {idx: class_name for class_name, idx in test_dataset.class_to_idx.items()}
label2id = {class_name: idx for idx, class_name in id2label.items()}

# Load model from directory
model = ViTForImageClassification.from_pretrained(
    model_dir,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
model.to(device)
model.eval()

# Inference
correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataloader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=inputs)
        _, preds = torch.max(outputs.logits, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print("---------------")
print(f"Test Accuracy on unseen data: {accuracy:.2f}%")
print("---------------")