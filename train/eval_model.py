import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.dataset_preparation import PlantDataset, transform
import torch
from transformers import ViTForImageClassification
from evaluate import load  # now correctly uses Hugging Face's library
from torch.utils.data import DataLoader


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
val_dir = "D:/leaf_dataset/val"
model_dir = "D:/medicinal-plant-classifier-and-insight-generator/outputs/vit-plant-classifier"

# Load validation dataset and dataloader
val_dataset = PlantDataset(val_dir)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Load model
model = ViTForImageClassification.from_pretrained(model_dir)
model.to(device)
model.eval()

# Load accuracy metric
accuracy_metric = load("accuracy")

# Evaluation loop
for batch in val_dataloader:
    inputs = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Update metric
    accuracy_metric.add_batch(predictions=predictions, references=labels)

# Final result
final_accuracy = accuracy_metric.compute()
print("-------------------")
print(f"Validation Accuracy: {final_accuracy['accuracy'] * 100:.2f}%")
print("-------------------")