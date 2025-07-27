import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import ViTForImageClassification, get_scheduler
from data_utils.dataset_preparation import train_dataloader, val_dataloader, id2label, label2id  # make sure you import these correctly

# Load model with 80 classes
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=80,
    id2label=id2label,
    label2id=label2id
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 5
train_steps = len(train_dataloader) * num_epochs

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=train_steps
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        predictions = logits.argmax(-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values)
        predictions = outputs.logits.argmax(-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

val_acc = 100 * correct / total
print(f"Validation Accuracy: {val_acc:.2f}%")

# Save model
model.save_pretrained("outputs/vit-plant-classifier")


from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
processor.save_pretrained("D:/medicinal-plant-classifier-and-insight-generator/outputs/vit-plant-classifier")
