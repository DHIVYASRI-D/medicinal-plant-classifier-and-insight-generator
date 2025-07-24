from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
import torch

model_dir = "D:/medicinal-plant-classifier-and-insight-generator/outputs/vit-plant-classifier"
image_path = r"C:\Users\dhivy\Downloads\Ecuualyptus.jpg"

# Load processor and model
processor = AutoImageProcessor.from_pretrained(model_dir)
model = ViTForImageClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prepare input
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()

# Get class label
id2label = model.config.id2label
print(f"Predicted class: {id2label[predicted_class]}")
