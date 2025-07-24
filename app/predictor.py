# app/predictor.py

from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image

# Load processor (from base model used during training)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load fine-tuned model
model_path = r"D:\medicinal-plant-classifier-and-insight-generator\outputs\vit-plant-classifier"
model = ViTForImageClassification.from_pretrained(model_path)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
    
    return model.config.id2label[predicted_class_idx]
