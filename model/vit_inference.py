import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Set device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
model_path = "D:/medicinal-plant-classifier-and-insight-generator/outputs/vit-plant-classifier"
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True).to(device)

# Normalize label map
id2label = {int(k): v for k, v in model.config.id2label.items()}
label_list = list(id2label.values())

def get_predictions(image: Image.Image, topk=3):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        topk_result = torch.topk(probs, k=topk)

    top_indices = topk_result.indices.tolist()
    top_scores = topk_result.values.tolist()
    top_labels = [id2label[i] for i in top_indices]

    return top_labels, top_scores, label_list
