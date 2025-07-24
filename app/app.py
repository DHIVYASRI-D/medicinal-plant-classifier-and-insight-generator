import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your fine-tuned model directory
model_path = "D:/medicinal-plant-classifier-and-insight-generator/outputs/vit-plant-classifier"

# Load processor and model
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True).to(device)

# Normalize id2label keys to integers (for safe lookup)
id2label_raw = model.config.id2label
id2label = {int(k): v for k, v in id2label_raw.items()}

# Streamlit UI
st.title("Medicinal Leaf Classifier ")
st.write("Upload a leaf image to identify the medicinal plant.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_id = logits.argmax(-1).item()
            label = id2label[pred_id]
            confidence = probs[0][pred_id].item()

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence:.2%}")

    except UnidentifiedImageError:
        st.error("Could not read image. Please upload a valid image file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
