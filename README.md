# Medicinal Plant Classifier and Insight Generator

This project aims to build a system that:
1. Identifies medicinal plants from leaf and plant images using a fine-tuned Vision Transformer (ViT).
2. Generates insights about the plant in 4 categories:
   - Medicinal Uses
   - How to Use
   - Growing Tips
   - Precautions

## Dataset
The dataset includes 80 medicinal plant classes. Source: [Kaggle](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset/data).

## Tech Stack
- Vision Transformer (ViT) – Hugging Face Transformers
- Streamlit – for frontend
- Insight generation – gemini-1.5-flash



---
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
---


