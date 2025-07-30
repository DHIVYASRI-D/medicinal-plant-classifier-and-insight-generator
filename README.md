# Medicinal Plant Classifier & Insight Generator

A lightweight, locally run app that classifies Indian medicinal plants from leaf images and generates meaningful insights, recommends YouTube videos about the plant.

## Project Overview

This app allows users to:

* Upload a leaf image
* Get the predicted plant class using a fine-tuned Vision Transformer (ViT)
* Confirm or correct the plant name
* Generate helpful plant insights using Gemini AI
* Optionally watch relevant YouTube guides


## Tech Stack

* **Model**: `google/vit-base-patch16-224-in21k` fine-tuned on the Indian Medicinal Leaf Dataset (**80 classes**)
* **Dataset**: [Kaggle: Indian Medicinal Leaf Dataset](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset) (labeled leaf images, 80 classes)
* **Frontend**: Streamlit
* **Image Classification**: Hugging Face Transformers (`AutoModelForImageClassification`)
* **Insight Generation**: Google Gemini 1.5 Flash via Google AI Studio
* **YouTube Integration**: YouTube Data API v3


## Features

### Leaf-Based Plant Classification

Upload a leaf photo -> Get prediction using a ViT model trained on medicinal plant images.

### AI-Generated Plant Insights

After confirming the plant, get:

* Medicinal Uses
* How to Use
* Growing Tips
* Precautions

(Generated via Google Gemini 1.5 Flash)

### YouTube Guide Finder

Fetches and displays relevant YouTube guides based on the confirmed plant name.

(Generated via YouTube Data API v3)

## Setup Instructions

> Requires Python 3.10+

1. **Clone the repo**

```bash
git clone https://github.com/DHIVYASRI-D/medicinal-plant-classifier-and-insight-generator.git
cd medicinal-plant-classifier-and-insight-generator
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app locally**

```bash
streamlit run app.py
```

4. **Use the Interface**

* Upload an image
* Confirm the plant name
* View insights + video links

## Model Access

Model also available at: [Hugging Face Model Hub](https://huggingface.co/dhivyasri-d/vit-plant-classifier)


## Created By

Dhivyasri D

[LinkedIn](https://www.linkedin.com/in/dhivyasri-d/) | [Hugging Face](https://huggingface.co/dhivyasri-d) 

