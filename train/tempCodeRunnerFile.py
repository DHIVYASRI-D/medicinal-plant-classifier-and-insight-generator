from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
processor.save_pretrained("D:/medicinal-plant-classifier-and-insight-generator/outputs/vit-plant-classifier")