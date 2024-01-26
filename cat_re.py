from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from torchvision import transforms
#thu mmot xiu
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

def preprocess(image_path):
    image = Image.open(image_path)

    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image = preprocess(image)
    return image

def predict(image_path):
    # Preprocess the image
    image = preprocess(image_path)

    # Prepare the image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Get the model's prediction
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the predicted class
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]

image_path = "meo.jpg"
print(f"Predicted class: {predict(image_path)}")
