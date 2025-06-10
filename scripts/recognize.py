import os
import sys
import json
import torch
from PIL import Image
import pytesseract
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F

# Ensure that pytesseract is correctly configured
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pill metadata from the JSON file
def load_metadata(metadata_path='pill_metadata.json'):
    if not os.path.exists(metadata_path):
        print(f"Metadata file '{metadata_path}' not found!")
        sys.exit(1)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

# Load the pretrained model
def load_model(model_path='C:/Users/hales/OneDrive/Desktop/pill_classifier/model/pill_classifier.pt'):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        sys.exit(1)
    
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(load_metadata()))  # Adjust output layer for the number of classes
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

# Process a single image and get predictions
def process_image(image_path, model, metadata):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
    
    # Get the predicted pill name from metadata
    predicted_pill = metadata.get(str(predicted_class.item()), 'Unknown')
    
    # Extract text from the image using OCR (optional)
    imprint_text = pytesseract.image_to_string(image).strip().replace('\n', ' ')

    # Return predictions and imprint text
    return predicted_pill, imprint_text

# Main function to process all pills
def process_all_pills(pills_directory, model, metadata):
    if not os.path.exists(pills_directory):
        print(f"Pills directory '{pills_directory}' not found!")
        sys.exit(1)

    for subdir, _, files in os.walk(pills_directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(subdir, file)
                print(f"Processing: {image_path}")
                predicted_pill, imprint_text = process_image(image_path, model, metadata)
                print(f"Predicted Pill: {predicted_pill}")
                print(f"Imprint Text: {imprint_text}")
                print("-" * 50)

# Main execution
if __name__ == "__main__":
    # Define the paths
    pills_directory = 'C:/Users/hales/OneDrive/Desktop/pill_classifier/data'  # Directory containing pill images
    metadata_path = 'pill_metadata.json'  # Path to the metadata file

    metadata = load_metadata(metadata_path)  # Load pill metadata
    model = load_model()  # Load the pretrained model

    # Process all pills in the dataset
    process_all_pills(pills_directory, model, metadata)
