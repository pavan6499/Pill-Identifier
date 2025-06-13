from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pill info (make sure pill_data.json exists with the pill info)
with open('pill_metadata.json') as f:
    pill_data = json.load(f)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model (no pre-trained weights initially, since we're loading custom weights)
model = resnet18(weights=None)  # No pre-trained weights for now, we're loading custom ones
model.fc = torch.nn.Linear(model.fc.in_features, len(pill_data))  # Adjust the final layer

# Load the custom checkpoint
checkpoint = torch.load('model/pill_classifier.pt', map_location=device)

# Extract the state_dict and class_names from the checkpoint
model_state_dict = checkpoint.get('model_state_dict')
class_names = checkpoint.get('class_names')

# Load the state_dict into the model
if model_state_dict is None:
    raise ValueError("Model state_dict not found in the checkpoint.")
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Open and transform the image
    img = Image.open(filepath).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Get the predicted class and its confidence
    class_idx = predicted.item()
    confidence_score = confidence.item()
    class_name = class_names[class_idx]

    # Get pill information from the pill_data
    pill_info = pill_data.get(class_name, {
        "name": "Unknown",
        "dosage": "N/A",
        "usage": "N/A",
        "side_effects": "N/A"
    })

    # Prepare the response
    response = {
        "name": class_name if confidence_score >= 0.5 else "Low Confidence - Check Again",
        "dosage": pill_info.get("dosage", "N/A"),
        "usage": pill_info.get("usage", "N/A"),
        "side_effects": pill_info.get("side_effects", "N/A"),
        "confidence": round(confidence_score * 100, 2),
        "image_url": f"/uploads/{file.filename}"
    }

    return jsonify(response)

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ✅ Modified only this part for Render compatibility
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
