Pill Identifier
An AI-powered web application that identifies pills from images using deep learning.

Overview
Pill Identifier uses a ResNet18-based deep learning model to recognize and identify pills from uploaded
images. The system provides information about the identified pill, including its name, dosage, intended
use, and potential side effects.

Features
Image Upload: Upload pill images through drag-and-drop or file selection
AR Camera Integration: Use device camera to capture pill images directly
Real-time Analysis: Deep learning model analyzes and identifies pills
Confidence Score: Display of model confidence level with visual indicator
Responsive Design: Mobile-friendly interface

Directory Structure
pill-identifier/
├── static/ # Static assets (images, charts)
│ ├── pill_logo.png
│ ├── chart.png
│ └── training_graph.png
├── uploads/ # Temporary storage for uploaded images
├── model/ # Trained model files
│ └── pill_classifier.pt
├── data/ # Training dataset (organized by pill class)
├── templates/ # HTML templates
│ └── index.html
├── pill_metadata.json # Pill information database
├── app.py # Flask web application
├── train.py # Model training script
└── requirements.txt # Python dependencies

Setup Instructions
1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Ensure the model directory and pill_metadata.json file exist
mkdir -p model uploads static

4. Train the model or download pre-trained weights
python train.py # Only if you have the training dataset

5.  Run the application
python app.py

6. Open your browser and navigate to http://127.0.0.1:5000

Usage
1. Upload an image of a pill using the upload button or drag-and-drop
2. Alternatively, use the AR Camera to capture a pill image directly
3. View the identification results, including:
Pill name
Dosage information
Usage details
Potential side effects
Confidence level of the identification

Model Information
Architecture: ResNet18 (transfer learning)
Training dataset: Custom pill image dataset
Accuracy metrics available in the Chart section of the application

Disclaimer
This application is for informational purposes only and should not replace professional medical advice.
Always consult healthcare professionals for medical guidance.

Team Members
Pavan
Arshiya Khan
Halesh M S






