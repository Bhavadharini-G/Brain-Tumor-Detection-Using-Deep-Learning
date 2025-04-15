from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import gdown  # for downloading from Google Drive

# Initialize Flask app
app = Flask(__name__)

# Define model path and Google Drive file ID
MODEL_PATH = 'Models/model.h5'
MODEL_ID = '16ZEqwgUvhoY31gUXUEiTmKzLad5EB0OA'  # <-- Your file ID

# Create Models directory if it doesn't exist
os.makedirs("Models", exist_ok=True)

# Download the model if not already downloaded
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the trained model
model = load_model(MODEL_PATH)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/Uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded images
@app.route('/Uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
