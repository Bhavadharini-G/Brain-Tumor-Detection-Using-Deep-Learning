import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Load model
model = load_model('Models/model.h5')

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Title and layout
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")
st.title("🧠 Brain Tumor Detection Using Deep Learning")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

# Predict function
def predict_tumor(image):
    IMAGE_SIZE = 128
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# If image is uploaded
if uploaded_file is not None:
    # Save file to uploads directory
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"✅ File saved to {save_path}")

    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Predict
    with st.spinner("Analyzing the image..."):
        result, confidence = predict_tumor(image)

    # Show prediction
    st.subheader("🩺 Prediction Result:")
    st.success(result)
    st.info(f"Confidence: {confidence * 100:.2f}%")
