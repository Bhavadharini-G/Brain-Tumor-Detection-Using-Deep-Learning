import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

# Load model
model = load_model('Models/model.h5')

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Ensure Uploads directory exists
UPLOAD_DIR = "Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Page config and title
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")
st.title("🧠 Brain Tumor Detection Using Deep Learning")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

# Predict function
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

# Main logic
if uploaded_file is not None:
    # Save file to Uploads folder
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"✅ Image saved to: {save_path}")

    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Predict
    with st.spinner("Analyzing the image..."):
        result, confidence = predict_tumor(save_path)

    # Display result
    st.subheader("🩺 Prediction Result:")
    st.success(result)
    st.info(f"Confidence: {confidence * 100:.2f}%")
