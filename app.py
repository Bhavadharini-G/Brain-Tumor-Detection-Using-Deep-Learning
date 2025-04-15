import streamlit as st

# ✅ Set page config FIRST
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")

import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
import gdown

# === Download model from Google Drive if not already present ===
MODEL_PATH = 'Models/model.h5'
MODEL_ID =  '16ZEqwgUvhoY31gUXUEiTmKzLad5EB0OA'  

if not os.path.exists(MODEL_PATH):
    os.makedirs('Models', exist_ok=True)
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# Load the model
model = load_model(MODEL_PATH)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Ensure upload folder exists
UPLOAD_DIR = "Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("🧠 Brain Tumor Detection Using Deep Learning")
st.markdown(
    "<p style='font-size:18px;'>Upload an MRI image, and this app will predict whether it contains a brain tumor.</p>",
    unsafe_allow_html=True,
)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class_index = np.argmax(predictions)
    confidence_score = np.max(predictions)

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score, predictions
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score, predictions

# Main logic
if uploaded_file is not None:
    try:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Image saved to: `{save_path}`")

        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded MRI Image", use_column_width=True)

        with st.spinner("🔎 Analyzing the image..."):
            result, confidence, all_probs = predict_tumor(save_path)

        st.subheader("🩺 Prediction Result")
        st.success(result)
        st.info(f"🔬 Confidence: {confidence * 100:.2f}%")

        st.subheader("📊 Class Probabilities")
        for i, label in enumerate(class_labels):
            st.write(f"• **{label.capitalize()}**: {all_probs[i] * 100:.2f}%")

        result_text = f"Prediction: {result}\nConfidence: {confidence * 100:.2f}%"
        st.download_button("📥 Download Result", data=result_text, file_name="tumor_prediction.txt")

    except Exception as e:
        st.error("⚠️ Error processing the image. Please try again.")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<small>🔬 This app is for educational purposes only and is not intended for medical diagnosis.</small>",
    unsafe_allow_html=True
)
