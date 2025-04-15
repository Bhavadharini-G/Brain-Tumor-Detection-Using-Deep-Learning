# Brain Tumor Detection Using Deep Learning

This project is a deep learning-powered web application that classifies brain tumors from MRI scans. It supports two interfaces: **Streamlit** and **Flask**, allowing interactive image upload and prediction. The model distinguishes between four tumor types.

---

## Project Highlights

- Upload MRI images in JPG/PNG format  
- Predict tumor type using a trained deep learning model  
- Confidence score output for predictions  
- Saves uploaded files to `Uploads/` directory  
- Dual UI support: Streamlit and Flask  

---

## Tumor Types Detected

- **Glioma**  
- **Meningioma**  
- **Pituitary**  
- **No Tumor**

---

## Model Information

- Framework: **TensorFlow/Keras**
- Architecture: Custom CNN
- Input Image Size: **128x128**
- Model File: `Models/model.h5`

---

## File & Folder Structure

```
BrainTumorDetection/
├── MRI Images/            # Optional: Dataset
├── Models/                # Trained model stored here
│   └── model.h5
├── Templates/             # HTML templates for Flask
│   └── index.html
├── Uploads/               # Uploaded MRI image files
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md              # Project documentation
├── app.py                 # Streamlit interface
├── main.py                # Flask interface
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version for Streamlit Cloud
```

---

## Getting Started

### Install Requirements

```bash
pip install -r requirements.txt
```

---

### Run with Streamlit

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

---

### Run with Flask

```bash
python main.py
```

Open your browser and go to `http://127.0.0.1:5000`

---

## File Upload Behavior

- Uploaded MRI images are stored inside the `Uploads/` directory.
- Both **Streamlit** and **Flask** versions save uploaded files and display predictions.

---

## Sample UI Screenshot

![App Screenshot](Uploads/sample.jpg) <!-- Replace with your own image if needed -->

---

## Deploy to Streamlit Cloud

> Make sure `runtime.txt` includes:

```
python-3.10
```

Push your code to GitHub, then link it to [Streamlit Cloud](https://streamlit.io/cloud).

---

## Author

**Bhavadharini G**  
M.Tech Data Science, KCT  
[gunasekaranbhavadharini@gmail.com](mailto:gunasekaranbhavadharini@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
