# cassava_tester.py

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Set page configuration
st.set_page_config(page_title="FarmGuard AI", page_icon="🌿")

# Load the TFLite model safely
MODEL_PATH = "cassava_model.tflite"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Make sure it is in the same directory.")
    st.stop()

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_names = ['CBSD', 'CBB', 'CGM', 'CMD', 'Healthy']

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    confidence = np.max(output)
    return class_names[predicted_class], confidence

# Streamlit UI
st.title("🌿 FarmGuard AI – Cassava Disease Detector")
st.write("Upload a cassava leaf image and let AI detect the disease.")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    with st.spinner("🔎 Classifying..."):
        label, confidence = predict(image)

    st.success("✅ Prediction complete!")
    st.markdown(f"### 🧠 **Prediction**: `{label}`")
    st.markdown(f"### 📊 **Confidence**: `{confidence * 100:.2f}%`")
