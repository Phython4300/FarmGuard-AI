import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('your_model.h5')  # Replace with your actual model

# Class names (update with your actual classes)
CLASS_NAMES = ['Cassava Healthy', 'Cassava Mosaic', 'Maize Rust', 'Tomato Blight']

st.title('FarmGuard AI - Crop Disease Detector')
st.write("Upload a crop leaf photo for instant AI diagnosis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = image.resize((224, 224))  # Match your model's input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])
    
    # Show results
    st.subheader("Diagnosis Results:")
    st.write(f"**{CLASS_NAMES[np.argmax(score)]}** with {100 * np.max(score):.2f}% confidence")
    
    # Show treatment advice
    disease = CLASS_NAMES[np.argmax(score)]
    if "Mosaic" in disease:
        st.info("💡 **Treatment:** Remove infected plants, use resistant varieties")
    elif "Rust" in disease:
        st.info("💡 **Treatment:** Apply fungicide, ensure proper spacing")
    elif "Blight" in disease:
        st.info("💡 **Treatment:** Remove affected leaves, improve air circulation")
    else:
        st.success("✅ Plant is healthy! Maintain current practices")