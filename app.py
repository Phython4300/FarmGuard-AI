<<<<<<< HEAD
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import io

# Configuration
APP_TITLE = "FarmGuard AI"
APP_SUBHEADER = "Cassava Disease Detection System"
PAGE_ICON = "🌿"
MODEL_PATH = "cassava_model.tflite"
IMAGE_SIZE = (224, 224)

# Professional Color Palette
COLOR_PALETTE = {
    "primary": "#1A5F3F",        # Deep forest green
    "primary_light": "#2D7D59",  # Lighter green
    "secondary": "#F8F9FA",      # Light background
    "text_dark": "#2D3748",      # Dark text
    "text_medium": "#4A5568",    # Medium text
    "text_light": "#718096",     # Light text
    "card_bg": "#FFFFFF",        # Card background
    "border": "#E2E8F0",         # Border color
    "success": "#38A169",        # Success green
    "warning": "#DD6B20",        # Warning orange
    "error": "#E53E3E",          # Error red
    "accent": "#4CAF50"          # Accent green
}

# Disease Information
CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)", 
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]

# Enhanced disease info with icons and severity
DISEASE_INFO = {
    CLASS_NAMES[0]: {
        "icon": "🦠",
        "severity": "High Risk",
        "color": COLOR_PALETTE["warning"],
        "description": "Characterized by angular leaf spots, wilting, and stem cankers.",
        "recommendation": "• Use certified disease-free planting materials\n• Practice crop rotation\n• Disinfect tools regularly"
    },
    CLASS_NAMES[1]: {
        "icon": "⚠️", 
        "severity": "Critical Risk",
        "color": COLOR_PALETTE["error"],
        "description": "Shows yellow/brown streaks on leaves with root necrosis.",
        "recommendation": "• Remove infected plants immediately\n• Plant resistant varieties\n• Control whiteflies"
    },
    CLASS_NAMES[2]: {
        "icon": "🍃",
        "severity": "Moderate Risk",
        "color": COLOR_PALETTE["warning"],
        "description": "Displays green mottling or mosaic patterns on leaves.",
        "recommendation": "• Use certified planting materials\n• Control insect vectors"
    },
    CLASS_NAMES[3]: {
        "icon": "🔄",
        "severity": "High Risk",
        "color": COLOR_PALETTE["warning"],
        "description": "Causes leaf distortion and mosaic patterns.",
        "recommendation": "• Plant resistant varieties\n• Remove infected plants\n• Use yellow sticky traps"
    },
    CLASS_NAMES[4]: {
        "icon": "✅",
        "severity": "No Risk",
        "color": COLOR_PALETTE["success"],
        "description": "No signs of disease detected.",
        "recommendation": "• Continue monitoring\n• Maintain good practices"
    }
}

# --- Model Loading ---
@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

interpreter = load_tflite_model(MODEL_PATH)
if interpreter is None:
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- UI Styles ---
def apply_professional_styles():
    st.markdown(f"""
    <style>
        /* Base styles */
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: {COLOR_PALETTE["secondary"]};
            color: {COLOR_PALETTE["text_dark"]};
            line-height: 1.6;
        }}
        
        /* Header styles */
        .app-header {{
            background: linear-gradient(135deg, {COLOR_PALETTE["primary"]}, {COLOR_PALETTE["primary_light"]});
            color: white;
            padding: 2rem 1rem;
            border-radius: 0 0 20px 20px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        .app-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .app-subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
            font-weight: 400;
        }}
        
        /* Card styles */
        .card {{
            background: {COLOR_PALETTE["card_bg"]};
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            border: 1px solid {COLOR_PALETTE["border"]};
        }}
        
        .result-card {{
            border-left: 4px solid {COLOR_PALETTE["primary"]};
        }}
        
        /* Confidence meter */
        .confidence-meter {{
            height: 8px;
            background: #EDF2F7;
            border-radius: 4px;
            margin: 1rem 0;
            overflow: hidden;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, {COLOR_PALETTE["primary"]}, {COLOR_PALETTE["accent"]});
            border-radius: 4px;
        }}
        
        /* Section headers */
        .section-header {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {COLOR_PALETTE["primary"]};
            margin-bottom: 1.2rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {COLOR_PALETTE["border"]};
        }}
        
        /* Form elements */
        .file-upload-container {{
            border: 2px dashed {COLOR_PALETTE["border"]};
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background: rgba(255, 255, 255, 0.7);
            transition: all 0.3s ease;
        }}
        
        .file-upload-container:hover {{
            border-color: {COLOR_PALETTE["primary"]};
            background: rgba(26, 95, 63, 0.03);
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: {COLOR_PALETTE["primary"]} !important;
        }}
        
        .sidebar-title {{
            color: white !important;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        /* Buttons */
        .stButton>button {{
            background: {COLOR_PALETTE["primary"]};
            color: white;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            border: none;
            transition: all 0.2s;
        }}
        
        .stButton>button:hover {{
            background: {COLOR_PALETTE["primary_light"]};
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 1.5rem;
            margin-top: 3rem;
            color: {COLOR_PALETTE["text_medium"]};
            font-size: 0.9rem;
            border-top: 1px solid {COLOR_PALETTE["border"]};
        }}
    </style>
    """, unsafe_allow_html=True)

# --- Disease Card Component ---
def create_disease_card(disease_name, disease_data, confidence):
    # Pre-process the recommendation to replace newlines with HTML breaks
    recommendation_html = disease_data['recommendation'].replace('\n', '<br>')
    
    return f"""
    <div class="card result-card" style="border-left-color: {disease_data['color']}">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 1.8rem; margin-right: 0.8rem;">{disease_data['icon']}</span>
            <div>
                <h3 style="margin: 0; color: {disease_data['color']};">{disease_name}</h3>
                <p style="margin: 0; color: {COLOR_PALETTE['text_medium']};">{disease_data['severity']}</p>
            </div>
        </div>
        
        <p style="color: {COLOR_PALETTE['text_dark']};">
            <strong>Confidence:</strong> {confidence:.1f}%
        </p>
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {confidence}%"></div>
        </div>
        
        <div style="margin: 1.5rem 0;">
            <p style="color: {COLOR_PALETTE['text_dark']}; margin-bottom: 0.5rem;">
                <strong>Description</strong>
            </p>
            <p style="color: {COLOR_PALETTE['text_medium']};">
                {disease_data['description']}
            </p>
        </div>
        
        <div style="background: #F7FAFC; padding: 1rem; border-radius: 8px;">
            <p style="margin: 0; color: {COLOR_PALETTE['text_dark']};">
                <strong>Recommendations</strong>
            </p>
            <p style="color: {COLOR_PALETTE['text_medium']}; margin-top: 0.5rem;">
                {recommendation_html}
            </p>
        </div>
    </div>
    """

# --- Voice Advisory Tab ---
def voice_advisory_content():
    st.markdown('<div class="section-header">Voice Advisory</div>', unsafe_allow_html=True)
    
    with st.expander("How to use this feature", expanded=True):
        st.markdown("""
        **Describe symptoms in your own words:**
        1. Record or upload an audio file describing your cassava plants
        2. Our AI will transcribe your query
        3. Receive expert recommendations based on your description
        
        For best results:
        - Speak clearly about symptoms you observe
        - Describe leaf appearance, growth patterns, and affected areas
        - Mention any environmental conditions
        """)
    
    # Simplified version since we don't have whisper_transcriber in this example
    st.warning("Voice advisory service requires additional setup")
    st.info("Please see documentation for setting up the voice advisory feature")

# --- Main Application ---
def main():
    st.set_page_config(
        page_title=f"{APP_TITLE} | {APP_SUBHEADER}",
        layout="wide",
        page_icon=PAGE_ICON,
        initial_sidebar_state="expanded"
    )
    
    apply_professional_styles()
    
    # App Header
    st.markdown(f"""
    <div class="app-header">
        <div class="app-title">{APP_TITLE}</div>
        <div class="app-subtitle">{APP_SUBHEADER}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f'<div class="sidebar-title">{APP_TITLE}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: rgba(255,255,255,0.8);">{APP_SUBHEADER}</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        selected_tab = st.radio(
            "Navigation",
            ["Disease Detection", "Voice Advisory"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="color: white; padding: 0.5rem;">
            <p><strong>Agricultural Support</strong></p>
            <p>Contact your local extension office for field verification and additional resources.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    if selected_tab == "Disease Detection":
        st.markdown('<div class="section-header">Cassava Leaf Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Upload Leaf Image</h3>
                <p>For accurate results, please provide a clear image of cassava leaves.</p>
                <p style="color: #718096; font-size: 0.95rem;">
                    <strong>Best practices:</strong><br>
                    - Use natural lighting<br>
                    - Capture against plain background<br>
                    - Show both sides of leaves
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Select image file",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            
            if uploaded_file is None:
                st.info("Awaiting image upload for analysis")
        
        with col2:
            if uploaded_file is not None:
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    st.image(img, use_column_width=True)
                    
                    with st.spinner("Analyzing plant health..."):
                        # Preprocess image
                        img_resized = img.resize(IMAGE_SIZE)
                        img_array = image.img_to_array(img_resized) / 255.0
                        
                        # Predict
                        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img_array, axis=0).astype(np.float32))
                        interpreter.invoke()
                        output = interpreter.get_tensor(output_details[0]['index'])
                        
                        predicted_idx = np.argmax(output)
                        confidence = np.max(output) * 100
                        disease_name = CLASS_NAMES[predicted_idx]
                        disease_data = DISEASE_INFO[disease_name]
                        
                        # Display results
                        st.markdown(create_disease_card(disease_name, disease_data, confidence), unsafe_allow_html=True)
                        
                        # Probability distribution
                        st.markdown("""
                        <div class="card">
                            <h3>Diagnostic Confidence</h3>
                            <p>Probability distribution across possible conditions:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.bar_chart(
                            dict(zip(CLASS_NAMES, output[0] * 100)),
                            color=COLOR_PALETTE["primary"],
                            height=300
                        )
                
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
    
    elif selected_tab == "Voice Advisory":
        voice_advisory_content()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>FarmGuard AI | Precision Agriculture Solution | 3MTT Nigeria</p>
        <p>© 2025 | Developed by Nibrasudeen Kamal for Agricultural Innovation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
=======
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
>>>>>>> 191e879d08ad402996ad27385b5329ba3fdaf72f
