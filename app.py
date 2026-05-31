import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Leaf Disease Detection System")
st.write("Upload a tomato leaf image to detect diseases using AI.")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model_training/tomato_model.h5"
    )

model = load_model()

# ==========================
# CLASSES
# ==========================
class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Target_Spot",
    "Tomato_healthy"
]

# ==========================
# DISEASE INFO
# ==========================
disease_info = {

    "Tomato_Bacterial_spot": {
        "description": "Small dark water-soaked spots appear on leaves.",
        "treatment": "Use copper-based fungicides and remove infected leaves."
    },

    "Tomato_Early_blight": {
        "description": "Brown circular spots with concentric rings.",
        "treatment": "Apply fungicide and practice crop rotation."
    },

    "Tomato_Late_blight": {
        "description": "Large dark lesions causing rapid leaf death.",
        "treatment": "Remove infected plants and spray fungicide."
    },

    "Tomato_Target_Spot": {
        "description": "Brown spots with yellow halos on leaves.",
        "treatment": "Apply fungicide and maintain field sanitation."
    },

    "Tomato_healthy": {
        "description": "The leaf appears healthy.",
        "treatment": "No treatment required."
    }
}

# ==========================
# REFERENCE IMAGES
# ==========================
reference_folders = {
    "Tomato_Bacterial_spot": "reference_images/Tomato_Bacterial_spot",
    "Tomato_Early_blight": "reference_images/Tomato_Early_blight",
    "Tomato_Late_blight": "reference_images/Tomato_Late_blight",
    "Tomato_Target_Spot": "reference_images/Tomato_Target_Spot",
    "Tomato_healthy": "reference_images/Tomato_healthy"
}

# ==========================
# REFERENCE IMAGE FUNCTION
# ==========================
def get_reference_image(class_name):

    folder = reference_folders.get(class_name)

    if not folder or not os.path.exists(folder):
        return None

    files = os.listdir(folder)

    if len(files) == 0:
        return None

    return os.path.join(folder, files[0])

# ==========================
# IMAGE PREPROCESSING
# ==========================
def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((256, 256))

    img = np.array(img, dtype=np.float32)
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# PREDICTION
# ==========================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    processed_image = preprocess_image(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    st.subheader("Prediction Result")

    st.success(f"Detected Disease: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")

    # Disease Information
    info = disease_info[predicted_class]

    st.subheader("Disease Description")
    st.write(info["description"])

    st.subheader("Recommended Treatment")
    st.write(info["treatment"])

    # Prediction Probabilities
    st.subheader("Prediction Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(
            f"{class_names[i]} : {prob * 100:.2f}%"
        )

    # Reference Image
    image_path = get_reference_image(predicted_class)

    if image_path:

        ref_image = Image.open(image_path)

        st.subheader("Reference Image")

        st.image(
            ref_image,
            caption=predicted_class,
            use_container_width=True
        )

    else:
        st.warning("Reference image not found.")