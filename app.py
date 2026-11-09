import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration
IMAGE_SIZE = (160, 160)
MODEL_PATH = "models/crop_disease_model.keras"
LABELS_PATH = "models/class_names.json"

st.set_page_config(
    page_title="Crop Pest and Disease Detection",
    page_icon="🌱",
    layout="centered"
)

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


st.title("🌱 Crop Pest and Disease Detection")
st.write(
    "Upload a crop leaf image to predict its disease or pest category."
)

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    st.error("Model files are missing. Please train the model first.")
    st.stop()

try:
    model, class_names = load_model_and_labels()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a crop image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded crop image",
            use_container_width=True
        )

        if st.button("Predict Disease", type="primary"):
            resized_image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(
                resized_image, dtype=np.float32
            )

            image_array = preprocess_input(image_array)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(
                image_array, verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_label = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.success(f"Predicted class: {predicted_label}")
            st.metric("Model confidence", f"{confidence:.2f}%")

            st.caption(
                "The confidence score is the model's output, "
                "not a guarantee of a correct diagnosis."
            )

            st.subheader("Top 3 predictions")
            top_indices = np.argsort(predictions)[-3:][::-1]

            for index in top_indices:
                st.write(
                    f"{class_names[index]}: "
                    f"{predictions[index] * 100:.2f}%"
                )

    except Exception as error:
        st.error(f"Could not process this image: {error}")