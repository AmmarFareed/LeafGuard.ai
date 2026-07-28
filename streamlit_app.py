"""
LeafGuard AI — Streamlit Demo App

Run locally with:
    streamlit run streamlit_app.py

Requires:
    leafguard_model.keras   (saved from the notebook, Section 9)
    class_names.json        (saved from the notebook, Section 9)
"""

import json

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image

st.set_page_config(page_title="LeafGuard AI", page_icon="🌿", layout="centered")

IMG_SIZE = (160, 160)


@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model("leafguard_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


def predict(image: Image.Image, model, class_names):
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(predictions))
    label = class_names[top_idx]
    confidence = float(predictions[top_idx]) * 100

    # Return top-3 for a bit more transparency
    top3_idx = np.argsort(predictions)[-3:][::-1]
    top3 = [(class_names[i], float(predictions[i]) * 100) for i in top3_idx]

    return label, confidence, top3


st.title("🌿 LeafGuard AI")
st.write(
    "Upload a photo of a plant leaf and LeafGuard AI will predict whether it's "
    "healthy or which disease it likely has."
)

try:
    model, class_names = load_model_and_classes()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(
        "Model files not found yet. Train the model in the notebook first, "
        "then place `leafguard_model.keras` and `class_names.json` in this folder."
    )

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if model_loaded:
        with st.spinner("Analyzing leaf..."):
            label, confidence, top3 = predict(image, model, class_names)

        st.subheader("Prediction")
        readable_label = label.replace("___", " — ").replace("_", " ")
        st.success(f"**{readable_label}**  ({confidence:.1f}% confidence)")

        st.subheader("Top 3 possibilities")
        for name, conf in top3:
            readable = name.replace("___", " — ").replace("_", " ")
            st.write(f"- {readable}: {conf:.1f}%")

        st.caption(
            "⚠️ This is a decision-support tool, not a substitute for expert diagnosis. "
            "Accuracy may be lower on real-world field photos compared to clean lab images."
        )
