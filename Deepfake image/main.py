import streamlit as st
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# ---------- Config ----------
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

ID2LABEL = {
    "0": "fake",
    "1": "real"
}

# ---------- Model Loading (cached) ----------
@st.cache_resource
def load_model_and_processor():
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, processor, device

model, processor, device = load_model_and_processor()

# ---------- Prediction Function ----------
def classify_image_pil(image: Image.Image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        ID2LABEL[str(i)]: round(float(probs[i]), 3) for i in range(len(probs))
    }
    return prediction

# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Deepfake Detector")
st.markdown(
    """
Upload an image to classify whether it is **real** or **fake**  
(using `deepfake-detector-model-v1`).
"""
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    # Show image preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        with st.spinner("Running deepfake detector..."):
            prediction = classify_image_pil(image)

        st.subheader("Results")
        # Sort labels by probability, descending
        sorted_items = sorted(prediction.items(), key=lambda x: x[1], reverse=True)

        # Highlight top result
        top_label, top_prob = sorted_items[0]
        st.markdown(
            f"**Prediction:** `{top_label.upper()}` with probability **{top_prob:.3f}**"
        )

        st.markdown("**Class probabilities:**")
        for label, prob in sorted_items:
            st.write(f"- **{label}**: {prob:.3f}")

        st.info(
            "⚠️ This model may not be 100% accurate. "
            "Use it as an aid, not as a definitive decision-maker."
        )
else:
    st.info("👆 Upload an image file to get started.")
