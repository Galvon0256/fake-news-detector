import streamlit as st
import joblib
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# ---------- Page Config ----------
st.set_page_config(
    page_title="Fake News & Deepfake Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News & 🕵️ Deepfake Detector")
st.write("Analyze **news articles** and **images** to check if they are real or fake.")

# ---------- Text Fake News Model (cached) ----------
@st.cache_resource
def load_text_models():
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    return vectorizer, model

# ---------- Image Deepfake Model (cached) ----------
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

ID2LABEL = {
    "0": "fake",
    "1": "real"
}

@st.cache_resource
def load_image_model_and_processor():
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, processor, device

def classify_image_pil(image: Image.Image, model, processor, device):
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

# ---------- Tabs ----------
tab_text, tab_image = st.tabs(["📰 Text Fake News Detector", "🕵️ Image Deepfake Detector"])

# ===== Tab 1: Text Fake News Detector =====
with tab_text:
    st.header("📰 Fake News Detector (Text)")
    st.write("Enter a news article below to check whether it is **Fake** or **Real**.")

    inputn = st.text_area("News Article:", "")

    if st.button("Check News"):
        if inputn.strip():
            vectorizer, text_model = load_text_models()
            transform_input = vectorizer.transform([inputn])
            prediction = text_model.predict(transform_input)

            if prediction[0] == 1:
                st.success("✅ The News is Real!")
            else:
                st.error("❌ The News is Fake!")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# ===== Tab 2: Image Deepfake Detector =====
with tab_image:
    st.header("🕵️ Deepfake Detector (Image)")
    st.markdown(
        """
        Upload an image to classify whether it is **real** or **fake**  
        (using `deepfake-detector-model-v1`).
        """
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_uploader"
    )

    if uploaded_file is not None:
        # Show image preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Classify Image"):
            model, processor, device = load_image_model_and_processor()

            with st.spinner("Running deepfake detector..."):
                prediction = classify_image_pil(image, model, processor, device)

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
