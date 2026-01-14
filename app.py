import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

from hybrid_model import ParallelHybridCNNViT
st.set_page_config(
    page_title="Diagnosis Penyakit Mulut",
    page_icon="🦷",
    layout="centered"
)

# =========================
# CONFIG
# =========================
NUM_CLASSES = 6

CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth Ulcer",
    "Tooth Discoloration"
]

DISEASE_INFO = {
    "Calculus": (
        "Kalkulus gigi adalah plak yang mengeras akibat penumpukan mineral "
        "pada permukaan gigi. Kondisi ini dapat menyebabkan iritasi gusi "
        "dan meningkatkan risiko penyakit periodontal."
    ),
    "Caries": (
        "Karies gigi merupakan kerusakan jaringan keras gigi yang disebabkan "
        "oleh aktivitas bakteri. Kondisi ini ditandai dengan lubang pada gigi "
        "dan dapat menimbulkan nyeri jika tidak ditangani."
    ),
    "Gingivitis": (
        "Gingivitis adalah peradangan pada gusi yang umumnya disebabkan oleh "
        "penumpukan plak. Gejalanya meliputi gusi kemerahan, bengkak, dan mudah berdarah."
    ),
    "Hypodontia": (
        "Hipodonsia adalah kondisi bawaan di mana satu atau lebih gigi permanen "
        "tidak tumbuh. Kondisi ini dapat memengaruhi fungsi pengunyahan dan estetika."
    ),
    "Mouth Ulcer": (
        "Ulkus mulut merupakan luka terbuka pada jaringan lunak di dalam mulut "
        "yang dapat menimbulkan rasa nyeri, terutama saat makan atau berbicara."
    ),
    "Tooth Discoloration": (
        "Perubahan warna gigi dapat disebabkan oleh faktor ekstrinsik seperti makanan "
        "dan minuman, maupun faktor intrinsik seperti gangguan struktur gigi."
    ),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="juvencia/oral_hybrid_model",
        filename="best_hybrid_model.pth"
    )

    model = ParallelHybridCNNViT(num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()
    return model

model = load_model()

# =========================
# UI
# =========================

st.title("🦷 Diagnosis Penyakit Mulut Berbasis Citra")
st.write(
    "Unggah citra gigi atau mulut untuk mendapatkan hasil diagnosis "
    "berdasarkan model *Hybrid EfficientNet–Vision Transformer*."
)

uploaded_file = st.file_uploader(
    "Upload gambar (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)

# =========================
# AUTO RUN INFERENCE
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Citra Input", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    # =========================
    # OUTPUT
    # =========================
    st.success(f"Prediksi Penyakit: **{pred_class}**")
    st.write(f"Confidence Model: **{confidence:.2f}%**")

    st.subheader("📌 Penjelasan Penyakit")
    st.write(DISEASE_INFO[pred_class])

    st.info(
        "⚠️ Informasi ini bersifat edukatif dan tidak menggantikan diagnosis medis. "
        "Untuk pemeriksaan dan penanganan lebih lanjut, disarankan berkonsultasi "
        "dengan dokter gigi atau tenaga kesehatan."
    )

    st.subheader("📊 Probabilitas Setiap Kelas")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"- **{cls}**: {probs[i]*100:.2f}%")
