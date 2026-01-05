import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

from hybrid_model import ParallelHybridCNNViT

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

    model = ParallelHybridCNNViT(num_classes=6)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()
    return model

model = load_model()


# =========================
# UI
# =========================
st.title("Diagnosis Penyakit Mulut Berbasis Citra")
st.write("Upload citra untuk langsung mendapatkan hasil diagnosis.")

uploaded_file = st.file_uploader(
    "Upload gambar (jpg / png)",
    type=["jpg", "png", "jpeg"]
)

# =========================
# AUTO RUN INFERENCE
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Citra Input", use_container_width=True)

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

    st.success(f"Prediksi: **{CLASS_NAMES[pred_idx]}**")
    st.write(f"Confidence: **{probs[pred_idx]*100:.2f}%**")

    st.subheader("Probabilitas Kelas")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {probs[i]*100:.2f}%")
