"""Standalone Streamlit app outside `HAT-Deep_Learning`.

This demo runs multimodal few-shot inference with:
- CLIP ViT-B/16 encoder
- Linear head checkpoint: best_few_shot_10shot.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import clip
import gdown
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOCAL_CHECKPOINT = BASE_DIR / "best_few_shot_10shot.pth"
DEFAULT_GDRIVE_URL = "https://drive.google.com/file/d/1MPqx6Inl_85N03W2I5R5k1ZZ4G-mph8z/view?usp=sharing"

CLASS_NAMES = [
    "donuts",
    "french_fries",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "pho",
    "pizza",
    "steak",
    "sushi",
    "tacos",
]


class FewShotClassifier(nn.Module):
    def __init__(self, clip_model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.clip_model = clip_model

        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        feature_dim = clip_model.visual.output_dim
        self.classifier = nn.Linear(feature_dim * 2, num_classes)

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            text_tokens = clip.tokenize([str(text) for text in texts], truncate=True).to(images.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            fused = torch.cat([image_features, text_features], dim=1)
            fused = fused / fused.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return 10.0 * self.classifier(fused.float())


def resolve_checkpoint(local_path: Path, gdrive_url: str) -> Path:
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url=gdrive_url, output=str(local_path), fuzzy=True, quiet=False)

    if not local_path.exists():
        raise FileNotFoundError("Failed to download checkpoint from Google Drive.")
    return local_path


@st.cache_resource(show_spinner="Loading CLIP and few-shot checkpoint...")
def load_bundle(checkpoint_path: str) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    classifier = FewShotClassifier(clip_model, num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    return {
        "device": device,
        "preprocess": preprocess,
        "classifier": classifier,
    }


def predict_topk(bundle: dict[str, Any], image: Image.Image, caption: str, top_k: int) -> list[dict[str, Any]]:
    image_tensor = bundle["preprocess"](image).unsqueeze(0).to(bundle["device"])
    safe_caption = caption.strip() or "a photo of food"

    with torch.no_grad():
        logits = bundle["classifier"](image_tensor, [safe_caption])
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(-probs)[: max(1, min(top_k, len(CLASS_NAMES)))]
    return [
        {
            "rank": rank + 1,
            "label": CLASS_NAMES[int(index)],
            "probability": float(probs[int(index)]),
        }
        for rank, index in enumerate(top_indices)
    ]


def run_app() -> None:
    st.title("Multimodal Few-shot Demo")

    st.sidebar.header("Checkpoint")
    ckpt_path_text = st.sidebar.text_input("Local checkpoint path", value=str(DEFAULT_LOCAL_CHECKPOINT))
    gdrive_url = st.sidebar.text_input("Google Drive URL", value=DEFAULT_GDRIVE_URL)
    st.sidebar.markdown(f"[Open checkpoint link]({gdrive_url})")

    try:
        checkpoint_path = resolve_checkpoint(Path(ckpt_path_text), gdrive_url)
        bundle = load_bundle(str(checkpoint_path))
        st.success(f"Loaded checkpoint: {checkpoint_path}")
        st.sidebar.info(f"Device: {bundle['device']}")
    except Exception as exc:
        st.error(f"Cannot load model/checkpoint: {exc}")
        st.stop()

    left, right = st.columns([0.55, 0.45], gap="large")

    with left:
        uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png", "webp"])
        caption = st.text_area("Caption/Text", value="a delicious plate of sushi", height=120)
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
        run_button = st.button("Predict", type="primary")

    if uploaded_file is None:
        st.info("Please upload an image to get started.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    with right:
        st.image(image, caption="Input image", width="stretch")

    if not run_button:
        return

    predictions = predict_topk(bundle, image, caption, top_k)
    best = predictions[0]

    c1, c2 = st.columns(2)
    c1.metric("Predicted class", best["label"])
    c2.metric("Top-1 confidence", f"{best['probability']:.2%}")

    df = pd.DataFrame(predictions)
    df["probability"] = df["probability"].map(lambda value: f"{value:.2%}")
    st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    run_app()
