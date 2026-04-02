import streamlit as st
st.set_page_config(layout="wide", page_title="Forensic Sketch Recognition")

import torch
import glob
import os
from PIL import Image
import numpy as np
from torchvision import transforms

from model import PseudoSiameseNet
from gradcam import (
    CosineSimilarityGradCAM,
    blend_heatmap,
    analyze_facial_features,
    generate_explanation,
    draw_feature_boxes
)

GALLERY_DB_PATH = 'gallery_db_display.pt'
MODEL_CHECKPOINT = 'checkpoints/regularized_v1/best_model.pth'
SKETCH_POOL_DIR = 'data/dataset/CUFS_reorganized/display/sketches'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model_and_gallery():
    model = PseudoSiameseNet().to(DEVICE)
    if os.path.exists(MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
    model.eval()

    vector_db = {}
    if os.path.exists(GALLERY_DB_PATH):
        vector_db = torch.load(GALLERY_DB_PATH, map_location=DEVICE)

    return model, vector_db

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def inv_transform(tensor):
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5
    img_np = tensor.permute(1, 2, 0).numpy()
    return img_np

st.sidebar.title("Sketch Query Selection")

sketch_files = glob.glob(os.path.join(SKETCH_POOL_DIR, '*.jpg')) + \
               glob.glob(os.path.join(SKETCH_POOL_DIR, '*.png'))
sketch_names = [os.path.basename(f) for f in sketch_files]

run_btn = False
actual_match_id = None
selected_sketch_name = None
sketch_img = None
sketch_path = None

if not sketch_names:
    st.sidebar.error("No display sketches found. Please run reorganize_dataset.py with display split.")
else:
    selected_sketch_name = st.sidebar.selectbox("Choose a forensic sketch to query:", sorted(sketch_names))
    sketch_path = os.path.join(SKETCH_POOL_DIR, selected_sketch_name)

    sketch_img = Image.open(sketch_path).convert('RGB')
    st.sidebar.image(sketch_img, caption="Query Sketch", width=200)

    base_name = selected_sketch_name.replace('_Sz', '').replace('-1', '')
    base_name = os.path.splitext(base_name)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        actual_match_id = base_name + ext
        break

    st.sidebar.markdown(f"**Expected match:** {base_name}")

    run_btn = st.sidebar.button("Run Matching")

st.title("Cross-Modal Explainable Forensic Matcher")
st.markdown("Searching database for matching photographic identities using a shared latent space and visual explanations (Grad-CAM).")

st.markdown("""
**Split Configuration:**
- Training: 60% of data
- Testing: 30% of data  
- Display (UI): 10% of data
""")

try:
    st.markdown("### Initializing...")
    model, vector_db = load_model_and_gallery()
    st.success(f"Model loaded. Gallery size: {len(vector_db)}.")
except Exception as e:
    st.error(f"Failed to load model/gallery: {e}")
    st.exception(e)
    st.stop()

if sketch_names and run_btn and vector_db and sketch_img is not None:
    st.write(f"### Querying {len(vector_db)} gallery records for **{selected_sketch_name}**...")

    sketch_tensor = transform(sketch_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sketch_embedding = model.forward_sketch(sketch_tensor)
        sketch_embedding = torch.nn.functional.normalize(sketch_embedding, p=2, dim=1)

    results = []
    for photo_id, data in vector_db.items():
        photo_emb = data['embedding'].to(DEVICE)
        score = torch.sum(sketch_embedding * photo_emb).item()
        results.append({
            'photo_id': photo_id,
            'score': score,
            'filepath': data['filepath']
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    actual_match_rank = None
    for idx, res in enumerate(results):
        photo_base = os.path.splitext(res['photo_id'])[0]
        sketch_base = os.path.splitext(actual_match_id)[0] if actual_match_id else ""
        if photo_base == sketch_base:
            actual_match_rank = idx + 1
            break

    top10 = results[:10]

    st.write("## Top 10 Matches")

    if actual_match_rank is not None:
        if actual_match_rank <= 10:
            st.success(f"Correct match found at Rank {actual_match_rank}")
        else:
            st.warning(f"Correct match is at Rank {actual_match_rank} (not in top 10)")

        actual_match_data = None
        for res in results:
            photo_base = os.path.splitext(res['photo_id'])[0]
            sketch_base = os.path.splitext(actual_match_id)[0] if actual_match_id else ""
            if photo_base == sketch_base:
                actual_match_data = res
                break

        if actual_match_data:
            st.write("### Actual Match (Not in Top 10)")
            actual_img = Image.open(actual_match_data['filepath']).convert('RGB')
            st.image(actual_img, caption=f"Rank {actual_match_rank}: {actual_match_data['photo_id']} (Score: {actual_match_data['score']:.4f})", width=200)
    else:
        st.error("Correct match not found in gallery")

    st.markdown("""
    <style>
    .correct-match-box {
        border: 4px solid #00FF00;
        padding: 10px;
        background-color: rgba(144, 238, 144, 0.2);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    target_layer = model.branch_photo.block8
    cam_explainer = CosineSimilarityGradCAM(model, target_layer)

    cols = st.columns(5)

    for i, res in enumerate(top10):
        col_idx = i % 5
        photo_base = os.path.splitext(res['photo_id'])[0]
        sketch_base = os.path.splitext(actual_match_id)[0] if actual_match_id else ""
        is_correct = (photo_base == sketch_base)

        with cols[col_idx]:
            if is_correct:
                st.markdown('<div class="correct-match-box">', unsafe_allow_html=True)
                st.markdown(f"**Rank {i+1}** :white_check_mark: **CORRECT MATCH**")
            else:
                st.markdown(f"**Rank {i+1}**")

            st.caption(f"Score: {res['score']:.4f}")
            st.caption(f"ID: {res['photo_id']}")

            candidate_img = Image.open(res['filepath']).convert('RGB')
            candidate_tensor = transform(candidate_img).unsqueeze(0).to(DEVICE)

            cam_np, _ = cam_explainer(candidate_tensor, sketch_embedding)

            feature_scores = analyze_facial_features(cam_np)
            explanation = generate_explanation(feature_scores)

            orig_img_np = inv_transform(candidate_tensor.squeeze())
            blended_img = blend_heatmap(orig_img_np, cam_np)

            img_with_boxes = draw_feature_boxes(orig_img_np, cam_np)

            st.image(blended_img, caption="Grad-CAM Heatmap", width=200)
            st.image(img_with_boxes, caption="Feature Regions", width=200)

            with st.expander("Why this match?"):
                st.write(explanation)
                if feature_scores:
                    st.write("**Feature contributions:**")
                    for feature, score in feature_scores.items():
                        st.write(f" - {feature.replace('_', ' ').title()}: {score:.3f}")

            st.image(candidate_img, caption="Original Photo", width=200)

            if is_correct:
                st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

elif sketch_names and run_btn and not vector_db:
    st.error("Gallery database is empty or missing. Please run `python gallery.py --split display` first.")

st.markdown("### Instructions")
st.markdown("1. Run `python reorganize_dataset.py` to create train/test/display splits (60/30/10)")
st.markdown("2. Run `python gallery.py --split display` to generate `gallery_db_display.pt`")
st.markdown("3. Run `python train.py` to train the model.")
st.markdown("4. Restart the Streamlit server: `streamlit run app.py`")
