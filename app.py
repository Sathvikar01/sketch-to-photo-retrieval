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

DATASET_CONFIGS = {
    'CUFS': {
        'gallery_dbs': ['gallery_db_display.pt'],
        'sketch_dir': 'data/dataset/CUFS_reorganized/display/sketches',
        'info': 'CUFS Dataset: 25 test pairs for display'
    },
    'Color FERET': {
        'gallery_dbs': ['gallery_db_colorferet_display.pt'],
        'sketch_dir': 'data/dataset/colorferet/display/sketches',
        'info': 'Color FERET: 68 persons for display (60/30/10 split)'
    }
}
MODEL_CHECKPOINT = 'checkpoints/regularized_v1/best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    model = PseudoSiameseNet().to(DEVICE)
    if os.path.exists(MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE), strict=False)
        model.eval()
    return model

@st.cache_resource
def load_gallery(db_paths):
    vector_db = {}
    for db_path in db_paths:
        if os.path.exists(db_path):
            db = torch.load(db_path, map_location=DEVICE)
            vector_db.update(db)
    return vector_db

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

dataset_choice = st.sidebar.selectbox("Select Dataset:", list(DATASET_CONFIGS.keys()))
config = DATASET_CONFIGS[dataset_choice]

st.sidebar.caption(config['info'])

SKETCH_POOL_DIR = config['sketch_dir']

sketch_files = glob.glob(os.path.join(SKETCH_POOL_DIR, '*.jpg')) + \
    glob.glob(os.path.join(SKETCH_POOL_DIR, '*.png'))
sketch_names = [os.path.basename(f) for f in sketch_files]

run_btn = False
actual_match_id = None
selected_sketch_name = None
sketch_img = None
sketch_path = None

if not sketch_names:
    st.sidebar.error(f"No display sketches found for {dataset_choice}.")
else:
    selected_sketch_name = st.sidebar.selectbox("Choose a forensic sketch to query:", sorted(sketch_names))
    sketch_path = os.path.join(SKETCH_POOL_DIR, selected_sketch_name)

    sketch_img = Image.open(sketch_path).convert('RGB')

    base_name = selected_sketch_name.replace('_Sz', '').replace('-1', '')
    base_name = os.path.splitext(base_name)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        actual_match_id = base_name + ext
        break

    st.sidebar.markdown(f"**Expected match:** {base_name}")

    run_btn = st.sidebar.button("Run Matching")

st.title("Cross-Modal Explainable Forensic Matcher")
st.markdown("Searching database for matching photographic identities using a shared latent space and visual explanations (Grad-CAM).")

st.markdown(f"""
**Dataset:** {dataset_choice}
{config['info']}
""")

try:
    st.markdown("### Initializing...")
    model = load_model()
    vector_db = load_gallery(config['gallery_dbs'])
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
            st.image(actual_img, caption=f"Rank {actual_match_rank}: {actual_match_data['photo_id']} (Score: {actual_match_data['score']:.4f})", width="stretch")
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

    for i, res in enumerate(top10):
        photo_base = os.path.splitext(res['photo_id'])[0]
        sketch_base = os.path.splitext(actual_match_id)[0] if actual_match_id else ""
        is_correct = (photo_base == sketch_base)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"**Rank {i+1}**")
            if is_correct:
                st.success("CORRECT MATCH")
            st.caption(f"Score: {res['score']:.4f}")
            st.caption(f"ID: {res['photo_id']}")
            st.image(sketch_img, caption="Query Sketch", width=200)

        with col2:
            candidate_img = Image.open(res['filepath']).convert('RGB')
            candidate_tensor = transform(candidate_img).unsqueeze(0).to(DEVICE)

            cam_np, _ = cam_explainer(candidate_tensor, sketch_embedding)

            feature_scores = analyze_facial_features(cam_np)
            explanation = generate_explanation(feature_scores)

            orig_img_np = inv_transform(candidate_tensor.squeeze())
            blended_img = blend_heatmap(orig_img_np, cam_np)

            img_with_boxes = draw_feature_boxes(orig_img_np, cam_np)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(candidate_img, caption="Matched Photo", width=200)
            with c2:
                st.image(blended_img, caption="Grad-CAM Heatmap", width=200)
            with c3:
                st.image(img_with_boxes, caption="Feature Regions", width=200)

            with st.expander("Why this match?"):
                st.write(explanation)
                if feature_scores:
                    st.write("**Feature contributions:**")
                    for feature, score in feature_scores.items():
                        st.write(f" - {feature.replace('_', ' ').title()}: {score:.3f}")

        st.divider()

elif sketch_names and run_btn and not vector_db:
    st.error(f"Gallery database is empty or missing for {dataset_choice}.")

st.markdown("### Instructions")
st.markdown("1. Run `python reorganize_dataset.py` for CUFS or `python prepare_colorferet.py` for Color FERET")
st.markdown("2. Run `python gallery.py` to generate gallery databases")
st.markdown("3. Run `python train.py` or `python train_colorferet.py` to train the model.")
st.markdown("4. Restart the Streamlit server: `streamlit run app.py`")
