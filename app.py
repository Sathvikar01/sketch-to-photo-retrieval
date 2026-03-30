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

# Configs
GALLERY_DB_PATH = 'gallery_db_reorganized.pt'
MODEL_CHECKPOINT = 'checkpoints/regularized_v1/best_model.pth'
SKETCH_POOL_DIR = 'data/dataset/CUFS_reorganized/test/sketches'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cached Data Loaders
@st.cache_resource
def load_model_and_gallery():
    # Load tuned model or pretrained base
    model = PseudoSiameseNet().to(DEVICE)
    if os.path.exists(MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
    model.eval()
    
    # Load gallery DB
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
    # Convert normalized [-1, 1] tensor back to [0, 1] numpy array for display
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5
    img_np = tensor.permute(1, 2, 0).numpy()
    return img_np

# Sidebar - Sketch Selection
st.sidebar.title("Sketch Query Selection")

sketch_files = glob.glob(os.path.join(SKETCH_POOL_DIR, '*.jpg'))
sketch_names = [os.path.basename(f) for f in sketch_files]

run_btn = False
actual_match_id = None
selected_sketch_name = None
sketch_img = None
sketch_path = None

if not sketch_names:
    st.sidebar.error("No test sketches found. Please check data path.")
else:
    selected_sketch_name = st.sidebar.selectbox("Choose a forensic sketch to query:", sketch_names)
    sketch_path = os.path.join(SKETCH_POOL_DIR, selected_sketch_name)
    
    sketch_img = Image.open(sketch_path).convert('RGB')
    st.sidebar.image(sketch_img, caption="Query Sketch", width=200)
    
    base_name = selected_sketch_name.replace('_Sz', '').replace('-1', '')
    actual_match_id = base_name
    st.sidebar.markdown(f"**Expected match:** {actual_match_id}")
    
    run_btn = st.sidebar.button("Run Matching")

# Main Container
st.title("Cross-Modal Explainable Forensic Matcher")
st.markdown("Searching database for matching photographic identities using a shared latent space and visual explanations (Grad-CAM).")

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
    
    # 1. Forward Pass on Sketch
    sketch_tensor = transform(sketch_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sketch_embedding = model.forward_sketch(sketch_tensor)
        sketch_embedding = torch.nn.functional.normalize(sketch_embedding, p=2, dim=1)
        
    # 2. Iterate through Vector Database & Score
    results = []
    for photo_id, data in vector_db.items():
        photo_emb = data['embedding'].to(DEVICE)
        # Cosine similarity
        score = torch.sum(sketch_embedding * photo_emb).item()
        results.append({
            'photo_id': photo_id,
            'score': score,
            'filepath': data['filepath']
        })
        
# 3. Sort Descending & Get Top 10 and find actual match rank
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    actual_match_rank = None
    for idx, res in enumerate(results):
        if res['photo_id'] == actual_match_id:
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
                if res['photo_id'] == actual_match_id:
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
    
    # 4. Prepare Custom Grad-CAM
    # Target the last conv layer in the InceptionResnetV1 block before pooling
    target_layer = model.branch_photo.block8
    cam_explainer = CosineSimilarityGradCAM(model, target_layer)
    
    # Create 2x5 Grid for Top 10
    cols = st.columns(5)
    
    for i, res in enumerate(top10):
        col_idx = i % 5
        is_correct = (res['photo_id'] == actual_match_id)
        
        with cols[col_idx]:
            if is_correct:
                st.markdown('<div class="correct-match-box">', unsafe_allow_html=True)
                st.markdown(f"**Rank {i+1}** :white_check_mark: **CORRECT MATCH**")
            else:
                st.markdown(f"**Rank {i+1}**")
            
            st.caption(f"Score: {res['score']:.4f}")
            st.caption(f"ID: {res['photo_id']}")
            
            # Load candidate photo
            candidate_img = Image.open(res['filepath']).convert('RGB')
            candidate_tensor = transform(candidate_img).unsqueeze(0).to(DEVICE)
            
            # Compute Grad-CAM on the fly
            cam_np, _ = cam_explainer(candidate_tensor, sketch_embedding)
            
            # Analyze facial features
            feature_scores = analyze_facial_features(cam_np)
            explanation = generate_explanation(feature_scores)
            
            # Blend
            orig_img_np = inv_transform(candidate_tensor.squeeze())
            blended_img = blend_heatmap(orig_img_np, cam_np)
            
            # Draw feature boxes
            img_with_boxes = draw_feature_boxes(orig_img_np, cam_np)
            
            # Stack visualization
            st.image(blended_img, caption="Grad-CAM Heatmap", width=200)
            st.image(img_with_boxes, caption="Feature Regions", width=200)
            
            with st.expander("Why this match?"):
                st.write(explanation)
                if feature_scores:
                    st.write("**Feature contributions:**")
                    for feature, score in feature_scores.items():
                        st.write(f"  - {feature.replace('_', ' ').title()}: {score:.3f}")
            
            st.image(candidate_img, caption="Original Photo", width=200)
            
            if is_correct:
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
elif sketch_names and run_btn and not vector_db:
    st.error("Gallery database is empty or missing. Please run `python gallery.py` first.")

st.markdown("### Instructions")
st.markdown("1. Run `python gallery.py` to generate `gallery_db.pt`. ")
st.markdown("2. Run `python train.py` to train the model.")
st.markdown("3. Restart the Streamlit server: `streamlit run app.py`")
