# LC3-GradCAM: Landmark-Constrained Contrastive Explanations for Cross-Modal Face Retrieval

A novel explainable AI framework for forensic sketch-to-photo face matching with dual-branch, contrastive attribution maps.

## 🎯 Project Overview

This project implements **LC3-GradCAM** (Landmark-Constrained Contrastive Cross-Modal Grad-CAM), a novel explanation method designed specifically for cross-modal sketch-to-photo face retrieval. Unlike traditional Grad-CAM approaches, LC3-GradCAM:

1. **Generates attribution maps on BOTH branches** - shows which sketch strokes correspond to which photo regions
2. **Provides contrastive explanations** - positive (why match) and negative (why not match)
3. **Uses dynamic landmark detection** - scores regions based on actual facial landmarks, not fixed coordinates

## 📊 Baseline Results (CUFS Dataset)

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Recall@1** | 49.70% | [44.38%, 55.03%] |
| **Recall@5** | 81.95% | [77.81%, 86.39%] |
| **Recall@10** | 87.57% | - |
| **MRR** | 0.6398 | [0.5964, 0.6807] |
| **Mean Rank** | 4.78 ± 8.74 | - |

## 🚀 Key Improvements Implemented

### 1. Enhanced Data Augmentation
- Modality-aware transforms (stronger on photo branch)
- Color jitter + random grayscale to force structural learning
- Horizontal flipping for both branches

### 2. Batch-Hard Negative Mining
- Online selection of hardest negatives within each batch
- Combined Triplet + Contrastive loss for better gradients

### 3. Learning Rate Scheduling
- Cosine annealing with warm restarts
- 5-epoch warmup for stability
- Extended training (50 epochs vs. original 10)

### 4. LC3-GradCAM Explainability
- `compute_positive_attribution()` - why a photo matches a sketch
- `compute_negative_attribution()` - why a photo doesn't match
- `compute_dual_attribution()` - cross-modal correspondence
- `analyze_landmark_regions()` - dynamic landmark-based scoring

## 📁 Project Structure

```
xaip/
├── model.py              # Pseudo-Siamese network architecture
├── dataset.py            # CUFS dataset loader with augmentations
├── train.py              # Training script with all enhancements
├── gradcam.py            # LC3-GradCAM implementation
├── evaluation_metrics.py # Retrieval metrics + CMC + bootstrap CI
├── gallery.py            # Gallery database generation
├── app.py                # Streamlit demo interface
├── run_ablations.py      # Ablation experiment runner
├── paper_draft.md        # Research paper draft
└── data/
    └── dataset/
        └── CUFS/
            ├── train_sketches/
            ├── train_photos/
            ├── test_sketches/
            └── test_photos/
```

## 🔧 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Quick Start

### 1. Generate Gallery Database

```bash
python gallery.py
```

### 2. Train Model

```bash
# Baseline training
python train.py --experiment_name baseline --epochs 10

# Full enhanced training
python train.py --experiment_name full_stack --epochs 50 --use_amp
```

### 3. Evaluate

```bash
python evaluation_metrics.py
```

### 4. Run Streamlit Demo

```bash
streamlit run app.py
```

## 🧪 Running Ablation Experiments

The ablation study validates each improvement:

```bash
python run_ablations.py
```

This runs:
1. **Baseline** - Original training setup
2. **+Aug** - With modality-aware augmentations
3. **+Mining** - With batch-hard negative mining
4. **+Scheduler** - With cosine annealing
5. **Full Stack** - All improvements combined

Results are saved to `ablation_results_TIMESTAMP/` with:
- JSON results file
- LaTeX ablation table
- CMC curves
- Rank distributions

## 📖 LC3-GradCAM API

```python
from gradcam import LC3GradCAM

# Initialize
explainer = LC3GradCAM(model, target_layer_photo, target_layer_sketch)

# Positive attribution (why match)
photo_cam, similarity = explainer(sketch_tensor, photo_tensor, mode='positive')

# Negative attribution (why not match)
neg_cam, sim = explainer(sketch_tensor, photo_tensor, mode='negative')

# Dual-branch attribution
photo_cam, sketch_cam, sim = explainer(sketch_tensor, photo_tensor, mode='dual')

# Landmark-constrained region scoring
region_scores = explainer.analyze_landmark_regions(cam_np, landmarks)
```

## 📝 Research Paper

See `paper_draft.md` for the complete manuscript draft including:
- Abstract and Introduction
- Related Work
- Methodology (architecture, training, LC3-GradCAM)
- Experimental Setup
- Results Tables (to be filled after ablations)
- Discussion and Limitations
- References

## 📈 Evaluation Metrics

The evaluation pipeline computes:

1. **Recall@K** - Proportion of queries where correct match is in top-K
2. **MRR** - Mean Reciprocal Rank
3. **CMC Curve** - Cumulative Matching Characteristic
4. **Bootstrap 95% CI** - Confidence intervals via resampling

Visualizations are automatically generated:
- `cmc_curve.png` - Recognition rate vs. rank
- `rank_distribution.png` - Histogram of correct match ranks

## 🤝 Citation

If you use this code, please cite:

```bibtex
@article{lc3gradcam2026,
  title={LC3-GradCAM: Landmark-Constrained Contrastive Explanations for Cross-Modal Sketch-to-Photo Face Retrieval},
  author={Your Name},
  journal={Conference/Journal},
  year={2026}
}
```

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- InceptionResnetV1 from [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- CUFS Dataset from CUHK
- Grad-CAM original paper by Selvaraju et al.
