# LC3-GradCAM: Landmark-Constrained Contrastive Explanations for Cross-Modal Face Retrieval

A novel explainable AI framework for forensic sketch-to-photo face matching with dual-branch, contrastive attribution maps.

## 🎯 Project Overview

This project implements **LC3-GradCAM** (Landmark-Constrained Contrastive Cross-Modal Grad-CAM), a novel explanation method designed specifically for cross-modal sketch-to-photo face retrieval. Unlike traditional Grad-CAM approaches, LC3-GradCAM:

1. **Generates attribution maps on BOTH branches** - shows which sketch strokes correspond to which photo regions
2. **Provides contrastive explanations** - positive (why match) and negative (why not match)
3. **Uses dynamic landmark detection** - scores regions based on actual facial landmarks, not fixed coordinates

## 📊 Results

### CUFS Dataset (60/30/10 Split)

| Split | Size | Purpose |
|-------|------|---------|
| Training | 363 pairs (60%) | Model training |
| Testing | 181 pairs (30%) | Evaluation |
| Display | 62 pairs (10%) | UI demonstration |

**Performance on Test Set (30% holdout):**

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Recall@1** | 81.07% | [75.73%, 85.92%] |
| **Recall@5** | 95.15% | [92.23%, 97.58%] |
| **Recall@10** | 98.54% | - |
| **MRR** | 0.8791 | [84.19%, 91.30%] |
| **Mean Rank** | 1.59 ± 1.89 | - |
| **Median Rank** | 1 | - |

### Comparison with Previous Results

| Metric | Previous (25 test pairs) | New (181 test pairs) | Improvement |
|--------|-------------------------|---------------------|-------------|
| Recall@1 | 49.70% | 81.07% | +31.37% |
| Recall@5 | 81.95% | 95.15% | +13.20% |
| MRR | 0.6398 | 0.8791 | +0.2393 |

*Note: Previous results used a small 25-pair test set which may not have been representative. The new 30% holdout test set provides more reliable evaluation.*

## 🗂️ Dataset Structure

### CUFS (CUHK Face Sketch)
```
data/dataset/CUFS_reorganized/
├── train/           (363 pairs - 60%)
│   ├── photos/
│   └── sketches/
├── test/            (181 pairs - 30%)
│   ├── photos/
│   └── sketches/
└── display/         (62 pairs - 10%)
    ├── photos/
    └── sketches/
```

### CUFSF (CUHK Face Sketch & Face - Extended)
CUFSF extends CUFS with multiple photo variations per identity, testing robustness to lighting, expression, and pose changes.

```
data/dataset/CUFSF_reorganized/
├── train/           (60%)
│   ├── photos/
│   │   └── person_id/
│   │       ├── photo_1.jpg
│   │       ├── photo_2.jpg  (variation)
│   │       └── ...
│   └── sketches/
├── test/            (30%)
└── display/         (10%)
```

**Note:** CUFSF requires download from CUHK. Run:
```bash
python download_cufsf.py
```

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

### 5. Proper Data Splitting (NEW)
- **60% Training** - Model development
- **30% Testing** - Final evaluation (unseen during training)
- **10% Display** - UI demonstration (separate from test)

## 📁 Project Structure

```
xaip/
├── model.py              # Pseudo-Siamese network architecture
├── dataset.py            # CUFS/CUFSF dataset loader
├── train.py              # Training script with all enhancements
├── gradcam.py            # LC3-GradCAM implementation
├── evaluation_metrics.py # Retrieval metrics + CMC + bootstrap CI
├── gallery.py            # Gallery database generation (supports splits)
├── app.py                # Streamlit demo interface (uses display split)
├── reorganize_dataset.py # Dataset splitting (60/30/10)
├── download_cufsf.py     # CUFSF download helper
├── run_ablations.py      # Ablation experiment runner
├── paper_draft.md        # Research paper draft
└── data/
    └── dataset/
        ├── CUFS/
        ├── CUFS_reorganized/
        └── CUFSF/         (requires download)
```

## 🔧 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Quick Start

### 1. Reorganize Dataset (60/30/10 split)

```bash
# CUFS dataset
python reorganize_dataset.py --dataset CUFS

# CUFSF dataset (after download)
python reorganize_dataset.py --dataset CUFSF
```

### 2. Generate Gallery Databases

```bash
# Generate for all splits
python gallery.py --all

# Or specific splits
python gallery.py --split display
python gallery.py --split test
```

### 3. Train Model

```bash
# Baseline training
python train.py --experiment_name baseline --epochs 10

# Full enhanced training
python train.py --experiment_name full_stack --epochs 50 --use_amp
```

### 4. Evaluate

```bash
python evaluation_metrics.py
```

### 5. Run Streamlit Demo

```bash
streamlit run app.py
```

The demo uses the **display split (10%)** for UI demonstrations, keeping test data separate for proper evaluation.

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
- Results Tables
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

## 🔄 Data Split Rationale

### Why 60/30/10?

| Split | Purpose | Why This Size? |
|-------|---------|----------------|
| **Train (60%)** | Model learning | Sufficient data for training |
| **Test (30%)** | Evaluation | Large enough for reliable metrics |
| **Display (10%)** | UI demo | Separate from test to avoid leakage |

### Previous vs New Split

| Aspect | Previous | New |
|--------|----------|-----|
| Test size | 25 pairs (fixed) | 181 pairs (30%) |
| Display set | None (test used) | 62 pairs (10%) |
| Evaluation reliability | Low (small sample) | High (large sample) |
| Data leakage risk | High (test in UI) | None (separated) |

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
