import torch
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from model import PseudoSiameseNet
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

DATA_DIR = 'data/dataset/CUFS'
GALLERY_DB_PATH = 'gallery_db.pt'
MODEL_CHECKPOINT = 'checkpoints/pseudo_siamese.pth'
RESULTS_PATH = 'evaluation_results.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_test_pairs(data_dir: str = DATA_DIR) -> List[Tuple[str, str, str]]:
    """
    Load test sketch-photo pairs from CUFS dataset.
    
    Returns:
        List of tuples: (sketch_path, photo_path, photo_id)
    """
    test_sketches_dir = os.path.join(data_dir, 'test_sketches')
    test_photos_dir = os.path.join(data_dir, 'test_photos')
    
    sketch_files = glob.glob(os.path.join(test_sketches_dir, '*'))
    sketch_files = [f for f in sketch_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    pairs = []
    for sketch_path in sketch_files:
        sketch_name = os.path.basename(sketch_path)
        base = os.path.splitext(sketch_name)[0]
        
        for suffix in ['_Sz', '-1', '_sketch']:
            base = base.replace(suffix, '')
        
        photo_found = False
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            photo_path = os.path.join(test_photos_dir, base + ext)
            if os.path.exists(photo_path):
                pairs.append((sketch_path, photo_path, base + ext))
                photo_found = True
                break
        
        if not photo_found:
            direct_path = os.path.join(test_photos_dir, sketch_name)
            if os.path.exists(direct_path):
                pairs.append((sketch_path, direct_path, sketch_name))
    
    return pairs


def calculate_metrics(results: List[Dict], k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Calculate retrieval metrics.
    
    Args:
        results: List of dicts with 'rank' (actual rank of correct match)
        k_values: Values of k for Recall@K
    
    Returns:
        dict: Metrics including Recall@K, MRR, etc.
    """
    ranks = [r['rank'] for r in results]
    
    total_queries = len(ranks)
    if total_queries == 0:
        return {}
    
    metrics = {
        'total_queries': total_queries,
        'mean_rank': float(np.mean(ranks)),
        'median_rank': int(np.median(ranks)),
        'min_rank': int(np.min(ranks)),
        'max_rank': int(np.max(ranks)),
        'std_rank': float(np.std(ranks)),
    }
    
    for k in k_values:
        hits = sum(1 for r in ranks if r <= k)
        metrics[f'recall@{k}'] = hits / total_queries
        metrics[f'recall@{k}_count'] = hits
    
    reciprocal_ranks = [1.0 / r for r in ranks]
    metrics['mrr'] = float(np.mean(reciprocal_ranks))
    
    metrics['correct_rank1'] = sum(1 for r in ranks if r == 1)
    
    cmc = np.zeros(max(ranks))
    for k in range(1, max(ranks) + 1):
        cmc[k-1] = sum(1 for r in ranks if r <= k) / total_queries
    metrics['cmc'] = cmc.tolist()
    
    return metrics


def bootstrap_confidence_interval(results: List[Dict], metric: str = 'recall@1', 
                                   n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a given metric.
    
    Args:
        results: List of query results
        metric: Metric name ('recall@1', 'recall@5', 'mrr')
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(results)
    if n == 0:
        return (0.0, 0.0)
    
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(n, size=n, replace=True)
        sample_results = [results[i] for i in sample_indices]
        
        if metric.startswith('recall@'):
            k = int(metric.split('@')[1])
            ranks = [r['rank'] for r in sample_results]
            score = sum(1 for r in ranks if r <= k) / len(ranks)
        elif metric == 'mrr':
            ranks = [r['rank'] for r in sample_results]
            score = np.mean([1.0 / r for r in ranks])
        else:
            score = 0.0
        
        bootstrap_scores.append(score)
    
    lower = np.percentile(bootstrap_scores, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 + confidence) / 2 * 100)
    
    return (float(lower), float(upper))


def evaluate_retrieval(model: Optional[PseudoSiameseNet] = None,
                       data_dir: str = DATA_DIR,
                       gallery_db_path: str = GALLERY_DB_PATH,
                       model_checkpoint: str = MODEL_CHECKPOINT,
                       results_path: str = RESULTS_PATH,
                       device: torch.device = device,
                       verbose: bool = True) -> Dict:
    """
    Main evaluation function.
    
    Args:
        model: Pre-loaded model (optional)
        data_dir: Data directory path
        gallery_db_path: Path to gallery database
        model_checkpoint: Path to model checkpoint
        results_path: Path to save results JSON
        device: Torch device
        verbose: Print progress
    
    Returns:
        Dict of metrics
    """
    if verbose:
        print("=" * 60)
        print("EVALUATION: Sketch-to-Photo Retrieval")
        print("=" * 60)
    
    if not os.path.exists(gallery_db_path):
        if verbose:
            print(f"Error: Gallery database not found at {gallery_db_path}")
            print("Please run: python gallery.py")
        return {}
    
    if verbose:
        print("Loading gallery database...")
    vector_db = torch.load(gallery_db_path, map_location=device)
    if verbose:
        print(f"Gallery size: {len(vector_db)} images")
    
    if model is None:
        if verbose:
            print("Loading model...")
        model = PseudoSiameseNet().to(device)
        if os.path.exists(model_checkpoint):
            model.load_state_dict(torch.load(model_checkpoint, map_location=device))
            if verbose:
                print(f"Loaded checkpoint from {model_checkpoint}")
        else:
            if verbose:
                print("Warning: No checkpoint found, using pretrained weights")
    
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if verbose:
        print("Loading test pairs...")
    test_pairs = load_test_pairs(data_dir)
    if verbose:
        print(f"Found {len(test_pairs)} test pairs")
    
    if len(test_pairs) == 0:
        if verbose:
            print("Error: No test pairs found!")
        return {}
    
    if verbose:
        print("Running evaluation...")
    results = []
    
    with torch.no_grad():
        iterator = tqdm(test_pairs, desc="Evaluating") if verbose else test_pairs
        for sketch_path, photo_path, correct_id in iterator:
            sketch_img = Image.open(sketch_path).convert('RGB')
            sketch_tensor = transform(sketch_img).unsqueeze(0).to(device)
            
            sketch_emb = model.forward_sketch(sketch_tensor)
            sketch_emb = torch.nn.functional.normalize(sketch_emb, p=2, dim=1)
            
            scores = []
            for gallery_id, gallery_data in vector_db.items():
                gallery_emb = gallery_data['embedding'].to(device)
                score = torch.sum(sketch_emb * gallery_emb).item()
                scores.append((gallery_id, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            rank = None
            for i, (gid, _) in enumerate(scores):
                if gid == correct_id:
                    rank = i + 1
                    break
            
            if rank is None:
                rank = len(scores) + 1
            
            results.append({
                'sketch': os.path.basename(sketch_path),
                'correct_id': correct_id,
                'rank': rank,
                'top5_ids': [s[0] for s in scores[:5]],
                'top5_scores': [s[1] for s in scores[:5]]
            })
    
    if verbose:
        print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    if verbose:
        print("\nCalculating bootstrap confidence intervals...")
    metrics['recall@1_ci'] = bootstrap_confidence_interval(results, 'recall@1')
    metrics['recall@5_ci'] = bootstrap_confidence_interval(results, 'recall@5')
    metrics['mrr_ci'] = bootstrap_confidence_interval(results, 'mrr')
    
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"\n--- Recall@K ---")
        print(f"Recall@1 (Accuracy@1): {metrics['recall@1']:.4f} ({metrics['recall@1']*100:.2f}%)")
        print(f"  95% CI: [{metrics['recall@1_ci'][0]:.4f}, {metrics['recall@1_ci'][1]:.4f}]")
        print(f"  - Correct at rank 1: {metrics['correct_rank1']}/{metrics['total_queries']}")
        print(f"Recall@5 (Accuracy@5): {metrics['recall@5']:.4f} ({metrics['recall@5']*100:.2f}%)")
        print(f"  95% CI: [{metrics['recall@5_ci'][0]:.4f}, {metrics['recall@5_ci'][1]:.4f}]")
        print(f"Recall@10 (Accuracy@10): {metrics['recall@10']:.4f} ({metrics['recall@10']*100:.2f}%)")
        print(f"\n--- Mean Reciprocal Rank ---")
        print(f"MRR: {metrics['mrr']:.4f}")
        print(f"  95% CI: [{metrics['mrr_ci'][0]:.4f}, {metrics['mrr_ci'][1]:.4f}]")
        print(f"\n--- Rank Statistics ---")
        print(f"Mean Rank: {metrics['mean_rank']:.2f} ± {metrics['std_rank']:.2f}")
        print(f"Median Rank: {metrics['median_rank']}")
        print(f"Best Rank: {metrics['min_rank']}")
        print(f"Worst Rank: {metrics['max_rank']}")
        print("=" * 60)
    
    output = {
        'metrics': metrics,
        'per_query_results': results
    }
    
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    if verbose:
        print(f"\nDetailed results saved to {results_path}")
    
    return metrics


def plot_cmc_curve(metrics: Dict, save_path: str = 'cmc_curve.png'):
    """
    Plot Cumulative Matching Characteristic (CMC) curve.
    
    Args:
        metrics: Metrics dict containing 'cmc' list
        save_path: Path to save the figure
    """
    if 'cmc' not in metrics:
        print("No CMC data available")
        return
    
    cmc = metrics['cmc']
    ranks = list(range(1, len(cmc) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, cmc, 'b-', linewidth=2, label='CMC')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    for k in [1, 5, 10]:
        if k <= len(cmc):
            plt.axvline(x=k, color='red', linestyle=':', alpha=0.5)
            plt.scatter([k], [cmc[k-1]], color='red', s=100, zorder=5)
            plt.annotate(f'R@{k}={cmc[k-1]:.3f}', xy=(k, cmc[k-1]), 
                        xytext=(k+2, cmc[k-1]-0.05), fontsize=10)
    
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Recognition Rate', fontsize=12)
    plt.title('Cumulative Matching Characteristic (CMC) Curve', fontsize=14)
    plt.xlim([0, min(50, len(cmc) + 5)])
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CMC curve saved to {save_path}")


def plot_rank_distribution(results: List[Dict], save_path: str = 'rank_distribution.png'):
    """
    Plot histogram of rank distribution.
    
    Args:
        results: List of query results
        save_path: Path to save the figure
    """
    ranks = [r['rank'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(ranks, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(ranks), color='red', linestyle='--', label=f'Mean: {np.mean(ranks):.1f}')
    plt.axvline(x=np.median(ranks), color='green', linestyle='--', label=f'Median: {np.median(ranks):.1f}')
    
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Correct Match Ranks', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Rank distribution saved to {save_path}")


def print_summary_report(metrics: Dict):
    """
    Print a formatted summary report.
    """
    if metrics is None:
        return
    
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print("\nModel Performance on Sketch-to-Photo Retrieval:")
    print("-" * 40)
    print(f" Top-1 Accuracy: {metrics['recall@1']*100:6.2f}%")
    print(f" Top-5 Accuracy: {metrics['recall@5']*100:6.2f}%")
    print(f" Top-10 Accuracy: {metrics['recall@10']*100:6.2f}%")
    print("-" * 40)
    print(f" MRR: {metrics['mrr']:6.4f}")
    print("-" * 40)
    print(f" Mean Rank: {metrics['mean_rank']:6.2f}")
    print("=" * 60)


if __name__ == '__main__':
    metrics = evaluate_retrieval()
    if metrics:
        print_summary_report(metrics)
        
        with open('evaluation_results.json', 'r') as f:
            results_data = json.load(f)
        
        plot_cmc_curve(metrics, 'cmc_curve.png')
        plot_rank_distribution(results_data['per_query_results'], 'rank_distribution.png')
