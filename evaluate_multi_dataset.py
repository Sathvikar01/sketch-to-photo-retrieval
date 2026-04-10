"""
Multi-dataset evaluation script.
Evaluates model on both CUFS and Color FERET datasets separately,
then produces combined/overall results.
"""
import torch
import os
import glob
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import PseudoSiameseNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_test_pairs(data_dir):
    """Load sketch-photo test pairs from a dataset."""
    test_sketches_dir = os.path.join(data_dir, 'test', 'sketches')
    test_photos_dir = os.path.join(data_dir, 'test', 'photos')

    if not os.path.exists(test_sketches_dir) or not os.path.exists(test_photos_dir):
        return []

    sketch_files = glob.glob(os.path.join(test_sketches_dir, '*'))
    sketch_files = [f for f in sketch_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    pairs = []
    for sketch_path in sketch_files:
        sketch_name = os.path.basename(sketch_path)
        base = os.path.splitext(sketch_name)[0]

        for suffix in ['_Sz', '-1', '_sketch', '_syn_pencil', '_syn_edge', '_syn_xdog']:
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


def build_gallery(model, photos_dir, device):
    """Build gallery embeddings from photos directory."""
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    photo_files = glob.glob(os.path.join(photos_dir, '*'))
    photo_files = [f for f in photo_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    photo_files = sorted(photo_files)

    gallery = {}
    for photo_path in tqdm(photo_files, desc="Building gallery", leave=False):
        photo_id = os.path.basename(photo_path)
        try:
            img = Image.open(photo_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.forward_photo(img_tensor)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            gallery[photo_id] = emb
        except Exception as e:
            print(f"  Error processing {photo_id}: {e}")
    
    return gallery


def evaluate_on_dataset(model, data_dir, dataset_name, device):
    """Evaluate model on a single dataset."""
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    photos_dir = os.path.join(data_dir, 'test', 'photos')
    gallery = build_gallery(model, photos_dir, device)
    
    test_pairs = load_test_pairs(data_dir)
    if len(test_pairs) == 0:
        return None

    results = []
    with torch.no_grad():
        for sketch_path, photo_path, correct_id in tqdm(test_pairs, desc=f"Evaluating {dataset_name}", leave=False):
            sketch_img = Image.open(sketch_path).convert('RGB')
            sketch_tensor = transform(sketch_img).unsqueeze(0).to(device)
            
            sketch_emb = model.forward_sketch(sketch_tensor)
            sketch_emb = torch.nn.functional.normalize(sketch_emb, p=2, dim=1)
            
            scores = []
            for gallery_id, gallery_emb in gallery.items():
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
                'gallery_size': len(gallery)
            })

    ranks = [r['rank'] for r in results]
    total = len(ranks)
    
    metrics = {
        'dataset': dataset_name,
        'total_queries': total,
        'gallery_size': len(gallery),
        'recall@1': sum(1 for r in ranks if r <= 1) / total,
        'recall@1_count': sum(1 for r in ranks if r <= 1),
        'recall@5': sum(1 for r in ranks if r <= 5) / total,
        'recall@5_count': sum(1 for r in ranks if r <= 5),
        'recall@10': sum(1 for r in ranks if r <= 10) / total,
        'recall@10_count': sum(1 for r in ranks if r <= 10),
        'mrr': float(np.mean([1.0 / r for r in ranks])),
        'mean_rank': float(np.mean(ranks)),
        'median_rank': int(np.median(ranks)),
        'min_rank': int(np.min(ranks)),
        'max_rank': int(np.max(ranks)),
        'std_rank': float(np.std(ranks)),
    }
    
    return metrics, results


def print_results(all_metrics):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    header = f"{'Dataset':<15} {'Queries':>8} {'Gallery':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Mean Rank':>10}"
    print(header)
    print("-" * 80)
    
    total_queries = 0
    total_r1 = 0
    total_r5 = 0
    total_r10 = 0
    weighted_mrr_sum = 0
    
    for m in all_metrics:
        print(f"{m['dataset']:<15} {m['total_queries']:>8} {m['gallery_size']:>8} "
              f"{m['recall@1']*100:>7.2f}% {m['recall@5']*100:>7.2f}% {m['recall@10']*100:>7.2f}% "
              f"{m['mrr']:>8.4f} {m['mean_rank']:>10.2f}")
        
        total_queries += m['total_queries']
        total_r1 += m['recall@1_count']
        total_r5 += m['recall@5_count']
        total_r10 += m['recall@10_count']
        weighted_mrr_sum += m['mrr'] * m['total_queries']
    
    print("-" * 80)
    
    overall_r1 = total_r1 / total_queries if total_queries > 0 else 0
    overall_r5 = total_r5 / total_queries if total_queries > 0 else 0
    overall_r10 = total_r10 / total_queries if total_queries > 0 else 0
    overall_mrr = weighted_mrr_sum / total_queries if total_queries > 0 else 0
    
    print(f"{'OVERALL':<15} {total_queries:>8} {'':>8} "
          f"{overall_r1*100:>7.2f}% {overall_r5*100:>7.2f}% {overall_r10*100:>7.2f}% "
          f"{overall_mrr:>8.4f} {'':>10}")
    print("=" * 80)
    
    print("\nDetailed Per-Dataset Breakdown:")
    print("-" * 40)
    for m in all_metrics:
        print(f"\n{m['dataset'].upper()}:")
        print(f"  Recall@1:  {m['recall@1']*100:.2f}% ({m['recall@1_count']}/{m['total_queries']})")
        print(f"  Recall@5:  {m['recall@5']*100:.2f}% ({m['recall@5_count']}/{m['total_queries']})")
        print(f"  Recall@10: {m['recall@10']*100:.2f}% ({m['recall@10_count']}/{m['total_queries']})")
        print(f"  MRR:       {m['mrr']:.4f}")
        print(f"  Mean Rank: {m['mean_rank']:.2f} +/- {m['std_rank']:.2f}")
        print(f"  Median:    {m['median_rank']}")
        print(f"  Best:      {m['min_rank']}")
        print(f"  Worst:     {m['max_rank']}")
    
    print(f"\n{'='*40}")
    print(f"OVERALL SUMMARY:")
    print(f"  Total Queries: {total_queries}")
    print(f"  Recall@1:  {overall_r1*100:.2f}%")
    print(f"  Recall@5:  {overall_r5*100:.2f}%")
    print(f"  Recall@10: {overall_r10*100:.2f}%")
    print(f"  MRR:       {overall_mrr:.4f}")
    print(f"{'='*40}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Multi-dataset evaluation')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/regularized_v1/best_model.pth',
                        help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='multi_dataset_results.json',
                        help='Output results file')
    args = parser.parse_args()

    print(f"Using device: {device}")
    
    model = PseudoSiameseNet(pretrained='vggface2').to(device)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    model.eval()

    datasets = {
        'CUFS': 'data/dataset/CUFS_reorganized',
        'ColorFERET': 'data/dataset/colorferet',
    }

    all_metrics = []
    all_results = {}

    for name, data_dir in datasets.items():
        if not os.path.exists(data_dir):
            print(f"\nSkipping {name}: directory not found at {data_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating on {name}...")
        print(f"{'='*60}")
        
        result = evaluate_on_dataset(model, data_dir, name, device)
        if result is None:
            print(f"  No test data found for {name}")
            continue
        
        metrics, results = result
        all_metrics.append(metrics)
        all_results[name] = {
            'metrics': metrics,
            'per_query': results
        }

    if all_metrics:
        print_results(all_metrics)
        
        output = {
            'checkpoint': args.checkpoint,
            'datasets': all_results,
            'overall': {
                'total_queries': sum(m['total_queries'] for m in all_metrics),
                'recall@1': sum(m['recall@1_count'] for m in all_metrics) / sum(m['total_queries'] for m in all_metrics),
                'recall@5': sum(m['recall@5_count'] for m in all_metrics) / sum(m['total_queries'] for m in all_metrics),
                'recall@10': sum(m['recall@10_count'] for m in all_metrics) / sum(m['total_queries'] for m in all_metrics),
                'mrr': sum(m['mrr'] * m['total_queries'] for m in all_metrics) / sum(m['total_queries'] for m in all_metrics),
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
