"""
Fast multi-dataset evaluation with batched processing.
"""
import torch
import os
import glob
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from model import PseudoSiameseNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PhotoDataset(Dataset):
    def __init__(self, photos_dir):
        self.photos_dir = photos_dir
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.files = sorted([
            f for f in glob.glob(os.path.join(photos_dir, '*'))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.ids = [os.path.basename(f) for f in self.files]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), self.ids[idx]


class SketchDataset(Dataset):
    def __init__(self, data_dir):
        self.sketches_dir = os.path.join(data_dir, 'test', 'sketches')
        self.photos_dir = os.path.join(data_dir, 'test', 'photos')
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        sketch_files = glob.glob(os.path.join(self.sketches_dir, '*'))
        sketch_files = [f for f in sketch_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.pairs = []
        for sketch_path in sketch_files:
            sketch_name = os.path.basename(sketch_path)
            base = os.path.splitext(sketch_name)[0]
            for suffix in ['_Sz', '-1', '_sketch', '_syn_pencil', '_syn_edge', '_syn_xdog']:
                base = base.replace(suffix, '')
            
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                photo_path = os.path.join(self.photos_dir, base + ext)
                if os.path.exists(photo_path):
                    self.pairs.append((sketch_path, base + ext))
                    break
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sketch_path, photo_id = self.pairs[idx]
        img = Image.open(sketch_path).convert('RGB')
        return self.transform(img), photo_id, os.path.basename(sketch_path)


def build_gallery_fast(model, photos_dir, batch_size=64):
    """Build gallery embeddings using batched processing."""
    dataset = PhotoDataset(photos_dir)
    if len(dataset) == 0:
        return {}
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    gallery = {}
    model.eval()
    with torch.no_grad():
        for batch_imgs, batch_ids in tqdm(loader, desc="Building gallery", leave=False):
            batch_imgs = batch_imgs.to(device)
            embs = model.forward_photo(batch_imgs)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            
            for emb, pid in zip(embs, batch_ids):
                gallery[pid] = emb.cpu()
    
    return gallery


def evaluate_dataset_fast(model, data_dir, dataset_name, batch_size=64):
    """Evaluate model on a dataset using batched sketch processing."""
    photos_dir = os.path.join(data_dir, 'test', 'photos')
    if not os.path.exists(photos_dir):
        return None, None
    
    gallery = build_gallery_fast(model, photos_dir, batch_size)
    if len(gallery) == 0:
        return None, None
    
    sketch_dataset = SketchDataset(data_dir)
    if len(sketch_dataset) == 0:
        return None, None
    
    sketch_loader = DataLoader(sketch_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    gallery_ids = list(gallery.keys())
    gallery_tensor = torch.stack([gallery[gid] for gid in gallery_ids]).to(device)
    
    results = []
    model.eval()
    with torch.no_grad():
        for batch_sketches, batch_photo_ids, batch_sketch_names in tqdm(
            sketch_loader, desc=f"Evaluating {dataset_name}", leave=False
        ):
            batch_sketches = batch_sketches.to(device)
            sketch_embs = model.forward_sketch(batch_sketches)
            sketch_embs = torch.nn.functional.normalize(sketch_embs, p=2, dim=1)
            
            scores = sketch_embs @ gallery_tensor.T
            
            for i in range(len(batch_photo_ids)):
                scores_i = scores[i].cpu().numpy()
                sorted_indices = np.argsort(-scores_i)
                
                correct_id = batch_photo_ids[i]
                rank = None
                for rank_pos, idx in enumerate(sorted_indices):
                    if gallery_ids[idx] == correct_id:
                        rank = rank_pos + 1
                        break
                
                if rank is None:
                    rank = len(gallery_ids) + 1
                
                results.append({
                    'sketch': batch_sketch_names[i],
                    'correct_id': correct_id,
                    'rank': rank,
                    'gallery_size': len(gallery_ids)
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


def print_results_table(all_metrics):
    """Print formatted results."""
    print("\n" + "=" * 85)
    print("EVALUATION RESULTS")
    print("=" * 85)
    
    header = f"{'Dataset':<15} {'Queries':>8} {'Gallery':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Mean Rank':>10}"
    print(header)
    print("-" * 85)
    
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
    
    print("-" * 85)
    
    overall_r1 = total_r1 / total_queries if total_queries > 0 else 0
    overall_r5 = total_r5 / total_queries if total_queries > 0 else 0
    overall_r10 = total_r10 / total_queries if total_queries > 0 else 0
    overall_mrr = weighted_mrr_sum / total_queries if total_queries > 0 else 0
    
    print(f"{'OVERALL':<15} {total_queries:>8} {'':>8} "
          f"{overall_r1*100:>7.2f}% {overall_r5*100:>7.2f}% {overall_r10*100:>7.2f}% "
          f"{overall_mrr:>8.4f} {'':>10}")
    print("=" * 85)
    
    print("\n" + "=" * 50)
    print("DETAILED PER-DATASET BREAKDOWN")
    print("=" * 50)
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
    
    print(f"\n{'='*50}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*50}")
    print(f"  Total Queries: {total_queries}")
    print(f"  Recall@1:  {overall_r1*100:.2f}%")
    print(f"  Recall@5:  {overall_r5*100:.2f}%")
    print(f"  Recall@10: {overall_r10*100:.2f}%")
    print(f"  MRR:       {overall_mrr:.4f}")
    print(f"{'='*50}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/regularized_v1/best_model.pth')
    parser.add_argument('--output', type=str, default='multi_dataset_results.json')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    print(f"Device: {device}")
    
    model = PseudoSiameseNet(pretrained='vggface2').to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded: {args.checkpoint}")
    model.eval()

    datasets = {
        'CUFS': 'data/dataset/CUFS_reorganized',
        'ColorFERET': 'data/dataset/colorferet',
    }

    all_metrics = []
    all_results = {}

    for name, data_dir in datasets.items():
        if not os.path.exists(data_dir):
            print(f"\nSkipping {name}: not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {name}...")
        print(f"{'='*60}")
        
        metrics, results = evaluate_dataset_fast(model, data_dir, name, args.batch_size)
        if metrics is None:
            print(f"  No data for {name}")
            continue
        
        all_metrics.append(metrics)
        all_results[name] = {'metrics': metrics, 'per_query': results}

    if all_metrics:
        print_results_table(all_metrics)
        
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
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
