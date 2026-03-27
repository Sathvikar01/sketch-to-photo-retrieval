import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import json
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

from dataset import CUFSDataset, set_seed, get_test_transform
from model import PseudoSiameseNet
from evaluation_metrics import evaluate_retrieval


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss with online hard negative mining.
    For each anchor, finds the hardest positive and hardest negative in the batch.
    """
    def __init__(self, margin=0.5, soft_margin=False):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) tensor of L2-normalized embeddings
            labels: (B,) tensor of person IDs
        Returns:
            loss: scalar tensor
        """
        B = embeddings.shape[0]
        
        dist_mat = torch.cdist(embeddings, embeddings)
        
        mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_pos.fill_diagonal_(False)
        
        mask_neg = labels.unsqueeze(1) != labels.unsqueeze(0)
        
        pos_dist = dist_mat.clone()
        pos_dist[~mask_pos] = 0.0
        hardest_pos, _ = pos_dist.max(dim=1)
        
        neg_dist = dist_mat.clone()
        neg_dist[mask_neg == False] = float('inf')
        hardest_neg, _ = neg_dist.min(dim=1)
        
        if self.soft_margin:
            loss = F.softplus(hardest_pos - hardest_neg)
        else:
            loss = F.relu(hardest_pos - hardest_neg + self.margin)
        
        loss = loss[loss > 0]
        if loss.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Triplet + Contrastive loss for better gradient signal.
    """
    def __init__(self, triplet_margin=0.5, contrastive_margin=1.0):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin, p=2)
        self.contrastive_loss = nn.MarginRankingLoss(margin=contrastive_margin)
    
    def forward(self, anchor, positive, negative):
        triplet = self.triplet_loss(anchor, positive, negative)
        
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        target = torch.ones_like(d_pos)
        contrastive = self.contrastive_loss(d_pos, d_neg, target)
        
        return triplet + 0.5 * contrastive


def train_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch, use_amp=True):
    """
    Train for one epoch with optional AMP.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        sketches, photos, neg_photos, indices = batch
        sketches = sketches.to(device)
        photos = photos.to(device)
        neg_photos = neg_photos.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with autocast():
                emb_a, emb_p, emb_n = model(sketches, photos, neg_photos)
                loss = criterion(emb_a, emb_p, emb_n)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            emb_a, emb_p, emb_n = model(sketches, photos, neg_photos)
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, data_dir, device, gallery_path='gallery_db.pt'):
    """
    Run validation using evaluation_metrics.py
    Returns dict with Recall@1, Recall@5, MRR.
    """
    model.eval()
    
    try:
        metrics = evaluate_retrieval(
            model=model,
            data_dir=data_dir,
            gallery_db_path=gallery_path,
            device=device,
            verbose=False
        )
        return metrics
    except Exception as e:
        print(f"Validation error: {e}")
        return {'recall@1': 0.0, 'recall@5': 0.0, 'mrr': 0.0}


def main():
    parser = argparse.ArgumentParser(description='Train Pseudo-Siamese Network')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--margin', type=float, default=0.5, help='Triplet margin')
    parser.add_argument('--data_dir', type=str, default='data/dataset/CUFS', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--experiment_name', type=str, default='baseline', help='Experiment name')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    experiment_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    train_dataset = CUFSDataset(
        data_dir=args.data_dir,
        split='train',
        hard_negatives=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    model = PseudoSiameseNet(pretrained='vggface2').to(device)
    
    criterion = CombinedLoss(triplet_margin=args.margin)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs // 3,
        T_mult=2,
        eta_min=args.lr / 100
    )
    
    scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
    
    history = {
        'train_loss': [],
        'val_recall@1': [],
        'val_recall@5': [],
        'val_mrr': [],
        'lr': []
    }
    
    best_recall = 0.0
    patience_counter = 0
    
    config = vars(args)
    config['device'] = str(device)
    config['num_params'] = sum(p.numel() for p in model.parameters())
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"{'='*60}")
    
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor
            current_lr = args.lr * warmup_factor
        else:
            scheduler.step()
            current_lr = optimizer.param_group(0)['lr']
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, epoch, use_amp=args.use_amp
        )
        
        history['train_loss'].append(train_loss)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if epoch % 5 == 0 or epoch == args.epochs:
            metrics = validate(model, args.data_dir, device)
            
            history['val_recall@1'].append(metrics.get('recall@1', 0.0))
            history['val_recall@5'].append(metrics.get('recall@5', 0.0))
            history['val_mrr'].append(metrics.get('mrr', 0.0))
            
            current_recall = metrics.get('recall@1', 0.0)
            
            print(f"  Val Recall@1: {current_recall:.4f}")
            print(f"  Val Recall@5: {metrics.get('recall@5', 0.0):.4f}")
            print(f"  Val MRR: {metrics.get('mrr', 0.0):.4f}")
            
            if current_recall > best_recall:
                best_recall = current_recall
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_recall': best_recall,
                    'metrics': metrics,
                }, os.path.join(experiment_dir, 'best_model.pth'))
                
                print(f"  Saved best model (Recall@1: {best_recall:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(experiment_dir, 'final_model.pth'))
    
    with open(os.path.join(experiment_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"Best Recall@1: {best_recall:.4f}")
    print(f"Checkpoints saved to: {experiment_dir}")
    print(f"{'='*60}")
    
    return history


if __name__ == '__main__':
    main()
