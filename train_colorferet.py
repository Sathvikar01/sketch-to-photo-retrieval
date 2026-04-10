"""
Fine-tune model on Color FERET dataset starting from existing checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import json
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

from dataset import CUFSDataset, set_seed
from model import PseudoSiameseNet


class CombinedLoss(nn.Module):
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


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
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
        
        emb_a, emb_p, emb_n = model(sketches, photos, neg_photos)
        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--data_dir', type=str, default='data/dataset/colorferet')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--experiment_name', type=str, default='colorferet_finetuned')
    parser.add_argument('--pretrained', type=str, default='checkpoints/regularized_v1/best_model.pth')
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(os.path.join(args.checkpoint_dir, args.experiment_name), exist_ok=True)
    
    model = PseudoSiameseNet(pretrained='vggface2').to(device)
    
    if os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"Fine-tuning from: {args.pretrained}")
    
    train_dataset = CUFSDataset(data_dir=args.data_dir, split='train', hard_negatives=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    print(f"Training samples: {len(train_dataset)}")
    
    criterion = CombinedLoss(triplet_margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 3, T_mult=2, eta_min=args.lr / 100)
    
    history = {'train_loss': [], 'lr': []}
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning on Color FERET")
    print(f"{'='*60}")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(args.checkpoint_dir, args.experiment_name, 'best_model.pth'))
            print(f"  Saved best model (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(args.checkpoint_dir, args.experiment_name, 'final_model.pth'))
    
    with open(os.path.join(args.checkpoint_dir, args.experiment_name, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning Complete")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
