import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CUFSDataset(Dataset):
    """
    CUFS Dataset loader for Cross-Modal Sketch to Photo matching.
    Returns anchor (sketch), positive (matching photo), and negative (non-matching photo).
    
    Enhanced with:
    - Robust file handling (supports .jpg, .png)
    - Modality-aware augmentations
    - Hard negative sampling hooks
    """
    
    def __init__(self, data_dir, split='train', mtcnn=None, 
                 sketch_transform=None, photo_transform=None,
                 hard_negatives=False):
        self.data_dir = data_dir
        self.split = split
        self.mtcnn = mtcnn
        self.hard_negatives = hard_negatives
        
        self.sketches_dir = os.path.join(data_dir, f'{split}_sketches')
        self.photos_dir = os.path.join(data_dir, f'{split}_photos')
        
        self.filenames = self._get_matched_filenames()
        
        self.sketch_transform = sketch_transform or self._get_default_sketch_transform()
        self.photo_transform = photo_transform or self._get_default_photo_transform()
        
        self._build_id_mapping()
    
    def _get_matched_filenames(self):
        sketch_files = glob.glob(os.path.join(self.sketches_dir, '*'))
        sketch_files = [f for f in sketch_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        matched = []
        for sketch_path in sketch_files:
            sketch_name = os.path.basename(sketch_path)
            photo_path = os.path.join(self.photos_dir, sketch_name)
            if os.path.exists(photo_path):
                matched.append(sketch_name)
            else:
                base = os.path.splitext(sketch_name)[0]
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = os.path.join(self.photos_dir, base + ext)
                    if os.path.exists(alt_path):
                        matched.append(sketch_name)
                        break
        
        return sorted(matched)
    
    def _build_id_mapping(self):
        self.id_to_indices = {}
        for idx, fname in enumerate(self.filenames):
            base_id = self._get_person_id(fname)
            if base_id not in self.id_to_indices:
                self.id_to_indices[base_id] = []
            self.id_to_indices[base_id].append(idx)
    
    def _get_person_id(self, filename):
        base = os.path.splitext(filename)[0]
        for suffix in ['_Sz', '-1', '_sketch', '_photo']:
            base = base.replace(suffix, '')
        return base
    
    def _get_default_sketch_transform(self):
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _get_default_photo_transform(self):
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        sketch_path = os.path.join(self.sketches_dir, filename)
        photo_path = self._find_photo_path(filename)
        
        neg_idx = self._sample_negative(idx)
        neg_filename = self.filenames[neg_idx]
        neg_photo_path = self._find_photo_path(neg_filename)
        
        sketch = Image.open(sketch_path).convert('RGB')
        photo = Image.open(photo_path).convert('RGB')
        neg_photo = Image.open(neg_photo_path).convert('RGB')
        
        if self.mtcnn is not None:
            sketch = self._apply_mtcnn(sketch, self.sketch_transform)
            photo = self._apply_mtcnn(photo, self.photo_transform)
            neg_photo = self._apply_mtcnn(neg_photo, self.photo_transform)
        else:
            sketch = self.sketch_transform(sketch)
            photo = self.photo_transform(photo)
            neg_photo = self.photo_transform(neg_photo)
        
        return sketch, photo, neg_photo, idx
    
    def _find_photo_path(self, filename):
        direct_path = os.path.join(self.photos_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        
        base = os.path.splitext(filename)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            alt_path = os.path.join(self.photos_dir, base + ext)
            if os.path.exists(alt_path):
                return alt_path
        
        return direct_path
    
    def _sample_negative(self, anchor_idx):
        neg_idx = random.choice([i for i in range(len(self.filenames)) if i != anchor_idx])
        return neg_idx
    
    def _apply_mtcnn(self, img, fallback_transform):
        try:
            cropped = self.mtcnn(img)
            if cropped is not None:
                return cropped
        except Exception:
            pass
        return fallback_transform(img)
    
    def get_hard_negatives(self, anchor_embs, positive_embs, anchor_indices, k=4):
        """
        Find hard negatives within a batch for batch-hard mining.
        
        Args:
            anchor_embs: (B, D) tensor of sketch embeddings
            positive_embs: (B, D) tensor of positive photo embeddings
            anchor_indices: (B,) tensor of sample indices
            k: number of hard negatives per anchor
        
        Returns:
            List of (anchor_idx, neg_idx) tuples for hard negatives
        """
        B = anchor_embs.shape[0]
        
        with torch.no_grad():
            dist_matrix = torch.cdist(anchor_embs, positive_embs)
        
        hard_neg_pairs = []
        for i in range(B):
            anchor_id = self._get_person_id(self.filenames[anchor_indices[i].item()])
            
            valid_neg_mask = torch.ones(B, dtype=torch.bool)
            for j, idx_j in enumerate(anchor_indices):
                if self._get_person_id(self.filenames[idx_j.item()]) == anchor_id:
                    valid_neg_mask[j] = False
            valid_neg_mask[i] = False
            
            valid_dists = dist_matrix[i].clone()
            valid_dists[~valid_neg_mask] = float('inf')
            
            _, hardest_indices = torch.topk(valid_dists, min(k, valid_neg_mask.sum().item()), largest=False)
            
            for neg_j in hardest_indices:
                if valid_dists[neg_j] < float('inf'):
                    hard_neg_pairs.append((i, neg_j.item()))
        
        return hard_neg_pairs


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    dataset = CUFSDataset(data_dir='data/dataset/CUFS', split='train')
    print(f"Loaded {len(dataset)} train samples.")
    sketch, photo, neg_photo, idx = dataset[0]
    print(f"Sketch shape: {sketch.shape}")
    print(f"Photo shape: {photo.shape}")
    print(f"Neg Photo shape: {neg_photo.shape}")
    print(f"Index: {idx}")
