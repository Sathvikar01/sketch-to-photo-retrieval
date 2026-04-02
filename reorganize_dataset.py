"""
Reorganize dataset for proper train/test/display split:
- Training: 60% of data for model training
- Testing: 30% of data for evaluation
- Display: 10% of data for UI demonstration

Supports both CUFS and CUFSF datasets.
"""
import os
import shutil
import glob
import random
import argparse
from pathlib import Path


def reorganize_dataset(
    source_dir='data/dataset/CUFS',
    target_dir='data/dataset/CUFS_reorganized',
    train_ratio=0.60,
    test_ratio=0.30,
    display_ratio=0.10,
    seed=42,
    dataset_type='CUFS'
):
    """
    Reorganize dataset into train/test/display splits.
    
    Args:
        source_dir: Path to original dataset
        target_dir: Path for reorganized dataset
        train_ratio: Fraction for training (default 60%)
        test_ratio: Fraction for testing/evaluation (default 30%)
        display_ratio: Fraction for UI display (default 10%)
        seed: Random seed for reproducibility
        dataset_type: 'CUFS' or 'CUFSF'
    """
    assert abs(train_ratio + test_ratio + display_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + test_ratio + display_ratio}"
    
    random.seed(seed)
    
    print(f"Reorganizing {dataset_type} dataset...")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Test:  {test_ratio*100:.0f}%")
    print(f"  Display: {display_ratio*100:.0f}%")
    
    if dataset_type == 'CUFS':
        pairs = _collect_cufs_pairs(source_dir)
    elif dataset_type == 'CUFSF':
        pairs = _collect_cufsf_pairs(source_dir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"\nFound {len(pairs)} matching photo-sketch pairs")
    
    random.shuffle(pairs)
    
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_display = n_total - n_train - n_test
    
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:n_train + n_test]
    display_pairs = pairs[n_train + n_test:]
    
    print(f"\nSplit sizes:")
    print(f"  Training: {len(train_pairs)} pairs ({len(train_pairs)/n_total*100:.1f}%)")
    print(f"  Testing:  {len(test_pairs)} pairs ({len(test_pairs)/n_total*100:.1f}%)")
    print(f"  Display:  {len(display_pairs)} pairs ({len(display_pairs)/n_total*100:.1f}%)")
    
    for split in ['train', 'test', 'display']:
        for modality in ['photos', 'sketches']:
            os.makedirs(os.path.join(target_dir, split, modality), exist_ok=True)
    
    _copy_pairs(train_pairs, target_dir, 'train')
    _copy_pairs(test_pairs, target_dir, 'test')
    _copy_pairs(display_pairs, target_dir, 'display')
    
    with open(os.path.join(target_dir, 'split_info.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Train ratio: {train_ratio}\n")
        f.write(f"Test ratio: {test_ratio}\n")
        f.write(f"Display ratio: {display_ratio}\n")
        f.write(f"Total pairs: {n_total}\n")
        f.write(f"Train pairs: {len(train_pairs)}\n")
        f.write(f"Test pairs: {len(test_pairs)}\n")
        f.write(f"Display pairs: {len(display_pairs)}\n")
    
    print(f"\nSaved to: {target_dir}")
    
    return len(train_pairs), len(test_pairs), len(display_pairs)


def _collect_cufs_pairs(source_dir):
    """Collect photo-sketch pairs from CUFS dataset."""
    all_photos = glob.glob(os.path.join(source_dir, 'train_photos', '*')) + \
                 glob.glob(os.path.join(source_dir, 'test_photos', '*'))
    all_sketches = glob.glob(os.path.join(source_dir, 'train_sketches', '*')) + \
                   glob.glob(os.path.join(source_dir, 'test_sketches', '*'))
    
    photo_dict = {}
    for p in all_photos:
        base = os.path.basename(p)
        name = os.path.splitext(base)[0]
        photo_dict[name] = p
    
    sketch_dict = {}
    for s in all_sketches:
        base = os.path.basename(s)
        name = os.path.splitext(base)[0]
        for suffix in ['_Sz', '-1', '_sketch']:
            name = name.replace(suffix, '')
        sketch_dict[name] = s
    
    pairs = []
    for name, photo_path in photo_dict.items():
        sketch_candidates = [
            name + '_Sz',
            name + '-1',
            name + '_sketch',
            name
        ]
        for sketch_name in sketch_candidates:
            if sketch_name in sketch_dict:
                pairs.append((photo_path, sketch_dict[sketch_name], name))
                break
    
    return pairs


def _collect_cufsf_pairs(source_dir):
    """Collect photo-sketch pairs from CUFSF dataset with photo variations."""
    pairs = []
    
    sketches_dir = os.path.join(source_dir, 'sketches')
    photos_dir = os.path.join(source_dir, 'photos')
    
    if not os.path.exists(sketches_dir) or not os.path.exists(photos_dir):
        print(f"Warning: CUFSF directories not found at {source_dir}")
        return pairs
    
    sketch_files = glob.glob(os.path.join(sketches_dir, '*'))
    
    for sketch_path in sketch_files:
        sketch_name = os.path.basename(sketch_path)
        base = os.path.splitext(sketch_name)[0]
        for suffix in ['_Sz', '-1', '_sketch']:
            base = base.replace(suffix, '')
        
        photo_dir = os.path.join(photos_dir, base)
        if os.path.isdir(photo_dir):
            photo_variations = glob.glob(os.path.join(photo_dir, '*'))
            for photo_path in photo_variations:
                pairs.append((photo_path, sketch_path, base))
        else:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                photo_path = os.path.join(photos_dir, base + ext)
                if os.path.exists(photo_path):
                    pairs.append((photo_path, sketch_path, base))
                    break
    
    return pairs


def _copy_pairs(pairs, target_dir, split):
    """Copy pairs to target directory."""
    for photo_path, sketch_path, name in pairs:
        photo_ext = os.path.splitext(photo_path)[1]
        sketch_ext = os.path.splitext(sketch_path)[1]
        
        photo_target = os.path.join(target_dir, split, 'photos', name + photo_ext)
        sketch_target = os.path.join(target_dir, split, 'sketches', name + sketch_ext)
        
        if not os.path.exists(photo_target):
            shutil.copy2(photo_path, photo_target)
        if not os.path.exists(sketch_target):
            shutil.copy2(sketch_path, sketch_target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorganize dataset with 60/30/10 split')
    parser.add_argument('--dataset', type=str, default='CUFS', choices=['CUFS', 'CUFSF'],
                        help='Dataset to reorganize')
    parser.add_argument('--source', type=str, default=None,
                        help='Source directory (default: data/dataset/{dataset})')
    parser.add_argument('--target', type=str, default=None,
                        help='Target directory (default: data/dataset/{dataset}_reorganized)')
    parser.add_argument('--train_ratio', type=float, default=0.60,
                        help='Training split ratio (default: 0.60)')
    parser.add_argument('--test_ratio', type=float, default=0.30,
                        help='Test split ratio (default: 0.30)')
    parser.add_argument('--display_ratio', type=float, default=0.10,
                        help='Display split ratio (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    source_dir = args.source or f'data/dataset/{args.dataset}'
    target_dir = args.target or f'data/dataset/{args.dataset}_reorganized'
    
    reorganize_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        display_ratio=args.display_ratio,
        seed=args.seed,
        dataset_type=args.dataset
    )
