"""
Reorganize dataset for proper train/test split:
- UI demo: Only 25 pairs (test set for visualization)
- Training: All remaining pairs
"""
import os
import shutil
import glob
import random
from pathlib import Path

def reorganize_dataset(
    source_dir='data/dataset/CUFS',
    target_dir='data/dataset/CUFS_reorganized',
    ui_demo_size=25,
    seed=42
):
    random.seed(seed)
    
    # Collect all photo-sketch pairs
    all_photos = glob.glob(os.path.join(source_dir, 'train_photos', '*')) + \
                 glob.glob(os.path.join(source_dir, 'test_photos', '*'))
    all_sketches = glob.glob(os.path.join(source_dir, 'train_sketches', '*')) + \
                   glob.glob(os.path.join(source_dir, 'test_sketches', '*'))
    
    # Build mapping
    photo_dict = {}
    for p in all_photos:
        base = os.path.basename(p)
        name = os.path.splitext(base)[0]
        photo_dict[name] = p
    
    sketch_dict = {}
    for s in all_sketches:
        base = os.path.basename(s)
        name = os.path.splitext(base)[0]
        # Remove common suffixes
        for suffix in ['_Sz', '-1', '_sketch']:
            name = name.replace(suffix, '')
        sketch_dict[name] = s
    
    # Find matching pairs
    pairs = []
    for name, photo_path in photo_dict.items():
        # Try different sketch name patterns
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
    
    print(f"Found {len(pairs)} matching photo-sketch pairs")
    
    # Shuffle and split
    random.shuffle(pairs)
    ui_demo_pairs = pairs[:ui_demo_size]
    training_pairs = pairs[ui_demo_size:]
    
    # Create target directories
    for split in ['train', 'test']:
        for modality in ['photos', 'sketches']:
            os.makedirs(os.path.join(target_dir, split, modality), exist_ok=True)
    
    # Copy UI demo pairs (test set)
    for photo_path, sketch_path, name in ui_demo_pairs:
        photo_ext = os.path.splitext(photo_path)[1]
        sketch_ext = os.path.splitext(sketch_path)[1]
        
        shutil.copy2(photo_path, os.path.join(target_dir, 'test', 'photos', name + photo_ext))
        shutil.copy2(sketch_path, os.path.join(target_dir, 'test', 'sketches', name + sketch_ext))
    
    # Copy training pairs
    for photo_path, sketch_path, name in training_pairs:
        photo_ext = os.path.splitext(photo_path)[1]
        sketch_ext = os.path.splitext(sketch_path)[1]
        
        shutil.copy2(photo_path, os.path.join(target_dir, 'train', 'photos', name + photo_ext))
        shutil.copy2(sketch_path, os.path.join(target_dir, 'train', 'sketches', name + sketch_ext))
    
    print(f"\nReorganized dataset:")
    print(f"  UI Demo (test): {len(ui_demo_pairs)} pairs")
    print(f"  Training: {len(training_pairs)} pairs")
    print(f"  Total: {len(pairs)} pairs")
    print(f"\nSaved to: {target_dir}")
    
    return len(ui_demo_pairs), len(training_pairs)

if __name__ == '__main__':
    reorganize_dataset()
