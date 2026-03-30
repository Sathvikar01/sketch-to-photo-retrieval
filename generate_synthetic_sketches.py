"""
Generate synthetic sketches from face photos using various edge detection methods.
This augments the training data for sketch-to-photo matching.
"""
import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

def pencil_sketch(img, dilate=True):
    """Generate pencil sketch using OpenCV"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_inv = 255 - gray
    blur = cv2.GaussianBlur(gray_inv, ksize=(21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    
    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        sketch = cv2.dilate(sketch, kernel, iterations=1)
    
    return sketch

def edge_sketch(img, low_threshold=30, high_threshold=100):
    """Generate sketch using Canny edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges = 255 - edges
    return edges

def xdog_sketch(img, sigma=1.5, k=1.6, gamma=0.95, epsilon=0.01, phi=10):
    """XDoG (eXtended Difference of Gaussians) sketch"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    
    dog = g1 - gamma * g2
    
    xdog = np.where(dog >= epsilon, 
                    np.ones_like(dog), 
                    1 + np.tanh(phi * (dog - epsilon)))
    
    xdog = (xdog * 255).astype(np.uint8)
    return xdog

def generate_synthetic_sketches(
    photos_dir,
    output_dir,
    styles=['pencil', 'edge', 'xdog'],
    variations_per_image=3
):
    """
    Generate synthetic sketches from photos.
    
    Args:
        photos_dir: Directory containing face photos
        output_dir: Directory to save synthetic sketches
        styles: List of sketch styles to generate
        variations_per_image: Number of variations per photo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    photo_files = glob.glob(os.path.join(photos_dir, '*'))
    photo_files = [f for f in photo_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Generating synthetic sketches from {len(photo_files)} photos...")
    
    for photo_path in tqdm(photo_files, desc="Generating sketches"):
        img = cv2.imread(photo_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        base_name = os.path.splitext(os.path.basename(photo_path))[0]
        
        for i, style in enumerate(styles[:variations_per_image]):
            if style == 'pencil':
                sketch = pencil_sketch(img)
            elif style == 'edge':
                # Random threshold variations
                low = random.randint(20, 50)
                high = random.randint(80, 150)
                sketch = edge_sketch(img, low, high)
            elif style == 'xdog':
                sketch = xdog_sketch(img)
            else:
                continue
            
            sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
            
            output_name = f"{base_name}_syn_{style}.jpg"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR))
    
    total_generated = len(glob.glob(os.path.join(output_dir, '*')))
    print(f"Generated {total_generated} synthetic sketches in {output_dir}")

def augment_training_data(
    train_photos_dir='data/dataset/CUFS_reorganized/train/photos',
    train_sketches_dir='data/dataset/CUFS_reorganized/train/sketches',
    synthetic_output_dir='data/dataset/CUFS_reorganized/train/synthetic_sketches'
):
    """
    Augment training data with synthetic sketches.
    Creates additional sketch variations for each photo.
    """
    generate_synthetic_sketches(
        photos_dir=train_photos_dir,
        output_dir=synthetic_output_dir,
        styles=['pencil', 'edge', 'xdog'],
        variations_per_image=3
    )
    
    print(f"\nTraining data augmented:")
    original = len(glob.glob(os.path.join(train_sketches_dir, '*')))
    synthetic = len(glob.glob(os.path.join(synthetic_output_dir, '*')))
    print(f"  Original sketches: {original}")
    print(f"  Synthetic sketches: {synthetic}")
    print(f"  Total: {original + synthetic}")

if __name__ == '__main__':
    augment_training_data()
