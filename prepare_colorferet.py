"""
Prepare Color FERET dataset for sketch-to-photo matching.
Extracts bz2-compressed TIFF images, converts to JPG,
and organizes into CUFS-compatible train/test/display structure.

Uses actual FERET images:
- Frontal photos (fa, fb) as "photos" gallery
- Profile images (hl, hr) as "sketches" (cross-modal matching)
"""
import os
import bz2
import io
import glob
import random
import argparse
from PIL import Image
from tqdm import tqdm


def extract_bz2_to_pil(bz2_path):
    """Extract a bz2-compressed TIFF file and return as PIL Image."""
    with open(bz2_path, 'rb') as f:
        compressed_data = f.read()
    
    decompressed_data = bz2.decompress(compressed_data)
    img = Image.open(io.BytesIO(decompressed_data))
    return img


def collect_feret_images(images_dir):
    """
    Collect all FERET images and group by person ID.
    FERET naming: PPPPPTTSSS_DDDDDD.tif.bz2
    PPPPP = person ID (5 digits)
    TT = pose code (fa, fb, hl, hr, pl, pr, etc.)
    SSS = session number
    DDDDDD = date
    """
    bz2_files = glob.glob(os.path.join(images_dir, '*.bz2'))
    
    person_images = {}
    
    for bz2_path in bz2_files:
        filename = os.path.basename(bz2_path)
        name_part = filename.replace('.tif.bz2', '')
        
        person_id = name_part[:5]
        pose_code = name_part[5:7]
        
        if person_id not in person_images:
            person_images[person_id] = []
        
        person_images[person_id].append({
            'path': bz2_path,
            'person_id': person_id,
            'pose': pose_code,
            'full_name': name_part
        })
    
    return person_images


def prepare_colorferet(
    source_dir='data/colorferet/dvd2/gray_feret_cd1/data/images',
    output_dir='data/dataset/colorferet',
    train_ratio=0.60,
    test_ratio=0.30,
    display_ratio=0.10,
    seed=42,
    max_persons=None
):
    """
    Prepare Color FERET dataset with train/test/display splits.
    Uses actual images - no synthetic generation.
    """
    random.seed(seed)
    
    print(f"Preparing Color FERET dataset...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    person_images = collect_feret_images(source_dir)
    print(f"Found {len(person_images)} unique persons")
    
    if max_persons:
        person_ids = sorted(list(person_images.keys()))[:max_persons]
        print(f"Using first {max_persons} persons")
    else:
        person_ids = sorted(list(person_images.keys()))
    
    persons_with_pairs = []
    
    for person_id in tqdm(person_ids, desc="Processing persons"):
        images = person_images[person_id]
        
        photos = [img for img in images if img['pose'] in ['fa', 'fb']]
        profiles = [img for img in images if img['pose'] in ['hl', 'hr']]
        
        if len(photos) >= 1 and len(profiles) >= 1:
            persons_with_pairs.append({
                'person_id': person_id,
                'photos': photos,
                'profiles': profiles
            })
    
    print(f"\nFound {len(persons_with_pairs)} persons with photo+profile pairs")
    
    random.shuffle(persons_with_pairs)
    
    n_total = len(persons_with_pairs)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)
    n_display = n_total - n_train - n_test
    
    train_persons = persons_with_pairs[:n_train]
    test_persons = persons_with_pairs[n_train:n_train + n_test]
    display_persons = persons_with_pairs[n_train + n_test:]
    
    print(f"\nSplit sizes:")
    print(f"  Training: {len(train_persons)} persons ({len(train_persons)/n_total*100:.1f}%)")
    print(f"  Testing:  {len(test_persons)} persons ({len(test_persons)/n_total*100:.1f}%)")
    print(f"  Display:  {len(display_persons)} persons ({len(display_persons)/n_total*100:.1f}%)")
    
    for split_name, split_persons in [('train', train_persons), ('test', test_persons), ('display', display_persons)]:
        split_photos_dir = os.path.join(output_dir, split_name, 'photos')
        split_sketches_dir = os.path.join(output_dir, split_name, 'sketches')
        os.makedirs(split_photos_dir, exist_ok=True)
        os.makedirs(split_sketches_dir, exist_ok=True)
        
        print(f"\nProcessing {split_name} split...")
        
        photo_count = 0
        sketch_count = 0
        
        for person_data in tqdm(split_persons, desc=f"  {split_name}"):
            person_id = person_data['person_id']
            
            for photo_data in person_data['photos']:
                try:
                    img = extract_bz2_to_pil(photo_data['path'])
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    output_name = f"{person_id}.jpg"
                    output_path = os.path.join(split_photos_dir, output_name)
                    img.save(output_path, 'JPEG', quality=95)
                    photo_count += 1
                except Exception as e:
                    print(f"  Error processing {photo_data['path']}: {e}")
            
            for profile_data in person_data['profiles']:
                try:
                    img = extract_bz2_to_pil(profile_data['path'])
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    output_name = f"{person_id}_Sz.jpg"
                    output_path = os.path.join(split_sketches_dir, output_name)
                    img.save(output_path, 'JPEG', quality=95)
                    sketch_count += 1
                except Exception as e:
                    print(f"  Error processing {profile_data['path']}: {e}")
        
        print(f"  {split_name}: {photo_count} photos, {sketch_count} sketches")
    
    with open(os.path.join(output_dir, 'split_info.txt'), 'w') as f:
        f.write(f"Dataset: Color FERET\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Train ratio: {train_ratio}\n")
        f.write(f"Test ratio: {test_ratio}\n")
        f.write(f"Display ratio: {display_ratio}\n")
        f.write(f"Total persons: {n_total}\n")
        f.write(f"Train persons: {len(train_persons)}\n")
        f.write(f"Test persons: {len(test_persons)}\n")
        f.write(f"Display persons: {len(display_persons)}\n")
    
    print(f"\nDataset prepared at: {output_dir}")
    
    return len(train_persons), len(test_persons), len(display_persons)


def main():
    parser = argparse.ArgumentParser(description='Prepare Color FERET dataset')
    parser.add_argument('--source', type=str, default='data/colorferet/dvd2/gray_feret_cd1/data/images',
                        help='Source images directory')
    parser.add_argument('--output', type=str, default='data/dataset/colorferet',
                        help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.60)
    parser.add_argument('--test_ratio', type=float, default=0.30)
    parser.add_argument('--display_ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_persons', type=int, default=None,
                        help='Limit number of persons (for testing)')
    
    args = parser.parse_args()
    
    n_train, n_test, n_display = prepare_colorferet(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        display_ratio=args.display_ratio,
        seed=args.seed,
        max_persons=args.max_persons
    )
    
    print("\n" + "="*60)
    print("Color FERET dataset preparation complete!")
    print("="*60)
    print(f"  Train: {n_train} persons")
    print(f"  Test:  {n_test} persons")
    print(f"  Display: {n_display} persons")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
