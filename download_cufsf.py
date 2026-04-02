"""
Download and prepare CUFSF (CUHK Face Sketch & Face) dataset.

CUFSF extends CUFS with multiple photo variations per identity,
testing robustness to lighting, expression, and pose changes.

Note: CUFSF requires permission from CUHK. This script provides
instructions and creates the expected directory structure.

Dataset source: http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html

Expected CUFSF structure after download:
CUFSF/
├── photos/
│   ├── person_001/
│   │   ├── photo_1.jpg
│   │   ├── photo_2.jpg (variation)
│   │   └── photo_3.jpg (variation)
│   └── ...
└── sketches/
    ├── person_001_Sz.jpg
    └── ...
"""
import os
import urllib.request
import zipfile
import argparse


CUFSF_URLS = {
    'photos': 'http://mmlab.ie.cuhk.edu.hk/archive/facesketch/CUFSF/photos.zip',
    'sketches': 'http://mmlab.ie.cuhk.edu.hk/archive/facesketch/CUFSF/sketches.zip',
}

CUFSF_ALTERNATIVE = """
CUFSF dataset may need to be downloaded manually from:
http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html

Alternative sources:
1. CUHK Face Sketch Database (requires registration)
2. Academic Torrents (if available)
3. Contact authors directly

After downloading, extract to: data/dataset/CUFSF/
"""


def download_cufsf(target_dir='data/dataset/CUFSF', force=False):
    """
    Download CUFSF dataset.
    
    Note: This may require manual download due to licensing.
    """
    if os.path.exists(target_dir) and not force:
        print(f"CUFSF directory already exists: {target_dir}")
        print("Use --force to re-download")
        return False
    
    print("=" * 60)
    print("CUFSF DATASET DOWNLOAD")
    print("=" * 60)
    print("\nIMPORTANT: CUFSF dataset requires permission from CUHK.")
    print("Please visit: http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html")
    print("\n" + CUFSF_ALTERNATIVE)
    
    print("\nAttempting automated download...")
    
    os.makedirs(target_dir, exist_ok=True)
    photos_dir = os.path.join(target_dir, 'photos')
    sketches_dir = os.path.join(target_dir, 'sketches')
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(sketches_dir, exist_ok=True)
    
    success = False
    for name, url in CUFSF_URLS.items():
        try:
            print(f"\nDownloading {name}...")
            zip_path = os.path.join(target_dir, f'{name}.zip')
            urllib.request.urlretrieve(url, zip_path)
            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            os.remove(zip_path)
            success = True
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            print("Please download manually from the CUHK website.")
    
    if success:
        print("\nCUFSF dataset downloaded successfully!")
        print(f"Location: {target_dir}")
    else:
        print("\nAutomatic download failed. Please download manually.")
        create_placeholder_structure(target_dir)
    
    return success


def create_placeholder_structure(target_dir):
    """Create placeholder directory structure for manual download."""
    print("\nCreating placeholder structure...")
    
    photos_dir = os.path.join(target_dir, 'photos')
    sketches_dir = os.path.join(target_dir, 'sketches')
    
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(sketches_dir, exist_ok=True)
    
    readme_path = os.path.join(target_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("CUFSF Dataset Placeholder\n")
        f.write("=" * 40 + "\n\n")
        f.write("Please download CUFSF dataset from:\n")
        f.write("http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html\n\n")
        f.write("Expected structure:\n")
        f.write("CUFSF/\n")
        f.write("├── photos/\n")
        f.write("│   ├── person_001/\n")
        f.write("│   │   ├── photo_1.jpg\n")
        f.write("│   │   └── photo_2.jpg\n")
        f.write("│   └── ...\n")
        f.write("└── sketches/\n")
        f.write("    ├── person_001_Sz.jpg\n")
        f.write("    └── ...\n")
    
    print(f"\nPlaceholder created at: {target_dir}")
    print("Please populate with CUFSF data after downloading.")


def verify_cufsf(target_dir='data/dataset/CUFSF'):
    """Verify CUFSF dataset structure."""
    photos_dir = os.path.join(target_dir, 'photos')
    sketches_dir = os.path.join(target_dir, 'sketches')
    
    if not os.path.exists(photos_dir) or not os.path.exists(sketches_dir):
        return False, "Missing photos or sketches directory"
    
    import glob
    sketch_files = glob.glob(os.path.join(sketches_dir, '*'))
    if len(sketch_files) == 0:
        return False, "No sketch files found"
    
    photo_dirs = [d for d in glob.glob(os.path.join(photos_dir, '*')) if os.path.isdir(d)]
    if len(photo_dirs) == 0:
        photo_files = glob.glob(os.path.join(photos_dir, '*'))
        if len(photo_files) > 0:
            return True, f"CUFSF found: {len(photo_files)} photos, {len(sketch_files)} sketches (flat structure)"
        return False, "No photos found"
    
    return True, f"CUFSF found: {len(photo_dirs)} photo directories, {len(sketch_files)} sketches"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download/prepare CUFSF dataset')
    parser.add_argument('--target', type=str, default='data/dataset/CUFSF',
                        help='Target directory for CUFSF')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing CUFSF installation')
    
    args = parser.parse_args()
    
    if args.verify:
        success, message = verify_cufsf(args.target)
        if success:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
    else:
        download_cufsf(args.target, args.force)
