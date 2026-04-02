"""
Generate gallery database for sketch-to-photo retrieval.

Supports different splits: train, test, display
Creates separate gallery databases for each purpose.
"""
import torch
import os
import glob
import argparse
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from model import PseudoSiameseNet
from tqdm import tqdm


def generate_gallery(
    data_dir='data/dataset/CUFS_reorganized',
    split='display',
    checkpoint_path='checkpoints/regularized_v1/best_model.pth',
    output_path=None,
    device='cuda'
):
    """
    Generate gallery database from photos.
    
    Args:
        data_dir: Base data directory
        split: Which split to use ('train', 'test', 'display')
        checkpoint_path: Path to model checkpoint
        output_path: Output path for gallery database
        device: Device to use ('cuda' or 'cpu')
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    photos_dir = os.path.join(data_dir, split, 'photos')
    if output_path is None:
        output_path = f'gallery_db_{split}.pt'
    
    print(f"Generating gallery for {split} split...")
    print(f"  Photos directory: {photos_dir}")
    print(f"  Output: {output_path}")
    
    if not os.path.exists(photos_dir):
        print(f"Error: Photos directory not found: {photos_dir}")
        return None
    
    model = PseudoSiameseNet(pretrained='vggface2').to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found, using pretrained weights")
    model.eval()
    
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    vector_db = {}
    
    photo_files = glob.glob(os.path.join(photos_dir, '*'))
    photo_files = [f for f in photo_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    photo_files = sorted(photo_files)
    
    print(f"\nProcessing {len(photo_files)} photos...")
    
    for photo_path in tqdm(photo_files):
        photo_id = os.path.basename(photo_path)
        
        try:
            img = Image.open(photo_path).convert('RGB')
            
            try:
                face = mtcnn(img)
                if face is not None:
                    img_tensor = face
                else:
                    img_tensor = transform(img)
            except Exception:
                img_tensor = transform(img)
            
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model.forward_photo(img_tensor)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            vector_db[photo_id] = {
                'embedding': embedding.squeeze().cpu(),
                'filepath': photo_path
            }
            
        except Exception as e:
            print(f"  Error processing {photo_id}: {e}")
    
    torch.save(vector_db, output_path)
    print(f"\nGallery saved: {output_path}")
    print(f"  Total entries: {len(vector_db)}")
    
    return vector_db


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate gallery database')
    parser.add_argument('--data_dir', type=str, default='data/dataset/CUFS_reorganized',
                        help='Data directory')
    parser.add_argument('--split', type=str, default='display',
                        choices=['train', 'test', 'display'],
                        help='Which split to generate gallery for')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/regularized_v1/best_model.pth',
                        help='Model checkpoint path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: gallery_db_{split}.pt)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--all', action='store_true',
                        help='Generate galleries for all splits')
    
    args = parser.parse_args()
    
    if args.all:
        for split in ['train', 'test', 'display']:
            generate_gallery(
                data_dir=args.data_dir,
                split=split,
                checkpoint_path=args.checkpoint,
                output_path=f'gallery_db_{split}.pt',
                device=args.device
            )
    else:
        generate_gallery(
            data_dir=args.data_dir,
            split=args.split,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            device=args.device
        )
