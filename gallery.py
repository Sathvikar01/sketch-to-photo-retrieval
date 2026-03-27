import torch
import os
import glob
from PIL import Image
from torchvision import transforms
from model import PseudoSiameseNet
from facenet_pytorch import MTCNN
from tqdm import tqdm

DATA_DIR = 'data/dataset/CUFS/test_photos'
GALLERY_DB_PATH = 'gallery_db.pt'
MODEL_CHECKPOINT = 'checkpoints/pseudo_siamese.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_gallery():
    # Load model
    model = PseudoSiameseNet().to(device)
    if os.path.exists(MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
        print(f"Loaded tuned checkpoint from {MODEL_CHECKPOINT}")
    else:
        print("Warning: No tuned checkpoint found, using base pre-trained weights.")
    
    model.eval()
    
    # Setup MTCNN & Transforms
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    gallery_files = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
    print(f"Found {len(gallery_files)} photos in {DATA_DIR} to embed.")
    
    vector_db = {}
    
    with torch.no_grad():
        for f in tqdm(gallery_files):
            img = Image.open(f).convert('RGB')
            
            img_tensor = mtcnn(img)
            if img_tensor is None:
                img_tensor = transform(img)
                
            # Add batch dim
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Forward pass through branch_photo
            emb = model.forward_photo(img_tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            
            # Store
            # {"photo_id": <tensor_embedding>, "filepath": <string>}
            photo_id = os.path.basename(f)
            vector_db[photo_id] = {
                "embedding": emb.cpu(),
                "filepath": f
            }
            
    # Save the database
    torch.save(vector_db, GALLERY_DB_PATH)
    print(f"Saved {len(vector_db)} embeddings to {GALLERY_DB_PATH}")

if __name__ == '__main__':
    generate_gallery()
