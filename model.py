import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class PseudoSiameseNet(nn.Module):
    def __init__(self, pretrained='vggface2'):
        super(PseudoSiameseNet, self).__init__()
        
        # Load pre-trained models for each branch
        self.branch_sketch = InceptionResnetV1(pretrained=pretrained)
        self.branch_photo = InceptionResnetV1(pretrained=pretrained)
        
        # For a pseudo-siamese architecture, we want some layers unshared and some shared.
        # Since we want to capture modality-specific low-level features, we keep the early layers unshared.
        # We'll share the higher level layers.
        
        # Share weights for 'repeat_3', 'block8', 'avgpool', 'dropout', 'last_linear', 'last_bn'
        # Note: This is a design choice. You could also keep them completely unshared or 
        # completely shared (standard Siamese).
        
        # Here we choose to share the later parts of the network
        self.branch_sketch.repeat_3 = self.branch_photo.repeat_3
        self.branch_sketch.block8 = self.branch_photo.block8
        self.branch_sketch.avgpool_1a = self.branch_photo.avgpool_1a
        self.branch_sketch.last_linear = self.branch_photo.last_linear
        self.branch_sketch.last_bn = self.branch_photo.last_bn
        
    def forward_sketch(self, x):
        return self.branch_sketch(x)

    def forward_photo(self, x):
        return self.branch_photo(x)
        
    def forward(self, sketch, photo, neg_photo=None):
        # We return normalized embeddings directly 
        emb_sketch = self.forward_sketch(sketch)
        emb_photo = self.forward_photo(photo)
        
        # L2 Normalize
        emb_sketch = torch.nn.functional.normalize(emb_sketch, p=2, dim=1)
        emb_photo = torch.nn.functional.normalize(emb_photo, p=2, dim=1)
        
        if neg_photo is not None:
            emb_neg_photo = self.forward_photo(neg_photo)
            emb_neg_photo = torch.nn.functional.normalize(emb_neg_photo, p=2, dim=1)
            return emb_sketch, emb_photo, emb_neg_photo
            
        return emb_sketch, emb_photo

if __name__ == '__main__':
    model = PseudoSiameseNet()
    print("Model created successfully.")
    
    # Dummy tensors
    s = torch.randn(2, 3, 160, 160)
    p = torch.randn(2, 3, 160, 160)
    
    s_emb, p_emb = model(s, p)
    print(f"Sketch embedding shape: {s_emb.shape}")
    print(f"Photo embedding shape: {p_emb.shape}")
