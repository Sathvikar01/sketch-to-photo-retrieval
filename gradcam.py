import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN

FACIAL_REGIONS = {
    'left_eye': ((45, 55), (75, 85)),
    'right_eye': ((85, 95), (75, 85)),
    'nose': ((60, 100), (95, 115)),
    'mouth': ((55, 105), (125, 145)),
    'left_cheek': ((20, 50), (90, 130)),
    'right_cheek': ((110, 140), (90, 130)),
    'forehead': ((40, 120), (20, 60)),
    'chin': ((55, 105), (140, 158))
}

LANDMARK_MAPPING = {
    'left_eye': [36, 37, 38, 39, 40, 41],
    'right_eye': [42, 43, 44, 45, 46, 47],
    'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35],
    'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
    'left_eyebrow': [17, 18, 19, 20, 21],
    'right_eyebrow': [22, 23, 24, 25, 26],
    'jaw': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
}


def analyze_facial_features(cam_np, threshold=0.3):
    """
    Analyze which facial features contributed most to the match.
    
    Args:
        cam_np: Grad-CAM heatmap (H, W)
        threshold: Minimum activation to consider a region as important
    
    Returns:
        dict: Feature names and their contribution scores
    """
    feature_scores = {}
    h, w = cam_np.shape
    
    for feature_name, ((x1, x2), (y1, y2)) in FACIAL_REGIONS.items():
        x1_adj = int(x1 * w / 160)
        x2_adj = int(x2 * w / 160)
        y1_adj = int(y1 * h / 160)
        y2_adj = int(y2 * h / 160)
        
        x1_adj = max(0, min(w, x1_adj))
        x2_adj = max(0, min(w, x2_adj))
        y1_adj = max(0, min(h, y1_adj))
        y2_adj = max(0, min(h, y2_adj))
        
        if x2_adj > x1_adj and y2_adj > y1_adj:
            region_cam = cam_np[y1_adj:y2_adj, x1_adj:x2_adj]
            if region_cam.size > 0:
                score = float(np.mean(region_cam))
                if score >= threshold:
                    feature_scores[feature_name] = score
    
    sorted_features = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
    return sorted_features


def generate_explanation(feature_scores):
    """
    Generate a human-readable explanation of the match.
    
    Args:
        feature_scores: Dict of feature names and scores
    
    Returns:
        str: Human-readable explanation
    """
    if not feature_scores:
        return "No specific facial features strongly influenced this match."
    
    top_features = list(feature_scores.items())[:3]
    feature_names = ', '.join([f.replace('_', ' ').title() for f, _ in top_features])
    
    explanation = f"Match primarily based on: {feature_names}. "
    
    if len(top_features) == 1:
        explanation += "This feature showed strong similarity."
    else:
        explanation += "These features showed strong similarity."
    
    return explanation


def draw_feature_boxes(img_np, cam_np, threshold=0.3):
    """
    Draw bounding boxes around important facial features.
    
    Args:
        img_np: Original image (H, W, 3), values in [0, 1]
        cam_np: Grad-CAM heatmap (H, W)
        threshold: Minimum activation to highlight
    
    Returns:
        numpy array with drawn boxes (uint8)
    """
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
    
    img_with_boxes = img_np.copy()
    h, w = img_np.shape[:2]
    
    feature_scores = analyze_facial_features(cam_np, threshold)
    
    for feature_name, score in feature_scores.items():
        (x1, x2), (y1, y2) = FACIAL_REGIONS[feature_name]
        x1_adj = int(x1 * w / 160)
        x2_adj = int(x2 * w / 160)
        y1_adj = int(y1 * h / 160)
        y2_adj = int(y2 * h / 160)
        
        intensity = min(1.0, score / 0.5)
        color = (0, int(255 * intensity), int(255 * (1 - intensity)))
        
        cv2.rectangle(img_with_boxes, (x1_adj, y1_adj), (x2_adj, y2_adj), color, 2)
        
        label = feature_name.replace('_', ' ').title()
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img_with_boxes, (x1_adj, y1_adj - label_size[1] - 5), 
                      (x1_adj + label_size[0], y1_adj), color, -1)
        cv2.putText(img_with_boxes, label, (x1_adj, y1_adj - 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img_with_boxes


def blend_heatmap(img_np, cam_np, alpha=0.5):
    """
    Overlays Grad-CAM heatmap on the original image.
    
    Args:
        img_np: numpy array of original image (RGB), values in [0, 1] or [0, 255]
        cam_np: 2D numpy array of Grad-CAM activations, values in [0, 1]
        alpha: blending factor
    
    Returns:
        Blended image (uint8)
    """
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 1)
        img_np = np.uint8(255 * img_np)
    
    cam_np = np.clip(cam_np, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(img_np, 1-alpha, heatmap, alpha, 0)
    return blended


class CosineSimilarityGradCAM:
    """
    Original Grad-CAM for cosine similarity.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, photo_tensor, sketch_embedding):
        """
        Calculates Grad-CAM based on cosine similarity.
        
        Args:
            photo_tensor: (1, 3, 160, 160) input candidate photo
            sketch_embedding: (1, 512) frozen query sketch embedding
        """
        self.model.eval()
        photo_tensor.requires_grad_(True)
        
        self.model.zero_grad()
        
        emb_p = self.model.forward_photo(photo_tensor)
        emb_p = F.normalize(emb_p, p=2, dim=1)
        
        similarity = torch.sum(sketch_embedding * emb_p)
        
        similarity.backward()
        
        assert self.gradients is not None
        assert self.activations is not None
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        b, k, u, v = gradients.size()
        
        alpha = gradients.view(b, k, -1).mean(2)
        
        weights = alpha.view(b, k, 1, 1)
        
        cam = (weights * activations).sum(1, keepdim=True)
        
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam = F.interpolate(cam, size=(160, 160), mode='bilinear', align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        
        return cam_np, similarity.item()


class LC3GradCAM:
    """
    Landmark-Constrained Contrastive Cross-Modal Grad-CAM (LC3-GradCAM).
    
    Novel explanation method for cross-modal sketch-to-photo retrieval that:
    1. Generates attribution maps on BOTH sketch and photo branches
    2. Provides positive (why match) and negative (why not match) explanations
    3. Uses landmark detection for region-specific scoring
    
    This is the main contribution for the research paper.
    """
    
    def __init__(self, model, target_layer_photo, target_layer_sketch=None):
        """
        Initialize LC3-GradCAM.
        
        Args:
            model: PseudoSiameseNet model
            target_layer_photo: Target convolutional layer for photo branch
            target_layer_sketch: Target convolutional layer for sketch branch (optional)
        """
        self.model = model
        self.target_layer_photo = target_layer_photo
        self.target_layer_sketch = target_layer_sketch or target_layer_photo
        
        self.photo_gradients = None
        self.photo_activations = None
        self.sketch_gradients = None
        self.sketch_activations = None
        
        self.target_layer_photo.register_forward_hook(self._save_photo_activation)
        self.target_layer_photo.register_full_backward_hook(self._save_photo_gradient)
        
        if self.target_layer_sketch != self.target_layer_photo:
            self.target_layer_sketch.register_forward_hook(self._save_sketch_activation)
            self.target_layer_sketch.register_full_backward_hook(self._save_sketch_gradient)
        
        self.mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device='cpu')
    
    def _save_photo_activation(self, module, input, output):
        self.photo_activations = output
    
    def _save_photo_gradient(self, module, grad_input, grad_output):
        self.photo_gradients = grad_output[0]
    
    def _save_sketch_activation(self, module, input, output):
        self.sketch_activations = output
    
    def _save_sketch_gradient(self, module, grad_input, grad_output):
        self.sketch_gradients = grad_output[0]
    
    def _compute_cam(self, gradients, activations):
        """Compute CAM from gradients and activations."""
        b, k, u, v = gradients.size()
        
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def _get_landmarks(self, img_tensor):
        """
        Extract facial landmarks using MTCNN.
        
        Args:
            img_tensor: (1, 3, 160, 160) tensor in [-1, 1] range
        
        Returns:
            landmarks: dict of landmark points or None
        """
        try:
            img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            boxes, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)
            
            if landmarks is not None and len(landmarks) > 0 and landmarks[0] is not None:
                return landmarks[0]
        except Exception:
            pass
        
        return None
    
    def compute_positive_attribution(self, sketch_tensor, photo_tensor):
        """
        Compute positive attribution map (why this photo matches the sketch).
        
        Args:
            sketch_tensor: (1, 3, 160, 160) sketch input
            photo_tensor: (1, 3, 160, 160) photo input
        
        Returns:
            photo_cam: numpy array (160, 160)
            sketch_cam: numpy array (160, 160) 
            similarity: float
        """
        self.model.eval()
        photo_tensor = photo_tensor.clone().requires_grad_(True)
        sketch_tensor = sketch_tensor.clone().requires_grad_(True)
        
        self.model.zero_grad()
        
        with torch.no_grad():
            sketch_emb = self.model.forward_sketch(sketch_tensor)
            sketch_emb = F.normalize(sketch_emb, p=2, dim=1)
        
        self.photo_gradients = None
        self.photo_activations = None
        
        photo_emb = self.model.forward_photo(photo_tensor)
        photo_emb = F.normalize(photo_emb, p=2, dim=1)
        
        similarity = torch.sum(sketch_emb.detach() * photo_emb)
        similarity.backward()
        
        photo_cam = self._compute_cam(self.photo_gradients, self.photo_activations)
        photo_cam = F.interpolate(photo_cam, size=(160, 160), mode='bilinear', align_corners=False)
        photo_cam_np = photo_cam.squeeze().cpu().numpy()
        
        return photo_cam_np, similarity.item()
    
    def compute_negative_attribution(self, sketch_tensor, photo_tensor):
        """
        Compute negative attribution map (why this photo does NOT match the sketch).
        Highlights features that would need to change for a better match.
        
        Args:
            sketch_tensor: (1, 3, 160, 160) sketch input
            photo_tensor: (1, 3, 160, 160) photo input
        
        Returns:
            neg_cam: numpy array (160, 160)
            dissimilarity: float
        """
        self.model.eval()
        photo_tensor = photo_tensor.clone().requires_grad_(True)
        
        self.model.zero_grad()
        
        with torch.no_grad():
            sketch_emb = self.model.forward_sketch(sketch_tensor)
            sketch_emb = F.normalize(sketch_emb, p=2, dim=1)
        
        self.photo_gradients = None
        self.photo_activations = None
        
        photo_emb = self.model.forward_photo(photo_tensor)
        photo_emb = F.normalize(photo_emb, p=2, dim=1)
        
        similarity = torch.sum(sketch_emb.detach() * photo_emb)
        neg_similarity = -similarity
        neg_similarity.backward()
        
        neg_cam = self._compute_cam(self.photo_gradients, self.photo_activations)
        neg_cam = F.interpolate(neg_cam, size=(160, 160), mode='bilinear', align_corners=False)
        neg_cam_np = neg_cam.squeeze().cpu().numpy()
        
        return neg_cam_np, similarity.item()
    
    def compute_dual_attribution(self, sketch_tensor, photo_tensor):
        """
        Compute attribution maps on BOTH branches simultaneously.
        This shows which sketch strokes correspond to which photo regions.
        
        Args:
            sketch_tensor: (1, 3, 160, 160) sketch input
            photo_tensor: (1, 3, 160, 160) photo input
        
        Returns:
            photo_cam: numpy array (160, 160)
            sketch_cam: numpy array (160, 160)
            similarity: float
        """
        self.model.eval()
        photo_tensor = photo_tensor.clone().requires_grad_(True)
        sketch_tensor = sketch_tensor.clone().requires_grad_(True)
        
        self.model.zero_grad()
        
        self.photo_gradients = None
        self.photo_activations = None
        self.sketch_gradients = None
        self.sketch_activations = None
        
        sketch_emb = self.model.forward_sketch(sketch_tensor)
        sketch_emb = F.normalize(sketch_emb, p=2, dim=1)
        
        photo_emb = self.model.forward_photo(photo_tensor)
        photo_emb = F.normalize(photo_emb, p=2, dim=1)
        
        similarity = torch.sum(sketch_emb * photo_emb)
        similarity.backward()
        
        photo_cam = self._compute_cam(self.photo_gradients, self.photo_activations)
        photo_cam = F.interpolate(photo_cam, size=(160, 160), mode='bilinear', align_corners=False)
        photo_cam_np = photo_cam.squeeze().cpu().numpy()
        
        if self.sketch_gradients is not None:
            sketch_cam = self._compute_cam(self.sketch_gradients, self.sketch_activations)
            sketch_cam = F.interpolate(sketch_cam, size=(160, 160), mode='bilinear', align_corners=False)
            sketch_cam_np = sketch_cam.squeeze().cpu().numpy()
        else:
            sketch_cam_np = np.zeros((160, 160))
        
        return photo_cam_np, sketch_cam_np, similarity.item()
    
    def analyze_landmark_regions(self, cam_np, landmarks):
        """
        Analyze CAM using detected landmarks for precise region scoring.
        
        Args:
            cam_np: (H, W) Grad-CAM heatmap
            landmarks: landmark points from MTCNN
        
        Returns:
            region_scores: dict mapping facial regions to importance scores
        """
        if landmarks is None:
            return analyze_facial_features(cam_np)
        
        h, w = cam_np.shape
        region_scores = {}
        
        for region_name, indices in LANDMARK_MAPPING.items():
            try:
                region_points = landmarks[indices]
                points = region_points.reshape(-1, 2)
                
                x_coords = (points[:, 0] * w / 160).astype(int)
                y_coords = (points[:, 1] * h / 160).astype(int)
                
                x_min, x_max = max(0, x_coords.min()), min(w, x_coords.max())
                y_min, y_max = max(0, y_coords.min()), min(h, y_coords.max())
                
                if x_max > x_min and y_max > y_min:
                    region_cam = cam_np[y_min:y_max, x_min:x_max]
                    if region_cam.size > 0:
                        score = float(np.mean(region_cam))
                        if score > 0.1:
                            region_scores[region_name] = score
            except Exception:
                continue
        
        return dict(sorted(region_scores.items(), key=lambda x: x[1], reverse=True))
    
    def __call__(self, sketch_tensor, photo_tensor, mode='positive'):
        """
        Main entry point for computing attribution maps.
        
        Args:
            sketch_tensor: (1, 3, 160, 160) sketch input
            photo_tensor: (1, 3, 160, 160) photo input
            mode: 'positive', 'negative', or 'dual'
        
        Returns:
            Tuple based on mode
        """
        if mode == 'positive':
            return self.compute_positive_attribution(sketch_tensor, photo_tensor)
        elif mode == 'negative':
            return self.compute_negative_attribution(sketch_tensor, photo_tensor)
        elif mode == 'dual':
            return self.compute_dual_attribution(sketch_tensor, photo_tensor)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'positive', 'negative', or 'dual'.")


def visualize_lc3_comparison(sketch_np, photo_np, photo_pos_cam, photo_neg_cam, 
                              sketch_cam=None, alpha=0.5):
    """
    Create a side-by-side comparison visualization for LC3-GradCAM.
    
    Args:
        sketch_np: Sketch image (H, W, 3), uint8
        photo_np: Photo image (H, W, 3), uint8
        photo_pos_cam: Positive attribution map (H, W)
        photo_neg_cam: Negative attribution map (H, W)
        sketch_cam: Optional sketch attribution map (H, W)
        alpha: Blending factor
    
    Returns:
        Combined visualization image (H, W*3 or H, W*4)
    """
    if sketch_np.dtype != np.uint8:
        sketch_np = (np.clip(sketch_np, 0, 1) * 255).astype(np.uint8)
    if photo_np.dtype != np.uint8:
        photo_np = (np.clip(photo_np, 0, 1) * 255).astype(np.uint8)
    
    photo_pos_blend = blend_heatmap(photo_np, photo_pos_cam, alpha)
    photo_neg_blend = blend_heatmap(photo_np, photo_neg_cam, alpha)
    
    cv2.putText(photo_pos_blend, "Why Match", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(photo_neg_blend, "Why Not Match", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if sketch_cam is not None:
        sketch_blend = blend_heatmap(sketch_np, sketch_cam, alpha)
        combined = np.hstack([sketch_np, sketch_blend, photo_pos_blend, photo_neg_blend])
    else:
        combined = np.hstack([sketch_np, photo_pos_blend, photo_neg_blend])
    
    return combined


if __name__ == '__main__':
    print("LC3-GradCAM module loaded successfully.")
    print("Main classes:")
    print("  - CosineSimilarityGradCAM: Original Grad-CAM")
    print("  - LC3GradCAM: Landmark-Constrained Contrastive Cross-Modal Grad-CAM")
