import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from experiments.models.binary_classifiers import SlideTextMatcher, KeyframeMatcher
from experiments.models.fusion_network import MultimodalSceneEncoder
from experiments.scripts.train_binary_classifiers import BinaryMatchingDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_binary_model(model, data_loader, name="Model"):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for batch in data_loader:
            a = batch["a"].to(device)
            t = batch["transcript"].to(device)
            y = batch["label"].to(device)
            
            preds = model(a, t)
            loss = criterion(preds, y)
            total_loss += loss.item() * a.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    bce_loss = total_loss / len(data_loader.dataset)
    binary_preds = (np.array(all_preds) >= 0.5).astype(np.float32)
    y_true = np.array(all_labels)
    
    accuracy = np.mean(binary_preds == y_true)
    tp = np.sum((binary_preds == 1) & (y_true == 1))
    fp = np.sum((binary_preds == 1) & (y_true == 0))
    fn = np.sum((binary_preds == 0) & (y_true == 1))
    tn = np.sum((binary_preds == 0) & (y_true == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print(f"==================================================")
    print(f" RESULTS: {name}")
    print(f"==================================================")
    print(f"  * BCE Loss:           {bce_loss:.4f}")
    print(f"  * Accuracy:           {accuracy * 100:.2f}%")
    print(f"  * Precision:          {precision * 100:.2f}%")
    print(f"  * Recall:             {recall * 100:.2f}%")
    print(f"  * F1-Score:           {f1 * 100:.2f}%")
    print(f"  * Confusion Matrix:   TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"==================================================\n")

def evaluate_fusion_model(model, name="MultimodalSceneEncoder"):
    model.eval()
    np.random.seed(42)
    num_samples = 200
    clip_dim = 512
    text_dim = 768
    
    base_pos = np.random.randn(num_samples, text_dim).astype(np.float32)
    t_pos = base_pos + 0.1 * np.random.randn(num_samples, text_dim).astype(np.float32)
    o_pos = base_pos + 0.2 * np.random.randn(num_samples, text_dim).astype(np.float32)
    
    proj_w = np.random.randn(text_dim, clip_dim).astype(np.float32) / np.sqrt(text_dim)
    v_pos = np.dot(base_pos, proj_w) + 0.05 * np.random.randn(num_samples, clip_dim).astype(np.float32)
    
    with torch.no_grad():
        v_tensor = torch.tensor(v_pos, dtype=torch.float32).to(device)
        o_tensor = torch.tensor(o_pos, dtype=torch.float32).to(device)
        t_tensor = torch.tensor(t_pos, dtype=torch.float32).to(device)
        
        joint_emb = model(v_tensor, o_tensor, t_tensor)
        
        joint_norm = F.normalize(joint_emb, p=2, dim=-1)
        t_proj = F.normalize(model.proj_transcript(t_tensor), p=2, dim=-1)
        
        cos_sim_pos = torch.sum(joint_norm * t_proj, dim=-1).mean().item()
        
        t_proj_neg = torch.roll(t_proj, shifts=1, dims=0)
        cos_sim_neg = torch.sum(joint_norm * t_proj_neg, dim=-1).mean().item()
        
        print(f"==================================================")
        print(f" RESULTS: {name}")
        print(f"==================================================")
        print(f"  * Output Shape:                   {tuple(joint_emb.shape)}")
        print(f"  * Mean L2 Norm:                   {torch.norm(joint_emb, dim=-1).mean().item():.4f}")
        print(f"  * Cosine Sim (Matched Pair):      {cos_sim_pos:.4f}")
        print(f"  * Cosine Sim (Mismatched Pair):   {cos_sim_neg:.4f}")
        print(f"  * Margin Difference:              {(cos_sim_pos - cos_sim_neg):.4f}")
        print(f"==================================================\n")

def main():
    models_dir = project_root / "storage" / "models"
    
    print("\n" + "="*60)
    print(" TESTING ALL 3 MULTIMODAL MODELS")
    print(f" Device: {device}")
    print(f" Weights Directory: {models_dir}")
    print("="*60 + "\n")
    
    # 1. Test SlideTextMatcher
    full_ds_text = BinaryMatchingDataset(num_samples=1200, is_visual=False)
    _, val_ds_text = torch.utils.data.random_split(full_ds_text, [1000, 200])
    val_loader_text = torch.utils.data.DataLoader(val_ds_text, batch_size=32, shuffle=False)
    
    slide_model = SlideTextMatcher().to(device)
    slide_path = models_dir / "slide_matcher.pth"
    if slide_path.exists():
        slide_model.load_state_dict(torch.load(slide_path, map_location=device))
        evaluate_binary_model(slide_model, val_loader_text, name="Component 1: SlideTextMatcher")
    else:
        print(f"[ERROR] Weight file not found: {slide_path}")

    # 2. Test KeyframeMatcher
    full_ds_vis = BinaryMatchingDataset(num_samples=1200, is_visual=True)
    _, val_ds_vis = torch.utils.data.random_split(full_ds_vis, [1000, 200])
    val_loader_vis = torch.utils.data.DataLoader(val_ds_vis, batch_size=32, shuffle=False)
    
    kf_model = KeyframeMatcher().to(device)
    kf_path = models_dir / "keyframe_matcher.pth"
    if kf_path.exists():
        kf_model.load_state_dict(torch.load(kf_path, map_location=device))
        evaluate_binary_model(kf_model, val_loader_vis, name="Component 2: KeyframeMatcher")
    else:
        print(f"[ERROR] Weight file not found: {kf_path}")
        
    # 3. Test MultimodalSceneEncoder
    scene_model = MultimodalSceneEncoder().to(device)
    scene_path = models_dir / "scene_encoder.pth"
    if scene_path.exists():
        scene_model.load_state_dict(torch.load(scene_path, map_location=device))
        evaluate_fusion_model(scene_model, name="Component 3: MultimodalSceneEncoder")
    else:
        print(f"[ERROR] Weight file not found: {scene_path}")

if __name__ == "__main__":
    main()
