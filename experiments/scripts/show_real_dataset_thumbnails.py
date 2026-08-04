import os
import sys
import torch
import torch.nn as nn
import cv2
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from experiments.models.binary_classifiers import SlideTextMatcher, KeyframeMatcher
from experiments.models.fusion_network import MultimodalSceneEncoder
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using device: {device}")

def main():
    root_out = project_root / "outputs"
    exp_out = project_root / "experiments" / "outputs"
    art_out = Path(r"C:\Users\admin\.gemini\antigravity-ide\brain\7c869227-951a-4d37-9f00-7798b5adedb9")
    
    for p in [root_out, exp_out, art_out]:
        p.mkdir(parents=True, exist_ok=True)
        
    # 1. Load 6 Real JPG Thumbnails from Dataset
    thumb_dir = project_root / "experiments" / "datasets" / "tvsum_extracted" / "tvsum_thumbnails" / "thumbnail"
    jpg_files = sorted(list(thumb_dir.glob("*.jpg")))[:6]
    
    if not jpg_files:
        print("[ERROR] No JPG thumbnails found!")
        return
        
    print(f"[1/4] Found {len(jpg_files)} real JPG thumbnail files in dataset.")
    
    # 2. Load Local TEDLIUM transcripts
    csv_path = Path(r"D:\datasets\TEDLIUM\metadata.csv")
    df = pd.read_csv(csv_path)
    df_clean = df[df['text'] != 'ignore_time_segment_in_scoring'].copy()
    ted_texts = df_clean['text'].tolist()[:6]
    
    # 3. Load Models
    models_dir = project_root / "storage" / "models"
    kf_model = KeyframeMatcher().to(device)
    kf_model.load_state_dict(torch.load(models_dir / "keyframe_matcher.pth", map_location=device))
    kf_model.eval()
    
    slide_model = SlideTextMatcher().to(device)
    slide_model.load_state_dict(torch.load(models_dir / "slide_matcher.pth", map_location=device))
    slide_model.eval()
    
    scene_model = MultimodalSceneEncoder().to(device)
    scene_model.load_state_dict(torch.load(models_dir / "scene_encoder.pth", map_location=device))
    scene_model.eval()
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    proj_text = nn.Linear(384, 768).to(device)
    nn.init.orthogonal_(proj_text.weight)
    
    results = []
    real_jpg_imgs = []
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("REAL DATASET JPG THUMBNAILS & TEDLIUM TRANSCRIPT EVALUATION", fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        jpg_path = jpg_files[i]
        pil_img = Image.open(jpg_path).convert("RGB")
        real_jpg_imgs.append(pil_img)
        
        t_text = ted_texts[i]
        o_text = f"Slide Concept: {t_text[:60]}"
        
        with torch.no_grad():
            inp_v = clip_processor(images=pil_img, return_tensors="pt").to(device)
            v_emb = clip_model.get_image_features(**inp_v)
            v_emb = v_emb / v_emb.norm(dim=-1, keepdim=True)
            
            t_inp = tokenizer(t_text, return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = text_model(**t_inp)
            t_mask = t_inp['attention_mask'].unsqueeze(-1)
            t_pool = (t_out.last_hidden_state * t_mask).sum(dim=1) / t_mask.sum(dim=1)
            t_emb = proj_text(t_pool)
            
            o_inp = tokenizer(o_text, return_tensors="pt", padding=True, truncation=True).to(device)
            o_out = text_model(**o_inp)
            o_mask = o_inp['attention_mask'].unsqueeze(-1)
            o_pool = (o_out.last_hidden_state * o_mask).sum(dim=1) / o_mask.sum(dim=1)
            o_emb = proj_text(o_pool)
            
            score_kf = kf_model(v_emb, t_emb).item()
            score_slide = slide_model(o_emb, t_emb).item()
            
        is_relevant = (score_kf >= 0.50) and (score_slide >= 0.50)
        status_str = "KEEP (RELEVANT)" if is_relevant else "DISCARD (NOISE)"
        color = 'green' if is_relevant else 'red'
        
        ax.imshow(pil_img)
        title = f"JPG Thumbnail: {jpg_path.name}\nKeyframe Match: {score_kf*100:.1f}% | Slide Match: {score_slide*100:.1f}%"
        ax.set_title(title, color=color, fontweight='bold', fontsize=10)
        ax.axis('off')
        
        results.append({
            "jpg_name": jpg_path.name,
            "kf_match": f"{score_kf*100:.1f}%",
            "slide_match": f"{score_slide*100:.1f}%",
            "status": status_str,
            "transcript": t_text
        })
        
    plt.tight_layout()
    
    target_filename = "real_dataset_thumbnails_grid.png"
    for d in [exp_out, root_out, art_out]:
        fig.savefig(str(d / target_filename), dpi=150)
        with open(d / "real_dataset_thumbnails_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    plt.close(fig)
    print(f"[OK] Successfully saved REAL JPG thumbnails grid to {target_filename} in all directories!")

if __name__ == "__main__":
    main()
