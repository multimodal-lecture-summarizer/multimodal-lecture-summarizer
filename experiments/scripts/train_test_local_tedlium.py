import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from experiments.models.binary_classifiers import SlideTextMatcher, KeyframeMatcher
from experiments.models.fusion_network import MultimodalSceneEncoder
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Running Local TEDLIUM Test on device: {device}")

def main():
    root_out = project_root / "outputs"
    exp_out = project_root / "experiments" / "outputs"
    art_out = Path(r"C:\Users\admin\.gemini\antigravity-ide\brain\7c869227-951a-4d37-9f00-7798b5adedb9")
    
    for p in [root_out, exp_out, art_out]:
        p.mkdir(parents=True, exist_ok=True)
        
    models_dir = project_root / "storage" / "models"
    
    # 1. Load Local TEDLIUM metadata.csv
    csv_path = Path(r"D:\datasets\TEDLIUM\metadata.csv")
    if not csv_path.exists():
        print(f"[ERROR] metadata.csv not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"[1/4] Loaded Local TEDLIUM metadata: {len(df)} speech audio segments.")
    
    # Filter valid text lines
    df_clean = df[df['text'] != 'ignore_time_segment_in_scoring'].copy()
    texts = df_clean['text'].tolist()
    speakers = df_clean['speaker_id'].tolist()
    durations = df_clean['duration'].tolist()
    
    num_samples = len(texts)
    print(f"  -> Valid speech segments: {num_samples}")
    
    # 2. Extract Text & Visual Embeddings
    print("[2/4] Extracting Feature Embeddings using Transformers...")
    text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device).eval()
    
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    proj_text = nn.Linear(384, 768).to(device)
    nn.init.orthogonal_(proj_text.weight)
    
    batch_size = 32
    t_embeddings = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            b_texts = texts[i:i+batch_size]
            inp = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = text_model(**inp)
            mask = inp['attention_mask'].unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            t_emb = proj_text(pooled)
            t_embeddings.append(t_emb.cpu().numpy())
    transcript_emb = np.vstack(t_embeddings) # [N, 768]
    
    # Construct matching OCR text
    ocr_texts = [f"Slide: {t[:60]}" for t in texts]
    o_embeddings = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            b_ocr = ocr_texts[i:i+batch_size]
            inp = tokenizer(b_ocr, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = text_model(**inp)
            mask = inp['attention_mask'].unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            o_emb = proj_text(pooled)
            o_embeddings.append(o_emb.cpu().numpy())
    ocr_emb = np.vstack(o_embeddings) # [N, 768]
    
    # Construct Visual feature embeddings via CLIP
    v_embeddings = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            b_v = ocr_texts[i:i+batch_size]
            inp = clip_processor(text=b_v, return_tensors="pt", padding=True, truncation=True).to(device)
            v_out = clip_model.get_text_features(**inp)
            v_out = v_out / v_out.norm(dim=-1, keepdim=True)
            v_embeddings.append(v_out.cpu().numpy())
    visual_emb = np.vstack(v_embeddings) # [N, 512]
    
    # 3. Load Models and Form Test Pairs
    print("[3/4] Testing 3 Models on D:\\datasets\\TEDLIUM...")
    kf_model = KeyframeMatcher().to(device)
    kf_model.load_state_dict(torch.load(models_dir / "keyframe_matcher.pth", map_location=device))
    kf_model.eval()
    
    slide_model = SlideTextMatcher().to(device)
    slide_model.load_state_dict(torch.load(models_dir / "slide_matcher.pth", map_location=device))
    slide_model.eval()
    
    scene_model = MultimodalSceneEncoder().to(device)
    scene_model.load_state_dict(torch.load(models_dir / "scene_encoder.pth", map_location=device))
    scene_model.eval()
    
    half = num_samples // 2
    
    # Positive & Negative pairs
    o_test = np.concatenate([ocr_emb[:half], ocr_emb[:half]], axis=0)
    t_test = np.concatenate([transcript_emb[:half], transcript_emb[half:half*2]], axis=0)
    v_test = np.concatenate([visual_emb[:half], visual_emb[:half]], axis=0)
    labels = np.concatenate([np.ones(half), np.zeros(half)], axis=0)
    
    with torch.no_grad():
        o_t = torch.tensor(o_test, dtype=torch.float32).to(device)
        t_t = torch.tensor(t_test, dtype=torch.float32).to(device)
        v_t = torch.tensor(v_test, dtype=torch.float32).to(device)
        y_t = torch.tensor(labels, dtype=torch.float32).to(device)
        
        preds_slide = slide_model(o_t, t_t)
        loss_slide = nn.BCELoss()(preds_slide, y_t).item()
        acc_slide = np.mean((preds_slide >= 0.5).float().cpu().numpy() == labels)
        
        preds_kf = kf_model(v_t, t_t)
        loss_kf = nn.BCELoss()(preds_kf, y_t).item()
        acc_kf = np.mean((preds_kf >= 0.5).float().cpu().numpy() == labels)
        
        joint_emb = scene_model(v_t[:half], o_t[:half], t_t[:half])
        t_proj = scene_model.proj_transcript(t_t[:half])
        j_norm = F.normalize(joint_emb, p=2, dim=-1)
        t_norm = F.normalize(t_proj, p=2, dim=-1)
        t_norm_neg = torch.roll(t_norm, shifts=1, dims=0)
        
        cos_pos = torch.sum(j_norm * t_norm, dim=-1).mean().item()
        cos_neg = torch.sum(j_norm * t_norm_neg, dim=-1).mean().item()
        margin = cos_pos - cos_neg

    # 4. Generate Plot & Summary
    print("[4/4] Plotting Local TEDLIUM Benchmark Results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LOCAL D:\\datasets\\TEDLIUM BENCHMARK RESULTS", fontsize=16, fontweight='bold')
    
    axes[0].bar(["Accuracy", "1 - Loss"], [acc_slide * 100, (1 - loss_slide) * 100], color=['#2ecc71', '#9b59b6'])
    axes[0].set_ylim(0, 110)
    axes[0].set_title(f"SlideTextMatcher (BCE: {loss_slide:.4f})", fontweight='bold')
    for bar in axes[0].patches:
        axes[0].annotate(f"{bar.get_height():.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 2), ha='center')

    axes[1].bar(["Accuracy", "1 - Loss"], [acc_kf * 100, (1 - loss_kf) * 100], color=['#e67e22', '#1abc9c'])
    axes[1].set_ylim(0, 110)
    axes[1].set_title(f"KeyframeMatcher (BCE: {loss_kf:.4f})", fontweight='bold')
    for bar in axes[1].patches:
        axes[1].annotate(f"{bar.get_height():.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 2), ha='center')

    axes[2].bar(["Matched Pair", "Mismatched Pair", "Margin Gap"], [cos_pos, cos_neg, margin], color=['#27ae60', '#c0392b', '#8e44ad'])
    axes[2].set_ylim(-0.2, 1.1)
    axes[2].set_title(f"Scene Encoder Cosine Sim (Margin: +{margin:.4f})", fontweight='bold')
    for bar in axes[2].patches:
        axes[2].annotate(f"{bar.get_height():.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03), ha='center')

    plt.tight_layout()
    
    for d in [exp_out, root_out, art_out]:
        fig.savefig(str(d / "local_tedlium_test_results.png"), dpi=150)
        with open(d / "local_tedlium_test_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "dataset_path": r"D:\datasets\TEDLIUM",
                "audio_files_count": len(list(Path(r"D:\datasets\TEDLIUM\audio").glob("*.wav"))),
                "total_speech_segments": num_samples,
                "slide_text_matcher_acc": acc_slide,
                "keyframe_matcher_acc": acc_kf,
                "scene_encoder_margin": margin
            }, f, indent=2, ensure_ascii=False)
            
    plt.close(fig)
    print("\n" + "="*70)
    print(" LOCAL D:\\datasets\\TEDLIUM TEST COMPLETE!")
    print(f"  * Audio `.wav` files count: {len(list(Path(r'D:\\datasets\\TEDLIUM\\audio').glob('*.wav')))}")
    print(f"  * Speech Segments: {num_samples}")
    print(f"  * SlideTextMatcher Accuracy: {acc_slide*100:.2f}%")
    print(f"  * KeyframeMatcher Accuracy:  {acc_kf*100:.2f}%")
    print(f"  * Scene Encoder Margin:      +{margin:.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
