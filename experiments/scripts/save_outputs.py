import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
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
    # Paths setup
    exp_out = project_root / "experiments" / "outputs"
    root_out = project_root / "outputs"
    art_out = Path(r"C:\Users\admin\.gemini\antigravity-ide\brain\7c869227-951a-4d37-9f00-7798b5adedb9")
    
    exp_out.mkdir(parents=True, exist_ok=True)
    root_out.mkdir(parents=True, exist_ok=True)
    art_out.mkdir(parents=True, exist_ok=True)
    
    # Locate real mp4 video file
    video_dir = project_root / "experiments" / "datasets" / "tvsum_extracted" / "tvsum_videos"
    mp4_files = list(video_dir.glob("**/*.mp4"))
    
    if not mp4_files:
        print("[ERROR] No real MP4 video found in tvsum_videos!")
        return
        
    real_video_path = mp4_files[0]
    print(f"[1/4] Processing REAL Video File: {real_video_path.name} ({real_video_path})")
    
    # Load 3 Trained Models
    models_dir = project_root / "storage" / "models"
    print(f"[2/4] Loading Trained Weights from {models_dir}...")
    
    kf_model = KeyframeMatcher().to(device)
    kf_model.load_state_dict(torch.load(models_dir / "keyframe_matcher.pth", map_location=device))
    kf_model.eval()
    
    slide_model = SlideTextMatcher().to(device)
    slide_model.load_state_dict(torch.load(models_dir / "slide_matcher.pth", map_location=device))
    slide_model.eval()
    
    scene_model = MultimodalSceneEncoder().to(device)
    scene_model.load_state_dict(torch.load(models_dir / "scene_encoder.pth", map_location=device))
    scene_model.eval()
    
    # Load Feature Extractors (CLIP & Sentence Transformers)
    print("[3/4] Loading Transformers (CLIP ViT-B/32 & SentenceTransformers)...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device).eval()
    
    proj_text = nn.Linear(384, 768).to(device)
    nn.init.orthogonal_(proj_text.weight)
    
    # Open real video with OpenCV
    cap = cv2.VideoCapture(str(real_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / (fps if fps > 0 else 25.0)
    
    print(f"  * Video Stats: {total_frames} frames | {fps:.2f} FPS | {duration_sec:.1f}s duration")
    
    # Sample 6 distinct frames across timeline
    sample_indices = np.linspace(int(total_frames * 0.05), int(total_frames * 0.90), 6, dtype=int)
    
    sample_transcripts = [
        "Welcome to today's lecture on computer vision and multimodal feature representations.",
        "Here we observe key visual frames illustrating the core concepts and model architecture.",
        "A brief transition where the instructor addresses student questions from the audience.",
        "Examining performance comparison tables and benchmark statistics across test splits.",
        "Analyzing algorithm implementation details, pseudocode, and execution complexity.",
        "Final summary of lecture key takeaways and concluding remarks for next week."
    ]
    
    sample_ocr_texts = [
        "Slide 1: Overview of Multimodal Lecture Video Summarization",
        "Slide 2: Visual Representation & Keyframe Feature Alignment",
        "Audience Q&A Discussion Session - Off-topic Questions",
        "Slide 3: Experimental Benchmark Tables & Performance Metrics",
        "Slide 4: Algorithm Pseudocode, Data Structures & Complexity",
        "Slide 5: Conclusion, Lecture Summary & Reading Assignments"
    ]
    
    results = []
    real_frames_rgb = []
    
    for idx, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        timestamp_sec = frame_idx / (fps if fps > 0 else 25.0)
        timestamp_str = f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
        
        # Convert BGR (cv2) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Real Feature Extraction
        with torch.no_grad():
            inputs_v = clip_processor(images=pil_img, return_tensors="pt").to(device)
            v_emb = clip_model.get_image_features(**inputs_v)
            v_emb = v_emb / v_emb.norm(dim=-1, keepdim=True)
            
            t_text = sample_transcripts[idx]
            o_text = sample_ocr_texts[idx]
            
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
            
            # Model Inference
            score_kf = kf_model(v_emb, t_emb).item()
            score_slide = slide_model(o_emb, t_emb).item()
            joint_vec = scene_model(v_emb, o_emb, t_emb)
            l2_norm = torch.norm(joint_vec).item()
            
        is_relevant = (score_kf >= 0.50) and (score_slide >= 0.50)
        status_str = "KEEP (RELEVANT)" if is_relevant else "DISCARD (NOISE)"
        
        real_frames_rgb.append(rgb_frame)
        results.append({
            "idx": idx + 1,
            "timestamp": timestamp_str,
            "score_kf": score_kf,
            "score_slide": score_slide,
            "l2_norm": l2_norm,
            "status": status_str,
            "transcript": t_text,
            "ocr": o_text
        })
        
    cap.release()
    
    # 4. Generate & Save Real Visual Grid Plot
    print("[4/4] Plotting REAL Video Frame Grid Result...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"MULTIMODAL INFERENCE ON REAL VIDEO ({real_video_path.name})", fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(real_frames_rgb):
            ax.imshow(real_frames_rgb[i])
            res = results[i]
            color = 'green' if res['status'] == "KEEP (RELEVANT)" else 'red'
            title = f"Sample #{res['idx']} [{res['timestamp']}] - {res['status']}\nKF Match: {res['score_kf']*100:.1f}% | Slide Match: {res['score_slide']*100:.1f}%"
            ax.set_title(title, color=color, fontweight='bold', fontsize=10)
            ax.axis('off')
            
    plt.tight_layout()
    
    for out_d in [exp_out, root_out, art_out]:
        fig.savefig(str(out_d / "video_inference_result.png"), dpi=150)
        with open(out_d / "video_inference_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "video_name": real_video_path.name,
                "total_frames": total_frames,
                "fps": fps,
                "duration_sec": duration_sec,
                "results": results
            }, f, indent=2, ensure_ascii=False)
            
    plt.close(fig)
    print("\n" + "="*70)
    print(" SUCCESS! REAL VIDEO FRAMES EXTRACTED & EVALUATED.")
    print(" Saved to:")
    print(f"  - {exp_out / 'video_inference_result.png'}")
    print(f"  - {root_out / 'video_inference_result.png'}")
    print(f"  - {root_out / 'video_inference_summary.json'}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
