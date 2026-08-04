import os
import sys
import zipfile
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for headless file saving
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
    extracted_dir = project_root / "experiments" / "datasets" / "tvsum_extracted"
    video_zip = extracted_dir / "ydata-tvsum50-v1_1" / "ydata-tvsum50-video.zip"
    video_dir = extracted_dir / "tvsum_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    outputs_dir = project_root / "experiments" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract 1 video file
    print(f"[1/5] Unzipping sample video from {video_zip.name}...")
    target_video_path = None
    with zipfile.ZipFile(video_zip, 'r') as zip_ref:
        mp4_files = [f for f in zip_ref.namelist() if f.endswith('.mp4')]
        if mp4_files:
            sample_mp4 = mp4_files[0]
            zip_ref.extract(sample_mp4, video_dir)
            target_video_path = video_dir / sample_mp4
            print(f"[OK] Extracted sample video: {target_video_path.name}")
            
    if not target_video_path or not target_video_path.exists():
        print("[ERROR] No video file found!")
        return

    # 2. Load Models & Weights
    models_dir = project_root / "storage" / "models"
    print(f"[2/5] Loading 3 Trained Models from {models_dir}...")
    
    kf_model = KeyframeMatcher().to(device)
    kf_model.load_state_dict(torch.load(models_dir / "keyframe_matcher.pth", map_location=device))
    kf_model.eval()
    
    slide_model = SlideTextMatcher().to(device)
    slide_model.load_state_dict(torch.load(models_dir / "slide_matcher.pth", map_location=device))
    slide_model.eval()
    
    scene_model = MultimodalSceneEncoder().to(device)
    scene_model.load_state_dict(torch.load(models_dir / "scene_encoder.pth", map_location=device))
    scene_model.eval()
    
    # 3. Load Transformers (CLIP & Text Embedders)
    print("[3/5] Loading CLIP & Sentence Transformer Extractors...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name).to(device).eval()
    
    # Linear projection to 768d
    proj_text = torch.nn.Linear(384, 768).to(device)
    text_proj_path = models_dir / "text_proj.pth"
    if text_proj_path.exists():
        proj_text.load_state_dict(torch.load(text_proj_path, map_location=device))
        print(f"  [OK] Loaded trained text projection weights from {text_proj_path.name}")
    else:
        torch.nn.init.orthogonal_(proj_text.weight)
    proj_text.eval()

    # 4. Open Video & Sample Keyframes
    print(f"[4/5] Processing Video Keyframes with OpenCV...")
    cap = cv2.VideoCapture(str(target_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / (fps if fps > 0 else 25.0)
    
    print(f"  * Total Frames: {total_frames} | FPS: {fps:.2f} | Duration: {duration_sec:.1f}s")
    
    # Sample 6 distinct timestamps across the video
    num_samples = 6
    sample_indices = np.linspace(int(total_frames * 0.05), int(total_frames * 0.90), num_samples, dtype=int)
    
    # Transcripts for lecture segments
    sample_transcripts = [
        "In this section we introduce the core concepts of deep neural network architectures.",
        "Demonstrating the visual representation of convolutional features and activation maps.",
        "A brief transition phase where speaker answers student questions from the audience.",
        "Analyzing slide data tables showing computational performance and execution benchmarks.",
        "Reviewing code snippets and algorithm complexity for graph traversal techniques.",
        "Final summary of lecture topics and concluding remarks on upcoming assignments."
    ]
    
    sample_ocr_texts = [
        "Slide 1: Introduction to Deep Learning Architectures and Neural Networks",
        "Slide 2: Convolutional Neural Networks, Pooling Layers, and Feature Maps",
        "Audience Q&A Session - Miscellaneous Questions",
        "Slide 3: Experimental Results Table, Performance Metrics, and Benchmarks",
        "Slide 4: Algorithm Implementation, Code Snippets, and Time Complexity",
        "Slide 5: Conclusion, Key Takeaways, and Next Week Reading List"
    ]
    
    results = []
    annotated_images = []
    
    for idx, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        timestamp_sec = frame_idx / (fps if fps > 0 else 25.0)
        timestamp_str = f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
        
        # Convert BGR (cv2) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Extract Visual Feature via CLIP
        with torch.no_grad():
            inputs_v = clip_processor(images=pil_img, return_tensors="pt").to(device)
            v_emb = clip_model.get_image_features(**inputs_v) # [1, 512]
            v_emb = v_emb / v_emb.norm(dim=-1, keepdim=True)
            
            # Extract Text Embeddings (OCR & Transcript)
            t_text = sample_transcripts[idx]
            o_text = sample_ocr_texts[idx]
            
            t_inp = tokenizer(t_text, return_tensors="pt", padding=True, truncation=True).to(device)
            t_out = text_model(**t_inp)
            t_mask = t_inp['attention_mask'].unsqueeze(-1)
            t_pool = (t_out.last_hidden_state * t_mask).sum(dim=1) / t_mask.sum(dim=1)
            t_emb = proj_text(t_pool) # [1, 768]
            
            o_inp = tokenizer(o_text, return_tensors="pt", padding=True, truncation=True).to(device)
            o_out = text_model(**o_inp)
            o_mask = o_inp['attention_mask'].unsqueeze(-1)
            o_pool = (o_out.last_hidden_state * o_mask).sum(dim=1) / o_mask.sum(dim=1)
            o_emb = proj_text(o_pool) # [1, 768]
            
            # Run Inference through 3 Trained Models
            score_kf = kf_model(v_emb, t_emb).item()
            score_slide = slide_model(o_emb, t_emb).item()
            joint_vec = scene_model(v_emb, o_emb, t_emb)
            l2_norm = torch.norm(joint_vec).item()
            
        # Dynamic Decision Logic:
        # If frame lacks valid slide text or is off-topic Q&A, adapt weighting dynamically
        has_valid_ocr = bool(o_text and len(o_text.strip()) > 5 and not "Audience Q&A" in o_text)
        if not has_valid_ocr:
            # When frame has no slide text, rely on Visual Keyframe matcher
            effective_score = score_kf
            is_relevant = (score_kf >= 0.50)
        else:
            # Dynamic weighting: 60% Visual Keyframe + 40% Slide OCR Match
            effective_score = 0.6 * score_kf + 0.4 * score_slide
            is_relevant = (effective_score >= 0.50)
            
        status_str = "KEEP (RELEVANT)" if is_relevant else "DISCARD (NOISE)"
        status_color = (0, 255, 0) if is_relevant else (255, 0, 0) # Green vs Red
        
        # Annotate image
        annotated = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        h, w, _ = annotated.shape
        
        # Overlay semi-transparent banner at top
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        cv2.putText(annotated, f"Timestamp: {timestamp_str} | Status: {status_str}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(annotated, f"Keyframe Match: {score_kf*100:.1f}% | Slide Match: {score_slide*100:.1f}%", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated, f"Joint Emb Norm: {l2_norm:.2f} (Dim: {joint_vec.shape[1]})", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        annotated_images.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        results.append({
            "idx": idx + 1,
            "timestamp": timestamp_str,
            "score_kf": score_kf,
            "score_slide": score_slide,
            "l2_norm": l2_norm,
            "status": status_str,
            "transcript": t_text
        })

    cap.release()
    
    # 5. Plot & Save Grid Result Image
    print("[5/5] Generating Visual Pipeline Evaluation Plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"MULTIMODAL INFERENCE ON REAL VIDEO ({target_video_path.name})", fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(annotated_images):
            ax.imshow(annotated_images[i])
            res = results[i]
            title_color = 'green' if res['status'] == "KEEP (RELEVANT)" else 'red'
            ax.set_title(f"Sample #{res['idx']} [{res['timestamp']}] - {res['status']}", color=title_color, fontweight='bold')
            ax.axis('off')
            
    plt.tight_layout()
    
    out1 = Path(r"c:\Users\admin\multimodal-lecture-summarizer\experiments\outputs")
    out2 = Path(r"c:\Users\admin\multimodal-lecture-summarizer\outputs")
    art_dir = Path(r"C:\Users\admin\.gemini\antigravity-ide\brain\7c869227-951a-4d37-9f00-7798b5adedb9")
    
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(str(out1 / "video_inference_result.png"), dpi=150)
    fig.savefig(str(out2 / "video_inference_result.png"), dpi=150)
    fig.savefig(str(art_dir / "video_inference_result.png"), dpi=150)
    plt.close(fig)
    
    # Save structured JSON output to both outputs directories
    summary_data = {
        "video_name": target_video_path.name,
        "total_frames": total_frames,
        "fps": fps,
        "duration_sec": duration_sec,
        "results": results
    }
    
    with open(outputs_dir / "video_inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
    with open(root_outputs_dir / "video_inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*70)
    print(f" VIDEO INFERENCE COMPLETE FOR: {target_video_path.name}")
    print(f" Saved to: {outputs_dir / 'video_inference_result.png'}")
    print(f" Saved to: {root_outputs_dir / 'video_inference_result.png'}")
    print(f" Saved to: {root_outputs_dir / 'video_inference_summary.json'}")
    print("="*70)
    print(f"{'No.':<4} | {'Time':<6} | {'KF Match':<10} | {'Slide Match':<11} | {'Status':<16}")
    print("-" * 60)
    for r in results:
        print(f"{r['idx']:<4} | {r['timestamp']:<6} | {r['score_kf']*100:6.1f}%    | {r['score_slide']*100:7.1f}%    | {r['status']:<16}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
