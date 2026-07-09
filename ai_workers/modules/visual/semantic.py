"""Semantic analysis — OCR + vision encoding per keyframe.

Migrated from: src/mls/modules/semantic.py
NGƯỜI 2: CLIP Filtering, BLIP-2 Captioning, OCR (TODO)
"""

from __future__ import annotations

import os
import gc
import numpy as np
import torch
from typing import Any
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans

class SemanticAnalyzer:
    """Vision encoding (CLIP), semantic filtering, and captioning (BLIP-2) for keyframes."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.keep_ratio = self.config.get("keep_ratio", 0.7)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def filter_scenes_clip(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract CLIP embeddings and filter redundant scenes using K-Means."""
        valid_scenes = [s for s in scenes if s.get("keyframe_path") and os.path.exists(s["keyframe_path"])]
        num_frames = len(valid_scenes)
        
        if num_frames == 0:
            return []
            
        processor = None
        model = None
        
        try:
            print(f"[Semantic] Loading CLIP model to extract embeddings for {num_frames} frames...")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # Extract embeddings
            images = [Image.open(s["keyframe_path"]).convert("RGB") for s in valid_scenes]
            batch_size = 32
            all_embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i + batch_size]
                    inputs = processor(images=batch_images, return_tensors="pt").to(self.device)
                    pixel_values = inputs["pixel_values"]
                    vision_outputs = model.vision_model(pixel_values=pixel_values)
                    pooler_output = vision_outputs.pooler_output if hasattr(vision_outputs, "pooler_output") else vision_outputs[1]
                    image_features = model.visual_projection(pooler_output)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    all_embeddings.append(image_features.cpu().numpy())
                    
            embeddings = np.vstack(all_embeddings)
            
            # Clustering: Use configurable keep_ratio (default 0.7) and a high safety cap (50) instead of a hardcap of 12.
            target_clusters = max(1, int(num_frames * self.keep_ratio))
            max_keyframes = self.config.get("max_keyframes", 50)
            num_clusters = min(num_frames, min(max_keyframes, target_clusters))
            if num_clusters == num_frames:
                # Assign uniform importance
                for s in valid_scenes:
                    s["importanceScore"] = 1.0
                return valid_scenes
                
            print(f"[Semantic] Filtering scenes {num_frames} -> {num_clusters} using K-Means...")
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            
            selected_indices = []
            for i in range(num_clusters):
                cluster_center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(embeddings - cluster_center, axis=1)
                sorted_indices = np.argsort(distances)
                for idx in sorted_indices:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        break
                        
            selected_indices.sort()
            
            filtered_scenes = []
            for idx in selected_indices:
                scene = valid_scenes[idx]
                scene["importanceScore"] = 0.8 
                filtered_scenes.append(scene)
                
            print(f"[Semantic] Kept {len(filtered_scenes)} semantically distinct scenes.")
            return filtered_scenes
        finally:
            print("[Semantic] Releasing CLIP model from memory...")
            if model is not None:
                del model
            if processor is not None:
                del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[Semantic] CLIP model released successfully.")

    def caption_scenes_blip(self, scenes: list[dict[str, Any]]):
        """Generate captions for scenes using BLIP."""
        if not scenes:
            return
            
        processor = None
        model = None
        
        try:
            print(f"[Semantic] Loading BLIP model to caption {len(scenes)} frames...")
            model_name = "Salesforce/blip-image-captioning-base"
            processor = BlipProcessor.from_pretrained(model_name)
            
            # Load in float16 for T4/VRAM optimization
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = BlipForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=dtype,
            ).to(self.device)
            
            prompt = "a video scene showing"
            
            for scene in scenes:
                path = scene.get("keyframe_path")
                if not path or not os.path.exists(path):
                    continue
                    
                try:
                    raw_image = Image.open(path).convert("RGB")
                    inputs = processor(raw_image, text=prompt, return_tensors="pt").to(self.device, dtype)
                      
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs, 
                            max_new_tokens=30,
                            min_new_tokens=5,
                            repetition_penalty=1.5
                        )
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    
                    scene["caption"] = generated_text
                except Exception as e:
                    print(f"[Semantic] Error generating caption for {path}: {e}")
                    if "caption" not in scene:
                        scene["caption"] = "Image"
        finally:
            print("[Semantic] Releasing BLIP model from memory...")
            if model is not None:
                del model
            if processor is not None:
                del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[Semantic] BLIP model released successfully.")
        
    def process(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Full semantic pipeline: CLIP filtering -> BLIP captioning."""
        # Step 1: Filter redundant scenes using CLIP
        filtered_scenes = self.filter_scenes_clip(scenes)
        
        # Step 2: Generate rich captions for the remaining scenes
        self.caption_scenes_blip(filtered_scenes)
        
        return filtered_scenes
