from __future__ import annotations

import os
import gc
import cv2
import numpy as np
import torch
from typing import Any
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

class SemanticAnalyzer:
    """Vision encoding (CLIP), semantic filtering, and captioning (Florence-2) for keyframes."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.keep_ratio = self.config.get("keep_ratio", 0.7)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def filter_scenes_clip(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract CLIP embeddings, perform zero-shot filtering, cluster, and score (V3)."""
        from sklearn.cluster import AgglomerativeClustering
        
        valid_scenes = [s for s in scenes if s.get("keyframe_path") and os.path.exists(s["keyframe_path"])]
        num_frames = len(valid_scenes)
        
        if num_frames == 0:
            return []
            
        processor = None
        model = None
        
        # Configuration Constants for V3
        ABS_THRESHOLD = 0.22
        OVERLAY_PENALTY_FACTOR = 0.3
        DEDUP_THRESHOLD = 0.92
        MIN_GAP_FRAMES = 20
        
        def detect_overlay_region(img_gray):
            """Detect if there is abnormal edge density in corners or bottom, suggesting overlay/watermark."""
            h, w = img_gray.shape
            edges = cv2.Canny(img_gray, 50, 150)
            
            margin_h, margin_w = int(h * 0.15), int(w * 0.15)
            bottom_third = int(h * 0.66)
            
            regions = [
                edges[0:margin_h, 0:margin_w],
                edges[0:margin_h, w-margin_w:w],
                edges[h-margin_h:h, 0:margin_w],
                edges[h-margin_h:h, w-margin_w:w],
                edges[bottom_third:h, int(w*0.2):int(w*0.8)] # bottom center
            ]
            
            center_region = edges[margin_h:h-margin_h, margin_w:w-margin_w]
            center_density = np.mean(center_region) if center_region.size > 0 else 0
            
            for region in regions:
                if region.size > 0:
                    density = np.mean(region)
                    # If local density is extremely high or much higher than center, likely a text overlay
                    if density > 50 or (center_density > 0 and density / center_density > 3.0):
                        return True
            return False
        
        try:
            print(f"[Semantic] Loading CLIP model to extract embeddings for {num_frames} frames...")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # Prompts for zero-shot filtering and action scoring
            text_prompts = [
                "a real photograph of people, objects, actions, or a scene",
                "a graphic, title card, logo, slide, or text on a plain background",
                "a person actively performing a task, demonstration, or specific action"
            ]
            text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Extract embeddings and quality metrics
            images = [Image.open(s["keyframe_path"]).convert("RGB") for s in valid_scenes]
            batch_size = 32
            all_embeddings = []
            zero_shot_sims = []
            
            entropies = []
            laplacians = []
            has_overlay = []
            
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
                    
                    sims = torch.matmul(image_features, text_features.T).cpu().numpy()
                    zero_shot_sims.extend(sims)
                    
            embeddings = np.vstack(all_embeddings)
            
            # Calculate quality metrics
            for s in valid_scenes:
                img = cv2.imread(s["keyframe_path"], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    laplacians.append(cv2.Laplacian(img, cv2.CV_64F).var())
                    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                    hist = hist.ravel() / hist.sum()
                    hist = hist[hist > 0]
                    entropies.append(-np.sum(hist * np.log2(hist)))
                    has_overlay.append(detect_overlay_region(img))
                else:
                    laplacians.append(0.0)
                    entropies.append(0.0)
                    has_overlay.append(False)
                    
            # Normalize quality metrics per-video
            laplacians = np.array(laplacians)
            entropies = np.array(entropies)
            lap_range = laplacians.max() - laplacians.min() + 1e-6
            ent_range = entropies.max() - entropies.min() + 1e-6
            laplacian_norm = (laplacians - laplacians.min()) / lap_range
            entropy_norm = (entropies - entropies.min()) / ent_range
            
            zero_shot_sims = np.array(zero_shot_sims)
            action_scores = zero_shot_sims[:, 2]
            action_range = action_scores.max() - action_scores.min() + 1e-6
            action_norm = (action_scores - action_scores.min()) / action_range
            
            quality_scores = 0.5 * entropy_norm + 0.5 * laplacian_norm
            # Apply overlay penalty
            for i in range(num_frames):
                if has_overlay[i]:
                    quality_scores[i] *= OVERLAY_PENALTY_FACTOR
            
            # Step 1: Hard filter out "graphics/logos"
            filtered_indices = []
            for i in range(num_frames):
                sim_photo = zero_shot_sims[i, 0]
                sim_graphic = zero_shot_sims[i, 1]
                is_full_graphic = (sim_graphic > sim_photo) and (sim_photo < ABS_THRESHOLD)
                if not is_full_graphic:
                    filtered_indices.append(i)
            
            print(f"[Semantic] Hard filter removed {num_frames - len(filtered_indices)} logo/graphic scenes.")
            if not filtered_indices:
                filtered_indices = list(range(num_frames))
                
            clean_embeddings = embeddings[filtered_indices]
            clean_indices = filtered_indices
            
            if len(clean_indices) > 1000:
                print(f"[Semantic] Warning: {len(clean_indices)} frames is large. Agglomerative clustering may be slow.")
                
            # Step 2: Semantic Clustering
            threshold = self.config.get("similarity_threshold", 0.82)
            distance_threshold = 1.0 - threshold
            
            if len(clean_indices) > 1:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    metric='cosine',
                    linkage='average',
                    distance_threshold=distance_threshold
                )
                labels = clustering.fit_predict(clean_embeddings)
            else:
                labels = np.array([0])
                
            clusters_map = {}
            for idx_in_clean, label in enumerate(labels):
                original_idx = clean_indices[idx_in_clean]
                clusters_map.setdefault(label, []).append(original_idx)
                
            # Step 3: Temporal Splitting
            sub_clusters = []
            gap_threshold = max(int(0.15 * num_frames), MIN_GAP_FRAMES)
            
            for label, indices in clusters_map.items():
                indices.sort()
                current_sub = [indices[0]]
                for i in range(1, len(indices)):
                    if indices[i] - indices[i-1] > gap_threshold:
                        sub_clusters.append(current_sub)
                        current_sub = [indices[i]]
                    else:
                        current_sub.append(indices[i])
                sub_clusters.append(current_sub)
                
            print(f"[Semantic] Formed {len(sub_clusters)} temporal sub-clusters after splitting.")
            
            # Step 4: Representative Selection & Importance Scoring
            max_sub_cluster_size = max(len(sc) for sc in sub_clusters) if sub_clusters else 1
            
            representatives = []
            for sc in sub_clusters:
                best_idx = max(sc, key=lambda idx: quality_scores[idx])
                
                size_score = np.log1p(len(sc)) / np.log1p(max_sub_cluster_size)
                q_score = quality_scores[best_idx]
                a_score = action_norm[best_idx]
                
                importance = 0.25 * size_score + 0.45 * a_score + 0.30 * q_score
                
                representatives.append({
                    "original_idx": best_idx,
                    "importance": importance,
                    "embedding": embeddings[best_idx]
                })
                
            # Step 5: Global Deduplication
            dedup_selection = []
            # Sort by importance descending so we keep the most important one in case of conflict
            representatives.sort(key=lambda x: x["importance"], reverse=True)
            
            for rep in representatives:
                is_duplicate = False
                for selected in dedup_selection:
                    sim = np.dot(rep["embedding"], selected["embedding"])
                    if sim > DEDUP_THRESHOLD:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    dedup_selection.append(rep)
                    
            # Step 6: Top-K Cut and Final Sort
            max_keyframes = self.config.get("max_keyframes", 15) # Default lowered to 15 for better distillation
            
            if len(dedup_selection) > max_keyframes:
                dedup_selection = dedup_selection[:max_keyframes]
                
            dedup_selection.sort(key=lambda x: x["original_idx"])
            
            filtered_scenes = []
            for sel in dedup_selection:
                scene = valid_scenes[sel["original_idx"]]
                # Scale final importance from [0.2, 1.0] for UI aesthetics
                final_score = 0.2 + (sel["importance"] * 0.8)
                scene["importanceScore"] = round(final_score, 2)
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

    def caption_scenes_florence2(self, scenes: list[dict[str, Any]]):
        """Generate detailed captions for scenes using Florence-2."""
        if not scenes:
            return
            
        processor = None
        model = None
        
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "florence2_vendor")
            print(f"[Semantic] Loading Florence-2 model to caption {len(scenes)} frames from {model_dir}...")
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(self.device)
            
            # <CAPTION> generates a very short, concise sentence focused on the main subject,
            # eliminating unnecessary details like backgrounds or skies.
            task_prompt = "<CAPTION>"
            
            for scene in scenes:
                path = scene.get("keyframe_path")
                if not path or not os.path.exists(path):
                    continue
                    
                try:
                    raw_image = Image.open(path).convert("RGB")
                    inputs = processor(text=task_prompt, images=raw_image, return_tensors="pt").to(self.device, dtype)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            num_beams=3
                        )
                        
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(raw_image.width, raw_image.height))
                    
                    # The answer is a dict, we extract the task value
                    caption = parsed_answer.get(task_prompt, "")
                    if not isinstance(caption, str):
                        caption = str(caption)
                        
                    caption = caption.strip()
                    # Remove robotic prefixes to make it sound natural
                    lower_cap = caption.lower()
                    prefixes = [
                        "the image shows a ", "the image shows an ", "the image shows ", 
                        "the image is a ", "the image is an ", "the image is ",
                        "this image shows a ", "this image shows an ", "this image shows ",
                        "this is a picture of a ", "this is a picture of an ", "this is a picture of ",
                        "this is a ", "this is an "
                    ]
                    for prefix in prefixes:
                        if lower_cap.startswith(prefix):
                            caption = caption[len(prefix):]
                            if caption:
                                caption = caption[0].upper() + caption[1:]
                            break
                    
                    scene["caption"] = caption
                        
                except Exception as e:
                    print(f"[Semantic] Error generating Florence-2 caption for {path}: {e}")
                    if "caption" not in scene:
                        scene["caption"] = "Image"
        finally:
            print("[Semantic] Releasing Florence-2 model from memory...")
            if model is not None:
                del model
            if processor is not None:
                del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[Semantic] Florence-2 model released successfully.")
        
    def extract_ocr_paddleocr(self, scenes: list[dict[str, Any]]):
        """Extract text from scenes using PaddleOCR."""
        if not scenes:
            return
            
        try:
            print("[Semantic] Loading PaddleOCR model...")
            # We import here to avoid slow loading if not needed
            from paddleocr import PaddleOCR
            import logging
            logging.getLogger("ppocr").setLevel(logging.WARNING) # Suppress noisy logs
            
            try:
                ocr = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=torch.cuda.is_available())
            except Exception as init_err:
                print(f"[Semantic] Fallback init for PaddleOCR: {init_err}")
                ocr = PaddleOCR(lang='vi')
            
            for scene in scenes:
                path = scene.get("keyframe_path")
                if not path or not os.path.exists(path):
                    scene["ocr_text"] = ""
                    continue
                
                try:
                    result = ocr.ocr(path, cls=True)
                    text_blocks = []
                    if result and result[0]:
                        for line in result[0]:
                            text_blocks.append(line[1][0])
                    
                    ocr_text = " | ".join(text_blocks)
                    scene["ocr_text"] = ocr_text.strip()
                except Exception as e:
                    print(f"[Semantic] OCR Error for {path}: {e}")
                    scene["ocr_text"] = ""
                    
            print(f"[Semantic] OCR extraction completed for {len(scenes)} scenes.")
        except Exception as e:
            print(f"[Semantic] Failed to load or run PaddleOCR: {e}")
            for scene in scenes:
                scene["ocr_text"] = ""
                
    def process(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Full semantic pipeline: CLIP filtering -> Florence-2 captioning -> OCR."""
        # Step 1: Filter redundant scenes using CLIP
        filtered_scenes = self.filter_scenes_clip(scenes)
        
        # Step 2: Generate rich captions for the remaining scenes
        self.caption_scenes_florence2(filtered_scenes)
        
        # Step 3: Extract text from slides via OCR
        self.extract_ocr_paddleocr(filtered_scenes)
        
        return filtered_scenes
