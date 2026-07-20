"""Timeline — cross-modal alignment, chaptering, RAG index.

Migrated from: src/mls/modules/timeline.py
NGƯỜI 3: Cross-modal alignment, chapter segmentation.
"""

from __future__ import annotations

from typing import Any


class TimelineBuilder:
    """Cross-modal alignment and chapter segmentation."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def align_modalities(
        self,
        utterances: list[dict],
        scenes: list[dict],
        slides: list[dict],
    ) -> list[dict[str, Any]]:
        """Align transcript utterances with slide/scene timestamps.

        Uses cross-modal similarity (CLIP embeddings) or timestamp proximity.

        Returns:
            List of aligned segments.
        """
        if not utterances or not scenes:
            return []
            
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Combine caption and OCR text for each scene
            scene_texts = []
            for scene in scenes:
                text = scene.get("caption", "") + " " + scene.get("ocr_text", "")
                scene_texts.append(text)
                
            vectorizer = TfidfVectorizer(stop_words=None)
            # Fit on both scenes and utterances to share vocabulary
            all_texts = scene_texts + [u.get("text", "") for u in utterances]
            vectorizer.fit(all_texts)
            
            scene_vectors = vectorizer.transform(scene_texts)
            
            aligned_segments = []
            for utt in utterances:
                utt_text = utt.get("text", "")
                utt_start = utt.get("start", 0.0)
                utt_end = utt.get("end", 0.0)
                utt_mid = (utt_start + utt_end) / 2
                
                utt_vector = vectorizer.transform([utt_text])
                
                # Calculate semantic similarity
                sim_scores = cosine_similarity(utt_vector, scene_vectors)[0]
                
                # Calculate temporal proximity score (0 to 1)
                time_scores = []
                for scene in scenes:
                    scene_start = scene.get("start_seconds", 0.0)
                    scene_end = scene.get("end_seconds", float('inf'))
                    
                    # If utterance is inside the scene temporally, score is 1.0
                    if scene_start <= utt_mid <= scene_end:
                        time_scores.append(1.0)
                    else:
                        # Decay based on distance from the scene boundaries
                        dist = min(abs(utt_mid - scene_start), abs(utt_mid - scene_end))
                        # e.g., drops to 0.5 at 30 seconds away
                        time_scores.append(max(0, 1.0 - (dist / 60.0))) 
                
                time_scores = np.array(time_scores)
                
                # Combined score: 70% time, 30% semantic to ensure we don't jump around too much
                final_scores = (0.7 * time_scores) + (0.3 * sim_scores)
                best_match_idx = np.argmax(final_scores)
                
                # Only match if there is a reasonable score
                if final_scores[best_match_idx] > 0.1:
                    aligned_segments.append({
                        "utterance": utt,
                        "scene_id": id(scenes[best_match_idx]),
                        "score": float(final_scores[best_match_idx])
                    })
                    
            print(f"[Timeline] Successfully aligned {len(aligned_segments)} utterances to scenes.")
            return aligned_segments
        except Exception as e:
            print(f"[Timeline] Semantic alignment failed: {e}. Returning empty alignment.")
            return []

    def segment_chapters(
        self,
        utterances: list[dict],
        slides: list[dict],
    ) -> list[dict[str, Any]]:
        """Auto-detect chapter boundaries.

        Methods: topic_shift | fixed_window | slide_boundary

        Returns:
            List of chapters: [{chapter_id, title, start_sec, end_sec}]
        """
        # Find visual boundaries
        visual_boundaries = []
        for slide in slides:
            v_start = slide.get("start_seconds")
            if v_start is not None and v_start > 0:
                visual_boundaries.append(v_start)
        
        # Find semantic boundaries
        semantic_boundaries = []
        
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # 1. Chunk utterances into windows (~30s windows)
            windows = []
            current_window_text = []
            current_window_start = 0.0
            
            if utterances:
                current_window_start = utterances[0].get("start", 0.0)
                
            for u in utterances:
                u_start = u.get("start", 0.0)
                u_text = u.get("text", "")
                
                if u_start - current_window_start > 30.0 and current_window_text:
                    windows.append({
                        "start": current_window_start,
                        "text": " ".join(current_window_text)
                    })
                    current_window_text = [u_text]
                    current_window_start = u_start
                else:
                    current_window_text.append(u_text)
                    
            if current_window_text:
                windows.append({
                    "start": current_window_start,
                    "text": " ".join(current_window_text)
                })
                
            if len(windows) > 2:
                print(f"[Timeline] Calculating semantic shifts using {len(windows)} windows...")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                texts = [w["text"] for w in windows]
                embeddings = model.encode(texts)
                
                # Calculate consecutive similarities
                sims = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                    sims.append(sim)
                    
                # Smoothing: Moving Average (window=3)
                smoothed_sims = []
                for i in range(len(sims)):
                    start_idx = max(0, i - 1)
                    end_idx = min(len(sims), i + 2)
                    smoothed_sims.append(np.mean(sims[start_idx:end_idx]))
                    
                # Dynamic Threshold
                mean_sim = np.mean(smoothed_sims)
                std_sim = np.std(smoothed_sims)
                k = 1.0 # Hyperparameter
                print(f"[Timeline] Semantic Chaptering: mean_sim={mean_sim:.3f}, std_sim={std_sim:.3f}, k={k}")
                
                threshold = mean_sim - k * std_sim
                
                # Find local minima below threshold
                for i in range(1, len(smoothed_sims) - 1):
                    if smoothed_sims[i] < threshold:
                        if smoothed_sims[i] < smoothed_sims[i-1] and smoothed_sims[i] < smoothed_sims[i+1]:
                            semantic_boundaries.append(windows[i+1]["start"])
                            
        except Exception as e:
            print(f"[Timeline] Semantic chaptering failed: {e}")
            
        # Merge semantic and visual boundaries
        all_boundaries = sorted(list(set(semantic_boundaries + visual_boundaries)))
        
        # Conflict Resolution and Minimum Chapter Length Enforcement
        MIN_CHAPTER_LENGTH = 60.0
        final_boundaries = [0.0]
        
        for b in all_boundaries:
            if b <= 0.0:
                continue
                
            last_b = final_boundaries[-1]
            if b - last_b < MIN_CHAPTER_LENGTH:
                # If it's too close, snap to visual if one is visual, else drop the newer one
                is_b_visual = b in visual_boundaries
                is_last_visual = last_b in visual_boundaries
                
                if is_b_visual and not is_last_visual and last_b != 0.0:
                    # Snap previous semantic boundary to this visual boundary
                    final_boundaries[-1] = b
                # Otherwise ignore b
            else:
                final_boundaries.append(b)
                
        # Get video end time
        end_time = 0.0
        if utterances:
            end_time = max(end_time, utterances[-1].get("end", 0.0))
        if slides:
            end_time = max(end_time, slides[-1].get("end_seconds", 0.0))
            
        # Degenerate Fallback: If < 2 chapters, slice by time
        if len(final_boundaries) < 2 and end_time > 180.0:
            print("[Timeline] Degenerate case detected. Fallback to time-based chaptering.")
            final_boundaries = [0.0]
            curr = 180.0
            while curr < end_time - 60.0:
                final_boundaries.append(curr)
                curr += 180.0
                
        # Construct candidate intervals
        chapters = []
        for i in range(len(final_boundaries)):
            start = final_boundaries[i]
            end = final_boundaries[i+1] if i + 1 < len(final_boundaries) else end_time
            if end - start > 0:
                chapters.append({
                    "startTime": start,
                    "endTime": end
                })
                
        print(f"[Timeline] Generated {len(chapters)} chapter boundaries.")
        return chapters

    def process(
        self,
        utterances: list[dict],
        scenes: list[dict],
        slides: list[dict],
    ) -> dict[str, Any]:
        """Full timeline pipeline: align → segment chapters."""
        aligned = self.align_modalities(utterances, scenes, slides)
        chapters = self.segment_chapters(utterances, slides)
        return {"aligned_segments": aligned, "chapters": chapters}
