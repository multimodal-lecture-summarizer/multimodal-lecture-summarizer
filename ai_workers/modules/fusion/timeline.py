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
        # TODO: topic shift detection + LLM title generation
        return []

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
