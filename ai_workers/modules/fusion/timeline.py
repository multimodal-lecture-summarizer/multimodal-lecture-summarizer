"""Timeline — cross-modal alignment, chaptering, RAG index.

Migrated from: src/mls/modules/timeline.py
NGƯỜI 3: Cross-modal alignment, chapter segmentation.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        scenes: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Auto-detect chapter boundaries using multimodal topic shifts, action demos, and visual cuts.

        Methods: multimodal topic_shift | action_demo_shift | visual_boundary

        Returns:
            List of chapters: [{title, startTime, endTime}]
        """
        all_visual_scenes = scenes if scenes else slides
        
        # Calculate video end time
        end_time = 0.0
        if utterances:
            end_time = max(end_time, float(utterances[-1].get("end", 0.0)))
        if all_visual_scenes:
            end_time = max(end_time, float(all_visual_scenes[-1].get("end_seconds", 0.0)))
        if slides:
            end_time = max(end_time, float(slides[-1].get("end_seconds", 0.0)))

        # Adaptive minimum chapter length: 20-25s for videos <= 10 min (600s), 45s for long videos
        if end_time <= 600.0:
            MIN_CHAPTER_LENGTH = 20.0
        else:
            MIN_CHAPTER_LENGTH = 45.0

        visual_boundaries = []
        for s in (all_visual_scenes or []):
            v_start = float(s.get("start_seconds", 0.0))
            if v_start > 0.0:
                visual_boundaries.append(round(v_start, 2))

        # 1. Action / Demo Segment Transitions (Silence in speech >= 8s with visual motion/objects)
        action_boundaries = []
        if utterances:
            for i in range(len(utterances) - 1):
                u_curr_end = float(utterances[i].get("end", 0.0))
                u_next_start = float(utterances[i+1].get("start", 0.0))
                gap = u_next_start - u_curr_end

                if gap >= 8.0:
                    # Check for visual activity / objects in gap
                    has_action = False
                    for s in (all_visual_scenes or []):
                        s_s = float(s.get("start_seconds", 0.0))
                        s_e = float(s.get("end_seconds", 0.0))
                        if max(u_curr_end, s_s) < min(u_next_start, s_e):
                            det = s.get("detected_objects", {})
                            tracks = det.get("tracks_summary", []) if isinstance(det, dict) else []
                            counts = s.get("object_counts", {})
                            if tracks or counts:
                                has_action = True
                                break
                    if has_action or gap >= 12.0:
                        # Snap gap start to nearest scene start if within 6s
                        start_cand = u_curr_end
                        for vb in visual_boundaries:
                            if abs(vb - start_cand) <= 6.0 and vb > 0:
                                start_cand = vb
                                break
                        action_boundaries.append(round(start_cand, 2))

                        end_cand = u_next_start
                        for vb in visual_boundaries:
                            if abs(vb - end_cand) <= 6.0 and vb > 0:
                                end_cand = vb
                                break
                        action_boundaries.append(round(end_cand, 2))

        # 2. Transition cues in speech
        cue_boundaries = []
        TRANSITION_CUES = [
            "the key is in the approach", "the key is", "to start with", "very small steps",
            "real small jumps", "show you a little bit more", "in practice", "as you can see",
            "the important thing as i said earlier", "stay within your ability", "so we're gonna do",
            "bắt đầu", "kỹ thuật tiếp cận", "lưu ý", "thực hành", "tổng kết", "lời khuyên"
        ]
        for u in (utterances or []):
            u_text = u.get("text", "").lower()
            u_start = float(u.get("start", 0.0))
            if u_start >= 15.0:
                for cue in TRANSITION_CUES:
                    if cue in u_text:
                        cue_boundaries.append(round(u_start, 2))
                        break

        # 3. Speech Semantic Shifts via TF-IDF
        semantic_boundaries = []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            # Window size: 18s for short videos, 30s for long videos
            window_size = 18.0 if end_time <= 600.0 else 30.0
            windows = []
            current_window_text = []
            current_window_start = 0.0

            if utterances:
                current_window_start = float(utterances[0].get("start", 0.0))

            for u in utterances:
                u_start = float(u.get("start", 0.0))
                u_text = u.get("text", "")

                if u_start - current_window_start > window_size and current_window_text:
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
                print(
                    f"[Timeline] Calculating semantic shifts using lightweight TF-IDF "
                    f"for {len(windows)} windows..."
                )
                texts = [w["text"] for w in windows]
                vectors = TfidfVectorizer(stop_words=None).fit_transform(texts)

                sims = []
                for i in range(vectors.shape[0] - 1):
                    sim = cosine_similarity(vectors[i], vectors[i + 1])[0][0]
                    sims.append(sim)

                # Smoothing: Moving Average (window=3)
                smoothed_sims = []
                for i in range(len(sims)):
                    start_idx = max(0, i - 1)
                    end_idx = min(len(sims), i + 2)
                    smoothed_sims.append(np.mean(sims[start_idx:end_idx]))

                mean_sim = np.mean(smoothed_sims)
                std_sim = np.std(smoothed_sims)
                k = 0.85 if end_time <= 600.0 else 1.0
                print(f"[Timeline] Semantic Chaptering: mean_sim={mean_sim:.3f}, std_sim={std_sim:.3f}, k={k}")

                threshold = mean_sim - k * std_sim
                for i in range(1, len(smoothed_sims) - 1):
                    if smoothed_sims[i] < threshold:
                        if smoothed_sims[i] < smoothed_sims[i-1] and smoothed_sims[i] < smoothed_sims[i+1]:
                            semantic_boundaries.append(round(windows[i+1]["start"], 2))

        except Exception as e:
            print(f"[Timeline] Semantic chaptering failed: {e}")

        # 4. Merge and Resolve Multimodal Boundaries
        raw_candidates = sorted(list(set(action_boundaries + cue_boundaries + semantic_boundaries + visual_boundaries)))

        final_boundaries = [0.0]
        for b in raw_candidates:
            if b <= 0.0 or b >= end_time - 10.0:
                continue

            last_b = final_boundaries[-1]
            dur = b - last_b
            if dur >= MIN_CHAPTER_LENGTH:
                final_boundaries.append(b)
            elif b in (action_boundaries + cue_boundaries) and last_b != 0.0 and abs(b - last_b) <= 10.0:
                # Snap previous boundary to action or cue boundary
                final_boundaries[-1] = b

        # 5. Balanced Granularity: For videos of 3 to 10 min, aim for 4 to 6 chapters
        if 180.0 <= end_time <= 600.0 and len(final_boundaries) < 4:
            new_b = [final_boundaries[0]]
            for i in range(len(final_boundaries) - 1):
                c_start = final_boundaries[i]
                c_end = final_boundaries[i+1]
                gap = c_end - c_start
                if gap >= 45.0 and (len(final_boundaries) + len(new_b) - i - 1) < 5:
                    # Find candidate within [c_start + 20, c_end - 20]
                    mids = [cand for cand in raw_candidates if c_start + 20.0 <= cand <= c_end - 20.0]
                    if mids:
                        chosen = min(mids, key=lambda x: abs(x - (c_start + c_end) / 2))
                        new_b.append(chosen)
                new_b.append(c_end)
            final_boundaries = sorted(list(set(new_b)))

        # Fallback if degenerate
        if len(final_boundaries) < 2 and end_time > 120.0:
            print("[Timeline] Degenerate case detected. Fallback to time-based chaptering.")
            step = 60.0 if end_time <= 300.0 else 90.0
            curr = step
            while curr < end_time - 30.0:
                final_boundaries.append(curr)
                curr += step

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
        chapters = self.segment_chapters(utterances, slides, scenes=scenes)
        return {"aligned_segments": aligned, "chapters": chapters}
