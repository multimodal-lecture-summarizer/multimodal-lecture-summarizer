from __future__ import annotations

import os
import cv2
from typing import Any
from scenedetect import ContentDetector


class SceneDetector:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.detector = self.config.get("detector", "content")
        self.threshold = self.config.get("threshold", 27.0)

    def detect_scenes(self, video_path: str) -> list[dict[str, Any]]:
        """Detect scene boundaries in video using PySceneDetect.

        Returns:
            List of scenes: [{scene_index, start_seconds, end_seconds, start_timecode, end_timecode, start_frame, end_frame}]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
            
        print(f"Detecting scenes using PySceneDetect for: {video_path}...")
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        # Detect scenes with 4-frame skipping for 5x speedup
        scene_manager.detect_scenes(video, frame_skip=4)
        scene_list = scene_manager.get_scene_list()
        
        scenes = []
        for idx, (start_time, end_time) in enumerate(scene_list):
            scenes.append({
                "scene_index": idx,
                "start_seconds": start_time.get_seconds(),
                "end_seconds": end_time.get_seconds(),
                "start_timecode": start_time.get_timecode(),
                "end_timecode": end_time.get_timecode(),
                "start_frame": start_time.get_frames(),
                "end_frame": end_time.get_frames(),
            })
        print(f"[OK] Detected {len(scenes)} scenes.")
        return scenes

    def extract_keyframes(
        self,
        video_path: str,
        scenes: list[dict],
        output_dir: str,
        strategy: str = "sharpest",
    ) -> list[str]:
        """Extract optimal keyframe images for each scene based on sharpness and quality.

        Args:
            strategy: "first" | "middle" | "sharpest" | "smart"

        Returns:
            List of keyframe image paths.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []
            
        keyframes = []
        
        job_id = os.path.basename(output_dir)
        static_dir = os.path.abspath(os.path.join(os.getcwd(), "storage", "mock_r2_bucket", "keyframes", job_id))
        os.makedirs(static_dir, exist_ok=True)
        
        for scene in scenes:
            start_f = scene["start_frame"]
            end_f = scene["end_frame"]
            total_scene_frames = max(1, end_f - start_f + 1)
            
            if strategy in ("sharpest", "smart"):
                # Sample 5 to 7 candidate frames evenly distributed across the scene
                num_cands = min(7, max(1, total_scene_frames))
                if num_cands <= 2:
                    candidates = [(start_f + end_f) // 2]
                else:
                    step = (end_f - start_f) / (num_cands + 1)
                    candidates = [int(start_f + (i + 1) * step) for i in range(num_cands)]

                best_frame = None
                best_sharpness = -1.0
                best_target_f = (start_f + end_f) // 2

                for cand_f in candidates:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cand_f)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    # Sharpness via Laplacian variance
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame = frame
                        best_target_f = cand_f

                if best_frame is not None:
                    filename = f"keyframe_scene_{scene['scene_index']}.png"
                    filepath = os.path.join(static_dir, filename)
                    cv2.imwrite(filepath, best_frame)
                    keyframes.append(filepath)

                    scene["keyframe_path"] = filepath
                    scene["keyframe_url"] = f"/static/mock_r2/keyframes/{job_id}/{filename}"
                    scene["caption"] = f"Keyframe for Scene {scene['scene_index']}"
                    scene["blur_score"] = round(best_sharpness, 2)
                    scene["keyframe_frame_idx"] = best_target_f
                else:
                    scene["keyframe_path"] = ""
                    scene["keyframe_url"] = ""
                    scene["caption"] = ""
                    scene["blur_score"] = 0.0
            else:
                if strategy == "first":
                    target_frame = start_f
                else:
                    target_frame = (start_f + end_f) // 2
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if ret and frame is not None:
                    filename = f"keyframe_scene_{scene['scene_index']}.png"
                    filepath = os.path.join(static_dir, filename)
                    cv2.imwrite(filepath, frame)
                    keyframes.append(filepath)

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    
                    scene["keyframe_path"] = filepath
                    scene["keyframe_url"] = f"/static/mock_r2/keyframes/{job_id}/{filename}"
                    scene["caption"] = f"Keyframe for Scene {scene['scene_index']}"
                    scene["blur_score"] = round(sharpness, 2)
                    scene["keyframe_frame_idx"] = target_frame
                else:
                    scene["keyframe_path"] = ""
                    scene["keyframe_url"] = ""
                    scene["caption"] = ""
                    scene["blur_score"] = 0.0
                
        cap.release()
        return keyframes

    def process(self, video_path: str, output_dir: str) -> dict[str, Any]:
        scenes = self.detect_scenes(video_path)
        keyframes = self.extract_keyframes(video_path, scenes, output_dir)
        return {"scenes": scenes, "keyframes": keyframes}
