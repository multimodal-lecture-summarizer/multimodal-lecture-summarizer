"""Celery Tasks — kết nối 3 modules (audio, visual, fusion).

Migrated from: src/mls/pipeline.py
Định nghĩa các Celery Tasks orchestrating the processing pipeline.
"""

from __future__ import annotations

from ai_workers.core.celery_app import app
from ai_workers.modules.audio.transcriber import AudioTranscriber
from ai_workers.modules.audio.speaker import SpeakerDiarizer
from ai_workers.modules.visual.scene_detector import SceneDetector
from ai_workers.modules.visual.semantic import SemanticAnalyzer
from ai_workers.modules.fusion.timeline import TimelineBuilder
from ai_workers.modules.fusion.summarizer import Summarizer


@app.task(bind=True, name="ai_workers.process_video")
def process_video(self, job_id: str, video_path: str, config_stack: str = "hybrid"):
    """Main Celery task: orchestrate full multimodal pipeline.

    Pipeline stages:
        1. Audio extraction + ASR (WhisperX)
        2. Speaker diarization (pyannote)
        3. Scene detection + keyframe extraction (PySceneDetect)
        4. Semantic analysis: OCR + CLIP + BLIP-2
        5. Timeline alignment + chapter segmentation
        6. LLM summarization + RAG index

    Stages 1-2 and 3-4 run in parallel where possible.
    """
    import time
    start_time = time.time()
    output_dir = f"./outputs/{job_id}"

    self.update_state(state="PROGRESS", meta={"stage": "download", "progress": 2})
    import os
    import requests
    import yt_dlp
    import uuid

    local_video_path = video_path
    is_temp_file = False
    video_file_url = None

    if video_path.startswith("http://") or video_path.startswith("https://"):
        if "youtube.com" not in video_path and "youtu.be" not in video_path:
            video_file_url = video_path

        temp_dir = "./storage/temp"
        os.makedirs(temp_dir, exist_ok=True)
        local_video_path = os.path.abspath(os.path.join(temp_dir, f"temp_{uuid.uuid4()}.mp4"))
        is_temp_file = True

        try:
            from ai_workers.core.config import worker_settings
            if "youtube.com" in video_path or "youtu.be" in video_path:
                print(f"Downloading YouTube video using yt-dlp from: {video_path}")
                ydl_opts = {
                    'outtmpl': local_video_path,
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                    'merge_output_format': 'mp4',
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_path])
                
                # yt-dlp might merge formats to a different extension (e.g. .mkv, .webm)
                import glob
                base_without_ext = local_video_path.rsplit(".", 1)[0]
                matching_files = glob.glob(base_without_ext + ".*")
                if matching_files:
                    local_video_path = matching_files[0]
                print(f"Successfully downloaded YouTube video to: {local_video_path}")

                # Get actual extension
                actual_ext = local_video_path.rsplit(".", 1)[-1]
                # Save downloaded video to Cloudflare R2 (or local mock folder)
                video_filename = f"youtube_{job_id}.{actual_ext}"
                object_key = f"videos/{video_filename}"
                
                # Default mock URL
                video_file_url = f"/static/mock_r2/{object_key}"
                
                # Copy to local mock directory
                mock_dir = os.path.abspath(os.path.join(os.getcwd(), "storage", "mock_r2_bucket", "videos"))
                os.makedirs(mock_dir, exist_ok=True)
                mock_filepath = os.path.join(mock_dir, video_filename)
                
                import shutil
                shutil.copy2(local_video_path, mock_filepath)
                print(f"Stored local copy of YouTube video in mock storage at: {mock_filepath}")
                
                # If R2 is enabled, upload to R2 and update url
                if worker_settings.CF_R2_ACCESS_KEY_ID:
                    try:
                        print(f"Uploading downloaded YouTube video to R2 under key: {object_key}...")
                        import boto3
                        s3_client = boto3.client(
                            "s3",
                            endpoint_url=f"https://{worker_settings.CF_R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
                            aws_access_key_id=worker_settings.CF_R2_ACCESS_KEY_ID,
                            aws_secret_access_key=worker_settings.CF_R2_SECRET_ACCESS_KEY,
                        )
                        s3_client.upload_file(
                            mock_filepath,
                            worker_settings.CF_R2_BUCKET_NAME,
                            object_key,
                            ExtraArgs={"ContentType": "video/mp4"}
                        )
                        public_base = worker_settings.CF_R2_PUBLIC_URL.rstrip("/")
                        video_file_url = f"{public_base}/{object_key}"
                        print(f"Successfully uploaded YouTube video to R2: {video_file_url}")
                        
                        # Cleanup local mock file since it's uploaded to R2
                        if os.path.exists(mock_filepath):
                            os.remove(mock_filepath)
                    except Exception as upload_err:
                        print(f"Failed to upload YouTube video to R2: {upload_err}. Falling back to mock URL.")
            elif ("r2" in video_path or "cloudflarestorage" in video_path) and worker_settings.CF_R2_ACCESS_KEY_ID:
                print(f"Downloading R2 video securely using boto3 from: {video_path}")
                import boto3
                bucket_name = worker_settings.CF_R2_BUCKET_NAME
                
                # Robustly extract object key by stripping the public URL prefix
                public_base = worker_settings.CF_R2_PUBLIC_URL.rstrip("/")
                if video_path.startswith(public_base):
                    object_key = video_path[len(public_base):].lstrip("/")
                else:
                    # Fallback for old S3 URL formats
                    parts = video_path.split(f"/{bucket_name}/")
                    if len(parts) > 1:
                        object_key = parts[1]
                    else:
                        object_key = "/".join(video_path.split("/")[3:])
                
                s3_client = boto3.client(
                    "s3",
                    endpoint_url=f"https://{worker_settings.CF_R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
                    aws_access_key_id=worker_settings.CF_R2_ACCESS_KEY_ID,
                    aws_secret_access_key=worker_settings.CF_R2_SECRET_ACCESS_KEY,
                )
                s3_client.download_file(bucket_name, object_key, local_video_path)
                print(f"Successfully downloaded R2 video securely to: {local_video_path}")
            else:
                print(f"Downloading remote file using requests from: {video_path}")
                res = requests.get(video_path, stream=True)
                res.raise_for_status()
                with open(local_video_path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Successfully downloaded remote file to: {local_video_path}")
        except Exception as e:
            print(f"Failed to download remote video: {e}")
            if is_temp_file and os.path.exists(local_video_path):
                os.remove(local_video_path)
            raise e

    try:
        # Stage 1: Audio
        print("[Stage 1/6] Starting audio transcription (Whisper ASR + VAD)...")
        self.update_state(state="PROGRESS", meta={"stage": "audio", "progress": 10})
        audio = AudioTranscriber()
        audio_result = audio.process(local_video_path)
        print("[Stage 1/6] Completed audio transcription successfully.")

        # Stage 2: Speaker diarization
        print("[Stage 2/6] Starting speaker diarization (PyAnnote)...")
        self.update_state(state="PROGRESS", meta={"stage": "speaker", "progress": 25})
        speaker = SpeakerDiarizer()
        utterances = speaker.process(
            local_video_path.rsplit(".", 1)[0] + ".wav",
            audio_result.get("segments", []),
        )
        print(f"[Stage 2/6] Completed speaker diarization successfully. Found {len(utterances)} speech segments.")

        # Stage 3: Visual
        print("[Stage 3/6] Starting scene boundary detection and keyframe extraction (PySceneDetect)...")
        self.update_state(state="PROGRESS", meta={"stage": "visual", "progress": 40})
        visual = SceneDetector()
        visual_result = visual.process(local_video_path, output_dir)
        print(f"[Stage 3/6] Completed scene detection successfully. Extracted {len(visual_result.get('keyframes', []))} keyframes.")

        # Stage 4: Semantic
        print("[Stage 4/6] Starting semantic analysis (OCR + CLIP keyframe processing)...")
        self.update_state(state="PROGRESS", meta={"stage": "semantic", "progress": 60})
        semantic = SemanticAnalyzer()
        # semantic.process now takes the list of scene dicts, filters them, and adds captions
        filtered_scenes = semantic.process(visual_result.get("scenes", []))
        visual_result["scenes"] = filtered_scenes
        slides = filtered_scenes
        print(f"[Stage 4/6] Completed semantic keyframe analysis. Kept {len(filtered_scenes)} keyframes.")

        # Upload keyframes to R2 if configured
        from ai_workers.core.config import worker_settings
        if worker_settings.CF_R2_ACCESS_KEY_ID:
            print("[Storage] Uploading extracted keyframes to Cloudflare R2 securely...")
            import boto3
            s3_client = boto3.client(
                "s3",
                endpoint_url=f"https://{worker_settings.CF_R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
                aws_access_key_id=worker_settings.CF_R2_ACCESS_KEY_ID,
                aws_secret_access_key=worker_settings.CF_R2_SECRET_ACCESS_KEY,
            )
            bucket_name = worker_settings.CF_R2_BUCKET_NAME
            
            for scene in visual_result.get("scenes", []):
                local_path = scene.get("keyframe_path")
                if local_path and os.path.exists(local_path):
                    filename = os.path.basename(local_path)
                    object_key = f"keyframes/{job_id}/{filename}"
                    
                    s3_client.upload_file(
                        local_path,
                        bucket_name,
                        object_key,
                        ExtraArgs={"ContentType": "image/png"}
                    )
                    
                    public_base = worker_settings.CF_R2_PUBLIC_URL.rstrip("/")
                    scene["keyframe_url"] = f"{public_base}/{object_key}"
                    print(f" -> Uploaded keyframe slide: {scene['keyframe_url']}")
            print("[Storage] All keyframes uploaded to R2 successfully.")

        # Stage 5: Timeline
        print("[Stage 5/6] Starting timeline and chapter boundaries mapping...")
        self.update_state(state="PROGRESS", meta={"stage": "timeline", "progress": 75})
        timeline = TimelineBuilder()
        timeline_result = timeline.process(
            utterances, visual_result.get("scenes", []), slides,
        )
        print(f"[Stage 5/6] Completed timeline mapping. Identified {len(timeline_result.get('chapters', []))} chapters.")

        # Stage 6: Summarization
        print("[Stage 6/6] Requesting AI detailed lecture summarization (Groq API)...")
        self.update_state(state="PROGRESS", meta={"stage": "text", "progress": 90})
        summarizer = Summarizer()
        text_result = summarizer.process(
            utterances, slides, timeline_result.get("chapters", [])
        )
        print("[Stage 6/6] Summarization generated successfully.")
    finally:
        # Cleanup temporary files
        if is_temp_file:
            if os.path.exists(local_video_path):
                try:
                    os.remove(local_video_path)
                    print(f"Cleaned up temporary video file: {local_video_path}")
                except Exception as err:
                    print(f"Failed to clean up temporary video file {local_video_path}: {err}")
            
            wav_path = local_video_path.rsplit(".", 1)[0] + ".wav"
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                    print(f"Cleaned up temporary audio file: {wav_path}")
                except Exception as err:
                    print(f"Failed to clean up temporary audio file {wav_path}: {err}")

    elapsed_time = time.time() - start_time
    duration = 0.0
    if utterances:
        duration = utterances[-1]["end"]

    # Align transcript utterances with visual scenes to populate "script" and update keyframe description
    scenes = visual_result.get("scenes", [])
    for scene in scenes:
        scene_start = scene["start_seconds"]
        scene_end = scene["end_seconds"]
        
        # Find overlapping utterances
        scene_utterances = []
        for utt in utterances:
            if utt["start"] < scene_end and utt["end"] > scene_start:
                scene_utterances.append(utt["text"])
                
        combined_script = " ".join(scene_utterances).strip()
        scene["script"] = combined_script

    # Construct keyframes from visual scenes for FE gallery compatibility
    keyframes = []
    for scene in scenes:
        keyframes.append({
            "timestamp": scene["start_seconds"],
            "imageUrl": scene["keyframe_url"],
            "description": scene.get("caption", f"Slide at {scene['start_timecode']}"),
            "transcript": scene.get("script", ""),
            "importanceScore": scene.get("importanceScore", 0.8)
        })

    return {
        "job_id": job_id,
        "status": "done",
        "video_title": text_result.get("video_title", "Untitled Lecture Video"),
        "summary": text_result.get("summary", ""),
        "chapters": text_result.get("chapters", []),
        "keyframes": keyframes,
        "transcript_text": audio_result.get("text", ""),
        "transcript_segments": utterances,
        "scenes": visual_result.get("scenes", []),
        "duration": duration,
        "model_used": text_result.get("model_used", "Groq"),
        "processing_time": elapsed_time,
        "video_file_path": video_file_url
    }



@app.task(name="ai_workers.process_audio")
def process_audio(job_id: str, video_path: str):
    """Subtask: audio extraction + ASR only."""
    audio = AudioTranscriber()
    return audio.process(video_path)


@app.task(name="ai_workers.process_visual")
def process_visual(job_id: str, video_path: str, output_dir: str):
    """Subtask: scene detection + keyframe extraction only."""
    visual = SceneDetector()
    return visual.process(video_path, output_dir)
