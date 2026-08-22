"""Celery Tasks — kết nối 3 modules (audio, visual, fusion).

Migrated from: src/mls/pipeline.py
Định nghĩa các Celery Tasks orchestrating the processing pipeline.
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress python warnings (deprecation, user warnings from PyTorch/Transformers)
import warnings
warnings.filterwarnings("ignore")

from ai_workers.core.celery_app import app
from ai_workers.core.resource_cleanup import (
    ensure_process_memory_available,
    release_worker_resources,
)


def _patch_transformers_torch_load_check() -> None:
    """Patch Transformers lazily so Celery startup does not import the full AI stack."""
    try:
        import transformers.modeling_utils
        import transformers.utils
        import transformers.utils.import_utils

        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
        transformers.utils.check_torch_load_is_safe = lambda: None
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass




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
    from ai_workers.core.config import worker_settings

    try:
        available_mb = ensure_process_memory_available(
            worker_settings.PROCESS_MIN_AVAILABLE_MEMORY_MB,
            soft_min_available_mb=worker_settings.PROCESS_SOFT_MIN_AVAILABLE_MEMORY_MB,
            retry_seconds=worker_settings.PROCESS_MEMORY_RETRY_SECONDS,
            retry_interval_seconds=worker_settings.PROCESS_MEMORY_RETRY_INTERVAL_SECONDS,
        )
        if available_mb is not None:
            print(f"[Preflight] RAM available before video processing: {available_mb} MB")
    except RuntimeError as preflight_err:
        message = str(preflight_err)
        print(f"[Preflight] {message}")
        self.update_state(
            state="PROGRESS",
            meta={"stage": "preflight", "progress": 0, "error": message},
        )
        raise

    _patch_transformers_torch_load_check()
    from ai_workers.modules.audio_v2.transcriber import AudioTranscriber
    from ai_workers.modules.audio_v2.speaker import SpeakerDiarizer
    from ai_workers.modules.visual_v2.scene_detector import SceneDetector
    from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer
    from ai_workers.modules.fusion.timeline import TimelineBuilder
    from ai_workers.modules.fusion.summarizer import Summarizer
    from ai_workers.modules.fusion.quality_postprocess import apply_quality_postprocess

    start_time = time.time()
    output_dir = f"./outputs/{job_id}"

    self.update_state(state="PROGRESS", meta={"stage": "download", "progress": 2})
    
    def check_revoked():
        if not hasattr(self, "request") or not self.request or not self.request.id:
            return
        from celery.result import AsyncResult
        try:
            state = AsyncResult(self.request.id, app=app).state
            if state == 'REVOKED':
                raise Exception("Tác vụ bị hủy bởi người dùng.")
        except Exception as e:
            if "Tác vụ bị hủy" in str(e):
                raise e
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
                    'retries': 10,
                    'fragment_retries': 10,
                    'source_address': '0.0.0.0',  # Ep dung IPv4 de tranh ranh IPv6 bi YouTube block
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android', 'web', 'mweb']
                        }
                    },
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

    logs = []
    def log_step(message, stage, progress):
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        logs.append(log_line)
        del logs[:-40]  # Keep the latest pipeline logs for the final status payload.
        print(f"[{stage.upper()}] {message}", flush=True)
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": stage,
                "progress": progress,
                "logs": logs
            }
        )

    try:
        # Stage 1: Audio
        check_revoked()
        log_step("Khởi chạy tiến trình xử lý video bài giảng...", "audio", 5)
        log_step("Bắt đầu trích xuất âm thanh từ tệp video...", "audio", 8)
        audio = AudioTranscriber()
        audio_result = audio.process(local_video_path)
        log_step("Trích xuất và nhận dạng giọng nói thành công.", "audio", 20)

        # Stage-level VRAM cleanup
        del audio
        release_worker_resources("audio stage")

        # Stage 2: Speaker diarization
        check_revoked()
        log_step("Bắt đầu phân tích phân biệt người nói...", "speaker", 22)
        speaker = SpeakerDiarizer()
        utterances = speaker.process(
            local_video_path.rsplit(".", 1)[0] + ".wav",
            audio_result.get("segments", []),
        )
        log_step(f"Hoàn thành phân tích người nói. Tìm thấy {len(utterances)} phân đoạn thoại.", "speaker", 35)

        del speaker
        release_worker_resources("speaker stage")

        # Stage 3: Visual
        check_revoked()
        log_step("Bắt đầu phát hiện phân cảnh và trích xuất slide keyframes...", "visual", 38)
        visual = SceneDetector()
        visual_result = visual.process(local_video_path, output_dir)
        log_step(f"Đã trích xuất xong {len(visual_result.get('keyframes', []))} slide keyframes.", "visual", 55)

        del visual
        release_worker_resources("visual stage")

        # Stage 4: Semantic
        check_revoked()
        log_step("Bắt đầu phân tích nội dung slide (CLIP embeddings)...", "semantic", 58)
        semantic = SemanticAnalyzer()
        # semantic.process now takes the list of scene dicts, filters them, and adds captions
        log_step("Đang chạy lọc slide trùng lặp bằng thuật toán K-Means...", "semantic", 60)
        filtered_scenes = semantic.filter_scenes_clip(visual_result.get("scenes", []))
        log_step(f"Đã lọc còn {len(filtered_scenes)} slide đặc trưng. Bắt đầu Florence-2 captioning/OCR...", "semantic", 64)
        semantic.caption_scenes_florence2(filtered_scenes)
        log_step("Đã hoàn thành Florence-2 captioning. Bắt đầu OCR nội dung slide...", "semantic", 67)
        semantic.extract_ocr_paddleocr(filtered_scenes)
        visual_result["scenes"] = filtered_scenes
        slides = filtered_scenes
        log_step(f"Đã hoàn thành phân tích nội dung slide. Giữ lại {len(filtered_scenes)} slide đặc trưng.", "semantic", 70)

        del semantic
        release_worker_resources("semantic stage")



        # Stage 5: Timeline
        check_revoked()
        log_step("Bắt đầu ánh xạ trục thời gian bài giảng và chương mục...", "timeline", 75)
        timeline = TimelineBuilder()
        timeline_result = timeline.process(
            utterances, visual_result.get("scenes", []), slides,
        )
        log_step(f"Đã ánh xạ xong {len(timeline_result.get('chapters', []))} chương bài giảng.", "timeline", 85)

        del timeline
        release_worker_resources("timeline stage")

        # Re-construct keyframes and run Quality Sprint BEFORE Summarization
        chapters = timeline_result.get("chapters", [])
        scenes = visual_result.get("scenes", [])
        scene_utterance_lists = {id(scene): [] for scene in scenes}
        
        aligned_segments = timeline_result.get("aligned_segments", [])
        if aligned_segments:
            for item in aligned_segments:
                scene_id = item.get("scene_id")
                utt_text = item.get("utterance", {}).get("text", "")
                if scene_id in scene_utterance_lists and utt_text:
                    scene_utterance_lists[scene_id].append(utt_text)
        else:
            for utt in utterances:
                u_start = utt.get("start", 0.0)
                u_end = utt.get("end", 0.0)
                best_scene = None
                max_overlap = 0.0
                for scene in scenes:
                    scene_start = scene.get("start_seconds", 0.0)
                    scene_end = scene.get("end_seconds", 0.0)
                    overlap_start = max(u_start, scene_start)
                    overlap_end = min(u_end, scene_end)
                    overlap = overlap_end - overlap_start
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_scene = scene
                if best_scene is not None and max_overlap > 0:
                    scene_utterance_lists[id(best_scene)].append(utt.get("text", ""))
                    
        for scene in scenes:
            scene["script"] = " ".join(scene_utterance_lists[id(scene)]).strip()

        keyframes = []
        for scene in scenes:
            keyframes.append({
                "timestamp": scene["start_seconds"],
                "local_path": scene.get("keyframe_path"),
                "imageUrl": scene.get("keyframe_url", ""),
                "description": scene.get("caption", f"Slide at {scene['start_timecode']}"),
                "transcript": scene.get("script", ""),
                "ocr_text": scene.get("ocr_text", ""),
                "blur_score": scene.get("blur_score"),
                "importanceScore": scene.get("importanceScore", 0.8),
            })

        sprint_stats = None
        export_meta = None
        from ai_workers.core.config import worker_settings
        if worker_settings.ENABLE_SPRINT_STACK:
            log_step(f"Đang chạy quality sprint stack ({worker_settings.SPRINT_STACK})...", "quality", 86)
            import time
            sprint_start_time = time.time()
            try:
                qp = apply_quality_postprocess(
                    chapters=chapters,
                    keyframes=keyframes,
                    utterances=utterances,
                    stack_name=worker_settings.SPRINT_STACK,
                    min_chapter_sec=worker_settings.MIN_CHAPTER_SEC,
                )
                chapters = qp["chapters"]
                keyframes = qp["keyframes"]
                sprint_stats = qp.get("sprint_stats")
                export_meta = qp.get("export_meta")
                sprint_duration = time.time() - sprint_start_time
                log_step(f"Sprint stack xong trong {sprint_duration:.2f}s: {len(chapters)} chapters, {len(keyframes)} keyframes.", "quality", 87)
            except Exception as qp_err:
                print(f"[Quality] Sprint stack failed, keeping baseline outputs: {qp_err}")

        # Storage (R2 Upload of FINAL keyframes)
        if worker_settings.CF_R2_ACCESS_KEY_ID:
            log_step("Đang tải các ảnh slide keyframe lên Cloudflare R2...", "storage", 87)
            import time
            r2_start = time.time()
            import boto3
            s3_client = boto3.client(
                "s3",
                endpoint_url=f"https://{worker_settings.CF_R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
                aws_access_key_id=worker_settings.CF_R2_ACCESS_KEY_ID,
                aws_secret_access_key=worker_settings.CF_R2_SECRET_ACCESS_KEY,
            )
            bucket_name = worker_settings.CF_R2_BUCKET_NAME
            
            for kf in keyframes:
                local_path = kf.get("local_path")
                if local_path and os.path.exists(local_path):
                    filename = os.path.basename(local_path)
                    object_key = f"keyframes/{job_id}/{filename}"
                    try:
                        s3_client.upload_file(local_path, bucket_name, object_key, ExtraArgs={"ContentType": "image/png"})
                        public_base = worker_settings.CF_R2_PUBLIC_URL.rstrip("/")
                        kf["imageUrl"] = f"{public_base}/{object_key}"
                    except Exception as upload_err:
                        print(f"R2 upload error: {upload_err}")
            
            r2_duration = time.time() - r2_start
            log_step(f"Đã tải xong {len(keyframes)} keyframes lên R2 trong {r2_duration:.2f}s.", "storage", 88)

        # Sync keyframes back to scenes (Quality Sprint filters + R2 URLs)
        valid_kf_map = {kf.get("local_path"): kf for kf in keyframes if kf.get("local_path")}
        synced_scenes = []
        for scene in scenes:
            path = scene.get("keyframe_path")
            if path in valid_kf_map:
                kf = valid_kf_map[path]
                if kf.get("imageUrl"):
                    scene["keyframe_url"] = kf["imageUrl"]
                synced_scenes.append(scene)
        
        # Update the visual result so the API job saver sees the filtered & URL-updated scenes
        visual_result["scenes"] = synced_scenes

        # Stage 6: Summarization
        check_revoked()
        log_step("Đang yêu cầu mô hình AI (Groq API) sinh bản tóm tắt chi tiết...", "text", 88)
        summarizer = Summarizer()
        text_result = summarizer.process(
            utterances, keyframes, chapters
        )
        # Update chapters with LLM output (titles and summaries)
        chapters = text_result.get("chapters", chapters)
        log_step("Đã tạo bản tóm tắt bài giảng thành công.", "text", 93)


        del summarizer
        release_worker_resources("summarization stage")
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
        release_worker_resources("process_video pipeline")

    elapsed_time = time.time() - start_time
    duration = 0.0
    if utterances:
        duration = utterances[-1]["end"]

    result = {
        "job_id": job_id,
        "status": "done",
        "video_title": text_result.get("video_title", "Untitled Lecture Video"),
        "summary": text_result.get("summary", ""),
        "chapters": chapters,
        "keyframes": keyframes,
        "transcript_text": audio_result.get("text", ""),
        "transcript_segments": utterances,
        "scenes": visual_result.get("scenes", []),
        "duration": duration,
        "model_used": text_result.get("model_used", "Groq"),
        "processing_time": elapsed_time,
        "video_file_path": video_file_url,
    }
    if sprint_stats is not None:
        result["sprint_stats"] = sprint_stats
    if export_meta is not None:
        result["export_meta"] = export_meta
    log_step("Hoàn tất xử lý video và lưu kết quả.", "completed", 100)
    result["stage"] = "completed"
    result["progress"] = 100
    result["logs"] = list(logs)
    release_worker_resources("process_video result assembly")
    return result



@app.task(name="ai_workers.process_audio")
def process_audio(job_id: str, video_path: str):
    """Subtask: audio extraction + ASR only."""
    from ai_workers.core.config import worker_settings

    available_mb = ensure_process_memory_available(
        worker_settings.PROCESS_MIN_AVAILABLE_MEMORY_MB,
        soft_min_available_mb=worker_settings.PROCESS_SOFT_MIN_AVAILABLE_MEMORY_MB,
        retry_seconds=worker_settings.PROCESS_MEMORY_RETRY_SECONDS,
        retry_interval_seconds=worker_settings.PROCESS_MEMORY_RETRY_INTERVAL_SECONDS,
    )
    if available_mb is not None:
        print(f"[Preflight] RAM available before audio processing: {available_mb} MB")

    _patch_transformers_torch_load_check()
    from ai_workers.modules.audio_v2.transcriber import AudioTranscriber

    audio = None
    try:
        audio = AudioTranscriber()
        return audio.process(video_path)
    finally:
        audio = None
        release_worker_resources("process_audio task")


@app.task(name="ai_workers.process_visual")
def process_visual(job_id: str, video_path: str, output_dir: str):
    """Subtask: scene detection + keyframe extraction only."""
    from ai_workers.modules.visual_v2.scene_detector import SceneDetector

    visual = None
    try:
        visual = SceneDetector()
        return visual.process(video_path, output_dir)
    finally:
        visual = None
        release_worker_resources("process_visual task")


@app.task(bind=True, name="ai_workers.build_rag_index", max_retries=3, autoretry_for=(Exception,), retry_backoff=True)
def build_rag_index(self, video_id: str):
    import sys
    import os
    import json
    
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
        
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        from app.models.summary import Summary
        from app.models.video import Video
        from app.core.constants import RagStatus
        from ai_workers.modules.fusion.summarizer import Summarizer
        
        summary = db.query(Summary).filter(Summary.video_id == video_id).first()
        if not summary:
            raise ValueError(f"No summary found for video {video_id}")
            
        utterances = []
        if summary.transcript_text:
            try:
                utterances = json.loads(summary.transcript_text)
            except Exception:
                pass
                
        slides = summary.keyframes_json or []
        
        summarizer = Summarizer()
        rag_success = summarizer.build_rag_index(video_id, utterances, slides)
        
        if rag_success:
            db.query(Video).filter(Video.video_id == video_id).update({"rag_status": RagStatus.READY})
        else:
            db.query(Video).filter(Video.video_id == video_id).update({"rag_status": RagStatus.FAILED})
        db.commit()
    except Exception as e:
        if self.request.retries >= self.max_retries:
            from app.models.video import Video
            from app.core.constants import RagStatus
            db.query(Video).filter(Video.video_id == video_id).update({"rag_status": RagStatus.FAILED})
            db.commit()
        raise e
    finally:
        db.close()
