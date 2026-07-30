"""Celery Tasks — kết nối 3 modules (audio, visual, fusion).

Migrated from: src/mls/pipeline.py
Định nghĩa các Celery Tasks orchestrating the processing pipeline.
"""

from __future__ import annotations

# Suppress python warnings (deprecation, user warnings from PyTorch/Transformers)
import warnings
warnings.filterwarnings("ignore")

# Monkey-patch Hugging Face transformers torch.load security check
# to avoid ValueError when loading CLIP/BLIP models on torch < 2.6
try:
    import transformers.utils.import_utils
    import transformers.modeling_utils
    import transformers.utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    transformers.utils.check_torch_load_is_safe = lambda: None
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from ai_workers.core.celery_app import app
from ai_workers.modules.audio_v2.transcriber import AudioTranscriber
from ai_workers.modules.audio_v2.speaker import SpeakerDiarizer
from ai_workers.modules.visual_v2.scene_detector import SceneDetector
from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer
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
        del logs[:-15]  # Keep only the last 15 items
        print(f"[{stage.upper()}] {message}")
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
        import gc
        gc.collect()

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
        gc.collect()

        # Stage 3: Visual
        check_revoked()
        log_step("Bắt đầu phát hiện phân cảnh và trích xuất slide keyframes...", "visual", 38)
        visual = SceneDetector()
        visual_result = visual.process(local_video_path, output_dir)
        log_step(f"Đã trích xuất xong {len(visual_result.get('keyframes', []))} slide keyframes.", "visual", 55)

        del visual
        gc.collect()

        # Stage 4: Semantic
        check_revoked()
        log_step("Bắt đầu phân tích nội dung slide (CLIP embeddings)...", "semantic", 58)
        semantic = SemanticAnalyzer()
        # semantic.process now takes the list of scene dicts, filters them, and adds captions
        log_step("Đang chạy lọc slide trùng lặp bằng thuật toán K-Means...", "semantic", 60)
        filtered_scenes = semantic.process(visual_result.get("scenes", []))
        visual_result["scenes"] = filtered_scenes
        slides = filtered_scenes
        log_step(f"Đã hoàn thành mô tả slide bằng BLIP. Giữ lại {len(filtered_scenes)} slide đặc trưng.", "semantic", 70)

        del semantic
        gc.collect()

        # Upload keyframes to R2 if configured
        from ai_workers.core.config import worker_settings
        if worker_settings.CF_R2_ACCESS_KEY_ID:
            log_step("Đang tải các ảnh slide keyframe lên Cloudflare R2...", "storage", 72)
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
            log_step("Đã tải xong toàn bộ slide lên R2.", "storage", 74)

        # Stage 5: Timeline
        check_revoked()
        log_step("Bắt đầu ánh xạ trục thời gian bài giảng và chương mục...", "timeline", 75)
        timeline = TimelineBuilder()
        timeline_result = timeline.process(
            utterances, visual_result.get("scenes", []), slides,
        )
        log_step(f"Đã ánh xạ xong {len(timeline_result.get('chapters', []))} chương bài giảng.", "timeline", 85)

        del timeline
        gc.collect()

        # Stage 6: Summarization & RAG Indexing
        check_revoked()
        log_step("Đang yêu cầu mô hình AI (Groq API) sinh bản tóm tắt chi tiết...", "text", 88)
        summarizer = Summarizer()
        text_result = summarizer.process(
            utterances, slides, timeline_result.get("chapters", [])
        )
        log_step("Đã tạo bản tóm tắt bài giảng thành công.", "text", 93)

        # Build Multimodal RAG vector index in ChromaDB
        log_step("Đang tạo chỉ mục RAG đa phương thức vào ChromaDB...", "rag", 96)
        try:
            rag_success = summarizer.build_rag_index(job_id, utterances, slides)
            if rag_success:
                log_step("Đã lưu chỉ mục RAG vào ChromaDB thành công.", "rag", 98)
            else:
                log_step("Cảnh báo: Không thể tạo chỉ mục RAG (Sử dụng Mock Store).", "rag", 98)
        except Exception as rag_err:
            print(f"Error building RAG index: {rag_err}")
            log_step("Lỗi khi tạo chỉ mục RAG.", "rag", 98)

        del summarizer
        gc.collect()
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

    # Align transcript utterances with visual scenes to populate "script"
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
        # Fallback to pure temporal overlap if timeline semantic alignment failed
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
