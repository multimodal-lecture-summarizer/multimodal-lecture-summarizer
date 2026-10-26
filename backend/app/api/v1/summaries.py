import io
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.exceptions import NotFoundException, ValidationException
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, SummaryDTO, ResummarizeRequest
from app.api.deps import get_current_active_user
from app.models.user import User
from app.models.summary import Summary
from app.models.video import Video

router = APIRouter(route_class=CamelCaseAPIRoute)


def format_summary_dto(video: Video, summary: Summary) -> SummaryDTO:
    """Helper to convert Summary ORM model to rich SummaryDTO with variant metadata."""
    from app.schemas.summary import ChapterDTO, KeyframeDTO
    import json
    import re
    from datetime import datetime

    def _pick(obj: dict, *keys, default=None):
        for key in keys:
            if key in obj and obj[key] is not None:
                return obj[key]
        return default

    chapters = []
    for idx, c in enumerate(summary.chapters_json or []):
        if not isinstance(c, dict):
            continue
        start = float(_pick(c, "startTime", "start_time", "start_seconds", default=0.0) or 0.0)
        end = float(_pick(c, "endTime", "end_time", "end_seconds", default=start) or start)
        chapters.append(
            ChapterDTO(
                title=str(_pick(c, "title", default=f"Chapter {idx + 1}") or f"Chapter {idx + 1}"),
                start_time=start,
                end_time=end,
                summary=str(_pick(c, "summary", default="") or ""),
            )
        )

    keyframes = []
    for k in summary.keyframes_json or []:
        if not isinstance(k, dict):
            continue
        keyframes.append(
            KeyframeDTO(
                timestamp=float(_pick(k, "timestamp", default=0.0) or 0.0),
                image_url=str(_pick(k, "imageUrl", "image_url", default="") or ""),
                description=str(_pick(k, "description", "caption", default="") or ""),
                importance_score=float(
                    _pick(k, "importanceScore", "importance_score", default=0.5) or 0.5
                ),
            )
        )

    parsed_segments = []
    clean_transcript_text = summary.transcript_text or ""
    try:
        parsed = json.loads(summary.transcript_text)
        if isinstance(parsed, list):
            parsed_segments = parsed
            text_parts = []
            for seg in parsed:
                if seg.get("text") and seg.get("text") != "[Nhạc nền / Im lặng]":
                    text_parts.append(seg["text"])
            clean_transcript_text = " ".join(text_parts)
    except Exception:
        pass

    if not parsed_segments and clean_transcript_text:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_transcript_text) if s.strip()]
        total_dur = video.duration or 10.0
        sec_per_sentence = total_dur / len(sentences) if sentences else 10.0
        for idx, sentence in enumerate(sentences):
            s_start = idx * sec_per_sentence
            s_end = (idx + 1) * sec_per_sentence
            words_list = sentence.split()
            word_tokens = []
            if words_list:
                sec_per_word = sec_per_sentence / len(words_list)
                for w_idx, w in enumerate(words_list):
                    word_tokens.append({
                        "word": w,
                        "start": s_start + w_idx * sec_per_word,
                        "end": s_start + (w_idx + 1) * sec_per_word
                    })
            parsed_segments.append({
                "speaker": "SPEAKER_01",
                "start": s_start,
                "end": s_end,
                "text": sentence,
                "words": word_tokens
            })

    raw_takeaways = getattr(summary, "key_takeaways_json", None) or []
    if not raw_takeaways:
        bullet_points = []
        if summary.summary_text:
            lines = summary.summary_text.splitlines()
            for line in lines:
                stripped = line.strip()
                if re.match(r"^(\*|\-|\•|\d+\.)\s+", stripped):
                    cleaned = re.sub(r"^(\*|\-|\•|\d+\.)\s+", "", stripped)
                    if len(cleaned) > 5:
                        bullet_points.append(cleaned)
        if bullet_points:
            raw_takeaways = bullet_points[:6]
        elif chapters:
            raw_takeaways = [
                {
                    "timestamp": c.start_time,
                    "title": c.title,
                    "description": c.summary or c.title,
                }
                for c in chapters[:6]
            ]

    active_text = summary.summary_text or ""
    active_len = getattr(summary, "summary_length", "medium") or "medium"
    w_count = len(active_text.split()) if active_text else 0
    r_sec = max(1, w_count // 3) if w_count else 0
    variants = getattr(summary, "summary_variants_json", None)
    if not variants and active_text:
        variants = {
            active_len: {
                "summary_text": active_text,
                "key_takeaways_json": raw_takeaways,
                "word_count": w_count,
                "reading_time_sec": r_sec,
                "model_used": summary.model_used,
                "generated_at": datetime.utcnow().isoformat(),
            }
        }

    return SummaryDTO(
        summary_id=summary.summary_id,
        video_id=summary.video_id,
        summary_text=active_text,
        summary_length=active_len,
        summary_variants_json=variants,
        word_count=w_count,
        reading_time_sec=r_sec,
        chapters=chapters,
        keyframes=keyframes,
        transcript_text=clean_transcript_text,
        transcript_segments=parsed_segments,
        key_takeaways=raw_takeaways,
        objects_summary_json=getattr(summary, "objects_summary_json", None),
        model_used=summary.model_used,
        processing_time=summary.processing_time,
        video_title=video.title,
    )



@router.get(
    "/video/{video_id}",
    response_model=BaseDTO[SummaryDTO],
    summary="Get video summarization results",
    description="Retrieves the generated text summary, segmented chapters, and keyframes details for a video.",
)
def get_video_summary(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieves AI summarization results for a processed video."""
    video = (
        db.query(Video)
        .filter(Video.video_id == video_id, Video.user_id == current_user.user_id)
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    summary = db.query(Summary).filter(Summary.video_id == video_id).first()
    if not summary:
        raise NotFoundException(
            message="Summarization is still processing or has failed."
        )

    dto = format_summary_dto(video, summary)
    return BaseDTO(
        success=True,
        data=dto,
        message="Summary results retrieved successfully",
    )


@router.post(
    "/video/{video_id}/resummarize",
    response_model=BaseDTO[SummaryDTO],
    summary="Switch or regenerate summary to target cognitive depth/length",
    description="Instantly switches to cached summary variant (~0ms) or generates on-demand without reprocessing video.",
)
@router.post(
    "/{video_id}/resummarize",
    response_model=BaseDTO[SummaryDTO],
    include_in_schema=False,
)
def resummarize_video(
    video_id: uuid.UUID,
    payload: ResummarizeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Instant switch or on-demand generation of summary variant based on cognitive depth / length."""
    target_len = (payload.target_length or "medium").strip().lower()
    if target_len not in ["short", "medium", "detailed"]:
        target_len = "medium"

    video = (
        db.query(Video)
        .filter(Video.video_id == video_id, Video.user_id == current_user.user_id)
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    summary = db.query(Summary).filter(Summary.video_id == video_id).first()
    if not summary:
        raise NotFoundException(message="Bản tóm tắt chưa được khởi tạo hoặc đang trong tiến trình xử lý.")

    variants = dict(summary.summary_variants_json or {})

    # STEP 1 (Cache Hit): Fast-Path ~0ms
    # Validate cache entry: ensure cached summary is not truncated or stale
    is_valid_cache = False
    if not payload.force_regenerate and target_len in variants:
        v_data = variants[target_len]
        v_words = v_data.get("word_count") or len((v_data.get("summary_text") or "").split())
        if target_len == "detailed":
            is_valid_cache = v_words >= 200
        elif target_len == "medium":
            is_valid_cache = v_words >= 90
        else:
            is_valid_cache = v_words >= 25

    if is_valid_cache:
        variant = variants[target_len]
        summary.summary_length = target_len
        if variant.get("summary_text"):
            summary.summary_text = variant["summary_text"]
        if variant.get("key_takeaways_json"):
            summary.key_takeaways_json = variant["key_takeaways_json"]
        if variant.get("model_used"):
            summary.model_used = variant["model_used"]

        # Ensure video title is specific and not generic or speaker card
        try:
            from ai_workers.modules.summarization.summarizer import is_generic_title, clean_video_title
            if is_generic_title(video.title) or "|" in (video.title or ""):
                better_title = clean_video_title(
                    raw_title=video.title,
                    summary_text=summary.summary_text or "",
                    lang_code=getattr(video, "language", "vi") or "vi",
                    chapters=summary.chapters_json or [],
                    takeaways=summary.key_takeaways_json or [],
                )
                if better_title and not is_generic_title(better_title):
                    video.title = better_title
        except Exception as e:
            logger.warning(f"Failed to auto-heal title on cache hit: {e}")

        db.commit()
        db.refresh(summary)
        dto = format_summary_dto(video, summary)
        return BaseDTO(
            success=True,
            data=dto,
            message=f"Chuyển đổi thành công sang phiên bản tóm tắt '{target_len}' (Cache Hit)."
        )

    # STEP 2 (Cache Miss / Force Regenerate): On-Demand Generation without re-running full pipeline
    import sys
    from pathlib import Path
    _workspace_root = str(Path(__file__).resolve().parents[4])
    if _workspace_root not in sys.path:
        sys.path.insert(0, _workspace_root)

    from app.models.job import Job, JobType
    from app.models.video import VideoScene
    from ai_workers.core.checkpoint import CheckpointManager, STAGE_QUALITY_COMPLETE
    from ai_workers.modules.summarization.summarizer import Summarizer
    from datetime import datetime

    job = (
        db.query(Job)
        .filter(Job.video_id == video_id, Job.job_type == JobType.SUMMARIZE)
        .order_by(Job.started_at.desc())
        .first()
    )

    ckpt_data = None
    if job:
        try:
            ckpt_mgr = CheckpointManager()
            checkpoint = ckpt_mgr.load_checkpoint(str(job.job_id), expected_stage=STAGE_QUALITY_COMPLETE)
            if checkpoint and isinstance(checkpoint.get("data"), dict):
                ckpt_data = checkpoint["data"]
        except Exception:
            ckpt_data = None

    if ckpt_data:
        utterances = ckpt_data.get("utterances", [])
        keyframes = ckpt_data.get("keyframes", [])
        chapters = ckpt_data.get("chapters", summary.chapters_json or [])
        output_language = ckpt_data.get("output_language", getattr(current_user, "output_language", "vi") or "vi")
    else:
        # Reconstruct from DB
        raw_transcript = summary.transcript_text or ""
        utterances = []
        try:
            import json
            parsed = json.loads(raw_transcript)
            if isinstance(parsed, list):
                utterances = parsed
        except Exception:
            pass

        if not utterances and raw_transcript:
            import re
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_transcript) if s.strip()]
            dur = float(video.duration or 60.0)
            sec_per_sent = dur / len(sents) if sents else 5.0
            for i, st in enumerate(sents):
                utterances.append({
                    "start": i * sec_per_sent,
                    "end": (i + 1) * sec_per_sent,
                    "text": st,
                    "speaker": "SPEAKER_00",
                })

        db_scenes = db.query(VideoScene).filter(VideoScene.video_id == video_id).order_by(VideoScene.scene_index).all()
        keyframes = []
        for sc in db_scenes:
            keyframes.append({
                "start_seconds": sc.start_seconds,
                "end_seconds": sc.end_seconds,
                "timestamp": sc.start_seconds,
                "start_timecode": sc.start_timecode,
                "caption": sc.caption or "",
                "ocr_text": sc.script or "",
                "description": sc.caption or "",
                "importanceScore": 0.85,
                "imageUrl": sc.keyframe_url or "",
                "detected_objects": sc.detected_objects_json or {},
                "object_counts": sc.object_counts_json or {},
            })
        chapters = summary.chapters_json or []
        output_language = getattr(current_user, "output_language", "vi") or "vi"

    # Invoke Summarizer with target length
    summarizer = Summarizer()
    job_id_str = str(job.job_id) if job else str(video_id)
    try:
        text_result = summarizer.process(
            utterances,
            keyframes,
            chapters,
            job_id=job_id_str,
            output_language=output_language,
            summary_length=target_len,
        )
    except Exception as exc:
        raise ValidationException(message=f"Tạo lại tóm tắt thất bại: {str(exc)}")

    new_summary_text = text_result.get("summary", "")
    new_takeaways = text_result.get("key_takeaways", summary.key_takeaways_json or [])
    new_chapters = text_result.get("chapters", summary.chapters_json or [])
    calc_words = text_result.get("word_count") or (len(new_summary_text.split()) if new_summary_text else 0)
    calc_read_sec = text_result.get("reading_time_sec") or (max(1, calc_words // 3) if calc_words else 0)
    model_used = text_result.get("model_used", summary.model_used)

    # Save to variants cache
    variants[target_len] = {
        "summary_text": new_summary_text,
        "key_takeaways_json": new_takeaways,
        "word_count": calc_words,
        "reading_time_sec": calc_read_sec,
        "model_used": model_used,
        "generated_at": datetime.utcnow().isoformat(),
    }
    summary.summary_variants_json = variants
    summary.summary_text = new_summary_text
    summary.summary_length = target_len
    summary.key_takeaways_json = new_takeaways
    if new_chapters:
        summary.chapters_json = new_chapters
    summary.model_used = model_used

    # Update video title if newly synthesized or currently generic or speaker card
    new_title = text_result.get("video_title")
    try:
        from ai_workers.modules.summarization.summarizer import is_generic_title, clean_video_title
        if not video.title or is_generic_title(video.title) or "|" in (video.title or ""):
            if new_title and not is_generic_title(new_title) and "|" not in new_title:
                video.title = new_title
            else:
                healed = clean_video_title(
                    raw_title=new_title or video.title,
                    summary_text=new_summary_text,
                    lang_code=output_language,
                    chapters=new_chapters,
                    takeaways=new_takeaways,
                )
                if healed and not is_generic_title(healed):
                    video.title = healed
    except Exception as e:
        logger.warning(f"Failed to update video title: {e}")

    db.commit()
    db.refresh(summary)

    dto = format_summary_dto(video, summary)
    return BaseDTO(
        success=True,
        data=dto,
        message=f"Đã tái tạo và lưu phiên bản tóm tắt '{target_len}' thành công (On-Demand Generation)."
    )


@router.get(
    "/video/{video_id}/export",
    summary="Export summary results to TXT, SRT, DOCX, or PDF files",
    description="Generates and streams a downloadable file of the summarization.",
)
def export_summary(
    video_id: uuid.UUID,
    format: str = Query("txt", description="File format to export: txt, srt, docx, pdf"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Generates downloadable summary files as StreamingResponse."""
    video = (
        db.query(Video)
        .filter(Video.video_id == video_id, Video.user_id == current_user.user_id)
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    summary = db.query(Summary).filter(Summary.video_id == video_id).first()
    if not summary:
        raise NotFoundException(message="Summarization results not found.")

    format = format.lower().strip()
    if format not in ["txt", "srt", "docx", "pdf"]:
        raise ValidationException(
            message="Invalid format. Supported formats are: txt, srt, docx, pdf"
        )

    file_stream = io.BytesIO()
    
    # Parse transcript segments for export formats
    import json
    import re
    parsed_segments = []
    clean_transcript_text = summary.transcript_text
    
    try:
        parsed = json.loads(summary.transcript_text)
        if isinstance(parsed, list):
            parsed_segments = parsed
            text_parts = []
            for seg in parsed:
                if seg.get("text") and seg.get("text") != "[Nhạc nền / Im lặng]":
                    text_parts.append(seg["text"])
            clean_transcript_text = " ".join(text_parts)
    except Exception:
        pass

    if not parsed_segments and clean_transcript_text:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_transcript_text) if s.strip()]
        total_dur = video.duration or 10.0
        sec_per_sentence = total_dur / len(sentences) if sentences else 10.0
        
        for idx, sentence in enumerate(sentences):
            s_start = idx * sec_per_sentence
            s_end = (idx + 1) * sec_per_sentence
            parsed_segments.append({
                "start": s_start,
                "end": s_end,
                "text": sentence
            })

    def _chapter_export_fields(chapter: dict, idx: int) -> tuple[str, float, float, str]:
        title = chapter.get("title") or f"Chapter {idx}"
        start = chapter.get("startTime", chapter.get("start_time", chapter.get("start_seconds", 0)))
        end = chapter.get("endTime", chapter.get("end_time", chapter.get("end_seconds", start)))
        ch_summary = chapter.get("summary") or ""
        return title, start, end, ch_summary

    if format == "txt":
        content = (
            f"=== VIDEO SUMMARY REPORT ===\n"
            f"Video ID: {video_id}\n"
            f"Original URL: {video.original_url or 'N/A'}\n"
            f"Duration: {video.duration} seconds\n"
            f"Model Used: {summary.model_used}\n"
            f"Processing Time: {summary.processing_time}s\n"
            f"============================\n\n"
            f"SUMMARY:\n{summary.summary_text}\n\n"
            f"=== CHAPTERS ===\n"
        )
        for idx, c in enumerate(summary.chapters_json or [], 1):
            if not isinstance(c, dict):
                continue
            title, start, end, ch_summary = _chapter_export_fields(c, idx)
            content += f"Chapter {idx}: {title} ({start}s - {end}s)\n"
            content += f"Summary: {ch_summary}\n\n"

        content += f"=== TRANSCRIPT ===\n{clean_transcript_text or ''}\n"

        file_stream.write(content.encode("utf-8"))
        file_stream.seek(0)
        return StreamingResponse(
            file_stream,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=summary_{video_id}.txt"},
        )

    elif format == "srt":
        def format_srt_time(seconds: float) -> str:
            if not isinstance(seconds, (int, float)): return "00:00:00,000"
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        srt_content = ""
        for i, seg in enumerate(parsed_segments, 1):
            start_str = format_srt_time(seg.get("start", 0))
            end_str = format_srt_time(seg.get("end", 0))
            text = seg.get("text", "")
            srt_content += f"{i}\n{start_str} --> {end_str}\n{text}\n\n"

        file_stream.write(srt_content.encode("utf-8"))
        file_stream.seek(0)
        return StreamingResponse(
            file_stream,
            media_type="text/srt",
            headers={"Content-Disposition": f"attachment; filename=subtitles_{video_id}.srt"},
        )

    elif format == "docx":
        try:
            from docx import Document
        except ImportError:
            raise ValidationException(message="DOCX export is not supported on this server.")
            
        doc = Document()
        doc.add_heading("Video Summary Report", 0)
        doc.add_paragraph(f"Video ID: {video_id}")
        doc.add_paragraph(f"Duration: {video.duration} seconds")
        
        doc.add_heading("Summary", level=1)
        doc.add_paragraph(summary.summary_text)
        
        doc.add_heading("Chapters", level=1)
        for idx, c in enumerate(summary.chapters_json or [], 1):
            if not isinstance(c, dict):
                continue
            title, start, end, ch_summary = _chapter_export_fields(c, idx)
            doc.add_heading(f"Chapter {idx}: {title} ({start}s - {end}s)", level=2)
            doc.add_paragraph(ch_summary)
            
        doc.add_heading("Transcript", level=1)
        doc.add_paragraph(clean_transcript_text)
        
        doc.save(file_stream)
        file_stream.seek(0)
        return StreamingResponse(
            file_stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=summary_{video_id}.docx"},
        )

    elif format == "pdf":
        try:
            from fpdf import FPDF
        except ImportError:
            raise ValidationException(message="PDF export is not supported on this server.")
            
        import os
        
        pdf = FPDF()
        pdf.add_page()
        
        # Load font for Vietnamese support
        font_path = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "Roboto-Regular.ttf")
        font_name = "Arial"
        if os.path.exists(font_path):
            pdf.add_font("Roboto", "", font_path, uni=True)
            font_name = "Roboto"
            
        pdf.set_font(font_name, size=16)
        pdf.cell(0, 10, "Video Summary Report", ln=True, align='C')
        pdf.set_font(font_name, size=12)
        pdf.cell(0, 10, f"Video ID: {video_id}", ln=True)
        pdf.cell(0, 10, f"Duration: {video.duration} seconds", ln=True)
        pdf.ln(5)
        
        pdf.set_font(font_name, size=14)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font(font_name, size=11)
        pdf.multi_cell(0, 8, summary.summary_text)
        pdf.ln(5)
        
        pdf.set_font(font_name, size=14)
        pdf.cell(0, 10, "Chapters", ln=True)
        for idx, c in enumerate(summary.chapters_json or [], 1):
            if not isinstance(c, dict):
                continue
            title, start, end, ch_summary = _chapter_export_fields(c, idx)
            pdf.set_font(font_name, size=12)
            pdf.cell(0, 8, f"Chapter {idx}: {title} ({start}s - {end}s)", ln=True)
            pdf.set_font(font_name, size=11)
            pdf.multi_cell(0, 8, ch_summary)
            pdf.ln(3)
            
        pdf_bytes = pdf.output(dest='S')
        # fpdf2 dest='S' returns a bytearray
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin1')
            
        file_stream.write(pdf_bytes)
        file_stream.seek(0)
        return StreamingResponse(
            file_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{video_id}.pdf"},
        )
