import uuid
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.exceptions import NotFoundException
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, QAAskRequest, QALogDTO
from app.api.deps import get_current_active_user
from app.models.user import User
from app.models.video import Video
from app.models.summary import Summary
from app.models.qa import QALog
from app.services.chromadb import chromadb_service
from app.services.llm import llm_service

router = APIRouter(route_class=CamelCaseAPIRoute)


@router.post(
    "/video/{video_id}",
    response_model=BaseDTO[QALogDTO],
    summary="Ask a question about the video content (RAG Q&A)",
    description="Queries ChromaDB vector store for context, calls Groq API to generate answers, and logs user questions.",
)
def ask_question(
    video_id: uuid.UUID,
    payload: QAAskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Processes user query against speech transcript index (ChromaDB + Groq LLM)."""
    # 1. Verify ownership
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
            message="No transcript index found. Video must be fully summarized before asking questions."
        )

    # 2. Query similar chunks from ChromaDB with metadata
    chunk_items = chromadb_service.query_similar_chunks_with_metadata(
        video_id=video_id,
        query=payload.question,
        limit=6,
    )

    # Fallback to chapters from DB if ChromaDB chunks all have 0.0 start_seconds (legacy videos)
    has_varied_timestamps = any(
        float(item.get("metadata", {}).get("start_seconds", 0.0)) > 0.0
        for item in chunk_items
    )

    if not has_varied_timestamps and summary:
        chapters = summary.chapters_json if isinstance(summary.chapters_json, list) else []
        keyframes = summary.keyframes_json if isinstance(summary.keyframes_json, list) else []
        
        first_speech_sec = 0.0
        if keyframes and len(keyframes) > 0:
            first_speech_sec = float(keyframes[0].get("timestamp", 0.0))

        if chapters:
            fallback_items = []
            for idx, ch in enumerate(chapters):
                st = float(ch.get("startTime", ch.get("start_time", idx * 120.0)))
                et = float(ch.get("endTime", ch.get("end_time", st + 120.0)))
                
                # If chapter 1 starts at 0.0 but actual first keyframe/speech is at >0s (e.g. 24s), use speech start
                if idx == 0 and st == 0.0 and first_speech_sec > 0.0:
                    st = first_speech_sec

                m = int(st) // 60
                s = int(st) % 60
                tc = f"{m:02d}:{s:02d}"
                c_title = ch.get("title", f"Phân đoạn {idx+1}")
                c_summary = ch.get("summary", "")
                c_doc = f"[{tc}] Nội dung phân đoạn '{c_title}': {c_summary}"
                fallback_items.append({
                    "document": c_doc,
                    "metadata": {"video_id": str(video_id), "start_seconds": st, "end_seconds": et, "timecode": tc}
                })
            if fallback_items:
                chunk_items = fallback_items

    retrieved_chunks = [item["document"] for item in chunk_items]
    citations = []
    ref_time = None

    context_lines = []
    for item in chunk_items:
        doc = item.get("document", "")
        meta = item.get("metadata", {})
        start_sec = meta.get("start_seconds")
        
        if start_sec is not None:
            s_val = float(start_sec)
            m = int(s_val) // 60
            s = int(s_val) % 60
            tc = f"{m:02d}:{s:02d}"
        else:
            tc = meta.get("timecode", "00:00")

        context_lines.append(f"Chunk [{tc}]: {doc}")
        
        if start_sec is not None:
            if ref_time is None:
                ref_time = float(start_sec)
            citations.append({
                "startSeconds": float(start_sec),
                "endSeconds": float(meta.get("end_seconds", float(start_sec) + 60.0)),
                "timecode": tc,
                "keyframeUrl": meta.get("keyframe_url", ""),
                "snippet": doc[:120] + "..." if len(doc) > 120 else doc
            })

    # 3. Retrieve recent chat history for multi-turn conversational RAG
    recent_history = (
        db.query(QALog)
        .filter(QALog.video_id == video_id, QALog.user_id == current_user.user_id)
        .order_by(QALog.asked_at.desc())
        .limit(4)
        .all()
    )
    recent_history.reverse()

    history_parts = []
    for log in recent_history:
        history_parts.append(f"User: {log.question}\nAI: {log.answer}")
    history_str = "\n".join(history_parts)

    # 4. Construct LLM context prompt (combine speech text & visual slide keyframes)
    visual_keyframe_lines = []
    if summary and summary.keyframes_json and isinstance(summary.keyframes_json, list):
        for kf in summary.keyframes_json:
            ts = float(kf.get("timestamp", 0.0))
            m = int(ts) // 60
            s = int(ts) % 60
            tc = f"{m:02d}:{s:02d}"
            desc = kf.get("description", "").strip()
            if desc and desc != f"Slide at {tc}":
                visual_keyframe_lines.append(f"Visual Slide [{tc}]: {desc}")

    context_str = "\n".join(context_lines)
    
    # 4. Construct LLM context prompt (combine overall summary, chapters, transcript chunks & visual slides)
    prompt_parts = []
    
    if summary and summary.summary_text:
        prompt_parts.append(f"Overall Lecture Summary:\n{summary.summary_text}")

    if summary and summary.chapters_json and isinstance(summary.chapters_json, list):
        ch_lines = []
        for idx, ch in enumerate(summary.chapters_json):
            st = float(ch.get("startTime", ch.get("start_time", 0.0)))
            m = int(st) // 60
            s = int(st) % 60
            tc = f"{m:02d}:{s:02d}"
            c_title = ch.get("title", f"Phân đoạn {idx+1}")
            c_sum = ch.get("summary", "")
            ch_lines.append(f"Chapter [{tc}] {c_title}: {c_sum}")
        if ch_lines:
            prompt_parts.append("Lecture Chapter Outline:\n" + "\n".join(ch_lines))

    prompt_parts.append(f"Retrieved Specific Video Chunks & Speech:\n{context_str}")

    if visual_keyframe_lines:
        prompt_parts.append("Visual Slide Keyframes & Images:\n" + "\n".join(visual_keyframe_lines))
    if history_str:
        prompt_parts.append(f"Recent Chat History:\n{history_str}")
    prompt_parts.append(f"Student Question: {payload.question}")
    prompt_parts.append("Answer (matching the language of question):")

    system_prompt = (
        "You are an intelligent academic assistant for lecture videos.\n"
        "Answer the student's question accurately based on the Overall Lecture Summary, Chapter Outline, Specific Transcripts, and Visual Slides provided below.\n"
        "CRITICAL RULES:\n"
        "1. OVERVIEW & SUMMARY QUESTIONS: When asked for a summary, key points, main takeaways, or chapter topics (e.g. '10 ý chính', '3 ý quan trọng', 'tóm tắt', 'bài giảng nói về gì'), synthesize the response directly from the Overall Lecture Summary, Chapter Outline, and Transcripts.\n"
        "2. MULTIMODAL INTEGRATION: Pay close attention to speech transcripts, chapter outlines, and Visual Slide descriptions. Answer accurately using on-screen slide text or speech when asked about specific concepts or objects.\n"
        "3. LANGUAGE MATCHING & NATURAL VIETNAMESE: Always respond in the EXACT SAME LANGUAGE as the student's question. In Vietnamese, use natural academic phrasing like 'nội dung bài giảng' or 'video'.\n"
        "4. CONCISENESS & STRUCTURE: Format answers with bullet points or numbered lists when requested (e.g., for '10 ý chính' or '3 ý quan trọng').\n"
        "5. TIMESTAMPS: Include exact [MM:SS] timestamp citations when referencing specific quotes, chapters, or sections of the video.\n"
        "6. TRUTHFULNESS: Only state that information is not mentioned if NEITHER the summary, chapters, slides, nor transcripts contain any relevant details."
    )

    prompt = "\n\n".join(prompt_parts)

    # 5. Invoke LLM API (OpenRouter -> Groq -> Mock)
    answer = llm_service.generate_chat_completion(
        prompt=prompt, system_prompt=system_prompt
    )

    # 5. Log transaction into PostgreSQL
    qa_log = QALog(
        video_id=video_id,
        user_id=current_user.user_id,
        question=payload.question,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
    )
    db.add(qa_log)
    db.commit()
    db.refresh(qa_log)

    response_dto = QALogDTO.model_validate(qa_log)
    response_dto.reference_time = ref_time
    response_dto.citations = citations

    return BaseDTO(
        success=True,
        data=response_dto,
        message="Question answered successfully",
    )


@router.get(
    "/video/{video_id}/history",
    response_model=BaseDTO[list[QALogDTO]],
    summary="Get QA chat history for a specific video",
)
def get_qa_history(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieves all past question & answer history for the video."""
    logs = (
        db.query(QALog)
        .filter(QALog.video_id == video_id, QALog.user_id == current_user.user_id)
        .order_by(QALog.asked_at.asc())
        .all()
    )
    result = [QALogDTO.model_validate(log) for log in logs]
    return BaseDTO(
        success=True,
        data=result,
        message="QA history retrieved successfully",
    )


@router.delete(
    "/video/{video_id}/history",
    response_model=BaseDTO[dict],
    summary="Delete all QA chat history for a video",
)
def delete_qa_history(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Deletes all QALog entries for a specific video and user."""
    db.query(QALog).filter(
        QALog.video_id == video_id, QALog.user_id == current_user.user_id
    ).delete(synchronize_session=False)
    db.commit()
    return BaseDTO(
        success=True,
        data={"deleted": True},
        message="QA history deleted successfully",
    )



