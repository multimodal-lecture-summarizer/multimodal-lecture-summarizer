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
    system_prompt = (
        "You are an intelligent academic assistant for lecture videos.\n"
        "Answer the student's question based strictly on the provided video context (speech transcript AND visual slide images) and recent conversation history.\n"
        "CRITICAL RULES:\n"
        "1. MULTIMODAL INTEGRATION: Pay close attention to both speech transcript and Visual Slide descriptions. If the user asks about an object, slide, or topic shown on screen (e.g. 'cell phones', 'điện thoại', 'máy ảnh', 'slide 3:03'), answer accurately using the Visual Slide content.\n"
        "2. LANGUAGE MATCHING & NATURAL VIETNAMESE: Always respond in the EXACT SAME LANGUAGE as the student's question. In Vietnamese, use natural terms like 'nội dung bài giảng' or 'video'—NEVER use weird literal translations like 'khuôn khắc' or 'khung bối cảnh'.\n"
        "3. CONCISENESS & CLARITY: Answer simple or direct questions in a direct, natural 1-2 sentence response. Do NOT dump unnecessary bullet lists unless specifically requested.\n"
        "4. TIMESTAMPS: Include exact [MM:SS] timestamp citations when referencing specific quotes or sections of the video.\n"
        "5. CONVERSATIONAL CONTINUITY: Use the recent chat history to understand pronouns ('ông ấy', 'ý đó') or meta-feedback.\n"
        "6. TRUTHFULNESS: If the lecture video does not contain information to answer the question, state politely in natural language (e.g. 'Nội dung bài giảng không đề cập đến thông tin này.')."
    )

    prompt_parts = [f"Video Speech & Text Context:\n{context_str}"]
    if visual_keyframe_lines:
        prompt_parts.append("Visual Slide Keyframes & Images:\n" + "\n".join(visual_keyframe_lines))
    if history_str:
        prompt_parts.append(f"Recent Chat History:\n{history_str}")
    prompt_parts.append(f"Student Question: {payload.question}")
    prompt_parts.append("Answer (matching the language of question):")

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



