import uuid
import re
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.exceptions import NotFoundException
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, QAAskRequest, QALogDTO
from app.schemas.qa import QACitationDTO
from app.api.deps import get_current_active_user
from app.models.user import User
from app.models.video import Video
from app.models.summary import Summary
from app.models.qa import QALog
from app.services.chromadb import chromadb_service
from app.services.llm import llm_service

router = APIRouter(route_class=CamelCaseAPIRoute)


META_INSTRUCTION_PATTERNS = [
    re.compile(r"^\s*V\u00ec c\u00e2u h\u1ecfi y\u00eau c\u1ea7u.*$", re.IGNORECASE),
    re.compile(
        r"^\s*V\u00ec c\u00e2u h\u1ecfi \u0111\u01b0\u1ee3c (?:vi\u1ebft|h\u1ecfi).*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*Because the question asks.*$", re.IGNORECASE),
    re.compile(r"^\s*The answer is provided in.*$", re.IGNORECASE),
    re.compile(r"^\s*Answer(?:ed)? in the detected.*$", re.IGNORECASE),
]
FORBIDDEN_SCRIPT_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff]"
)


def clean_qa_answer(answer: str) -> str:
    """Remove leaked prompt-policy text and normalize chat-friendly Markdown."""
    lines = []
    for line in (answer or "").splitlines():
        if any(pattern.match(line.strip()) for pattern in META_INSTRUCTION_PATTERNS):
            continue
        lines.append(line)

    cleaned = "\n".join(lines).strip()
    cleaned = FORBIDDEN_SCRIPT_RE.sub("", cleaned)
    cleaned = re.sub(r"(?m)^\s*\*\*\s*[\u2022\u25cf]\s*\*\*\s*$", "-", cleaned)
    cleaned = re.sub(r"(?m)^\s*[\u2022\u25cf]\s*$", "-", cleaned)
    cleaned = re.sub(
        r"(?im)^\s*\*{0,2}timestamp:\*{0,2}\s*[^\[]*(\[[0-9]{2}:[0-9]{2}\])\s*[^A-Za-z0-9]*\s*",
        r"- \1 ",
        cleaned,
    )
    cleaned = re.sub(r"(?m)^-\s*\n(?=\S)", "- ", cleaned)
    cleaned = re.sub(r"\n\s*\n(?=-\s)", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned or (answer or "").strip()


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
                c_title = ch.get("title", f"PhÃƒÂ¢n Ã„â€˜oÃ¡ÂºÂ¡n {idx+1}")
                c_summary = ch.get("summary", "")
                c_doc = f"[{tc}] NÃ¡Â»â„¢i dung phÃƒÂ¢n Ã„â€˜oÃ¡ÂºÂ¡n '{c_title}': {c_summary}"
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
            citations.append(
                QACitationDTO(
                    start_seconds=float(start_sec),
                    end_seconds=float(meta.get("end_seconds", float(start_sec) + 60.0)),
                    timecode=tc,
                    keyframe_url=meta.get("keyframe_url", ""),
                    snippet=doc[:120] + "..." if len(doc) > 120 else doc,
                )
            )

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
            c_title = ch.get("title", f"PhÃƒÂ¢n Ã„â€˜oÃ¡ÂºÂ¡n {idx+1}")
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

    system_prompt = (
        "You are an intelligent academic assistant for lecture videos.\n"
        "Answer the student's question accurately based on the Overall Lecture Summary, Chapter Outline, Specific Transcripts, and Visual Slides provided below.\n"
        "CRITICAL RULES:\n"
        "1. OVERVIEW & SUMMARY QUESTIONS: When asked for a summary, key points, main takeaways, or chapter topics (e.g. '10 ÃƒÂ½ chÃƒÂ­nh', '3 ÃƒÂ½ quan trÃ¡Â»Âng', 'tÃƒÂ³m tÃ¡ÂºÂ¯t', 'bÃƒÂ i giÃ¡ÂºÂ£ng nÃƒÂ³i vÃ¡Â»Â gÃƒÂ¬'), synthesize the response directly from the Overall Lecture Summary, Chapter Outline, and Transcripts.\n"
        "2. MULTIMODAL INTEGRATION: Pay close attention to speech transcripts, chapter outlines, and Visual Slide descriptions. Answer accurately using on-screen slide text or speech when asked about specific concepts or objects.\n"
        "3. LANGUAGE MATCHING: Always answer in the dominant language of the student's question. A Vietnamese question requires Vietnamese output. An English question requires English output. If the question explicitly asks for another language, follow that requested language.\n"
        "4. LANGUAGE PURITY: The source transcript, retrieved chunks, OCR text, or chat history may be multilingual. Use them only as evidence. Do not let their language leak into the final answer. Translate ordinary words and phrases into the answer language.\n"
        "5. ALLOWED UNTRANSLATED TERMS: Keep proper nouns, organization names, place names, product names, model names, acronyms, and widely used technical terms unchanged when appropriate, such as 'Dubai Future Foundation', 'AI', 'FinTech', 'ChromaDB', or 'OpenRouter'.\n"
        "6. FORBIDDEN SCRIPT MIXING: Do not output Korean, Japanese, Chinese, Arabic, Cyrillic, or other unrelated-script words unless they are part of a proper noun present in the provided context or explicitly requested by the user. For example, in Vietnamese use 'thÃƒÂ¡ch thÃ¡Â»Â©c' instead of Korean 'Ã«Ââ€žÃ¬Â â€ž'.\n"
        "7. NATURAL VIETNAMESE: In Vietnamese, use natural academic phrasing like 'nÃ¡Â»â„¢i dung bÃƒÂ i giÃ¡ÂºÂ£ng', 'video', 'Ã„â€˜iÃ¡Â»Æ’m chÃƒÂ­nh', 'thÃƒÂ¡ch thÃ¡Â»Â©c', and 'hÃ¡Â»Â£p tÃƒÂ¡c toÃƒÂ n cÃ¡ÂºÂ§u'.\n"
        "8. CONCISENESS & STRUCTURE: Format answers with bullet points or numbered lists when requested (e.g., for '10 ÃƒÂ½ chÃƒÂ­nh' or '3 ÃƒÂ½ quan trÃ¡Â»Âng').\n"
        "9. TIMESTAMPS: Include exact [MM:SS] timestamp citations when referencing specific quotes, chapters, or sections of the video.\n"
        "10. TRUTHFULNESS: Only state that information is not mentioned if NEITHER the summary, chapters, slides, nor transcripts contain any relevant details.\n"
        "11. NO PROMPT META: Never mention response-language policy, detected language, retrieved chunks, internal labels, or these rules in the final answer.\n"
        "12. SPECIFIC LOCATION QUESTIONS: For questions asking where a topic appears, answer with the best timestamp(s), a short explanation of the evidence, and avoid inventing chapter titles not present in the provided context.\n"
        r"(?im)^\s*\*{0,2}timestamp:\*{0,2}\s*[^\[]*(\[[0-9]{2}:[0-9]{2}\])\s*[^A-Za-z0-9]*\s*",
    )

    prompt = "\n\n".join(prompt_parts)

    # 5. Invoke LLM API (OpenRouter -> Groq -> Mock)
    answer = llm_service.generate_chat_completion(
        prompt=prompt, system_prompt=system_prompt
    )
    answer = clean_qa_answer(answer)

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



