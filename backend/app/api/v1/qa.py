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
from app.services.groq import groq_service

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

    # 2. Query similar chunks from ChromaDB
    retrieved_chunks = chromadb_service.query_similar_chunks(
        video_id=video_id,
        query=payload.question,
        limit=3,
    )

    # 3. Construct LLM context prompt
    context_str = "\n".join(
        [f"- [Context Chunk]: {chunk}" for chunk in retrieved_chunks]
    )
    system_prompt = (
        "You are an expert academic tutor. Answer the student's question based strictly on the provided video lecture transcript context chunks. "
        "Keep your response detailed, precise, and educational. If the context does not contain enough information to answer, state that."
    )
    prompt = (
        f"Video Context:\n{context_str}\n\n"
        f"Student Question: {payload.question}\n\n"
        f"Answer:"
    )

    # 4. Invoke Groq API
    answer = groq_service.generate_chat_completion(
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

    return BaseDTO(
        success=True,
        data=QALogDTO.model_validate(qa_log),
        message="Question answered successfully",
    )
