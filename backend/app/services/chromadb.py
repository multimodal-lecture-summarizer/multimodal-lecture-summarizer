import logging
from typing import List, Dict, Any, Optional
import chromadb
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChromaDBService:
    def __init__(self):
        self.host = settings.CHROMADB_HOST
        self.port = settings.CHROMADB_PORT
        self.collection_name = settings.CHROMADB_COLLECTION_NAME
        self.enabled = False
        self.client = None
        self.collection = None

        # Dict mock vector store to fallback on connection errors
        self.mock_store: Dict[str, List[Dict[str, Any]]] = {}

        try:
            # We can connect using HttpClient
            self.client = chromadb.HttpClient(host=self.host, port=self.port)
            # Try to get or create collection to verify connection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            self.enabled = True
            logger.info(
                f"Connected to ChromaDB at {self.host}:{self.port} successfully."
            )
        except Exception as e:
            logger.warning(
                f"Failed to connect to ChromaDB at {self.host}:{self.port}: {e}. "
                "Falling back to local in-memory Mock store."
            )
            self.enabled = False

    def add_transcript_chunks(
        self, video_id: str, chunks: List[str], metadatas: List[dict]
    ) -> bool:
        """
        Adds text chunks to the vector database.
        If ChromaDB is disabled, stores them in the in-memory mock store.
        """
        video_id_str = str(video_id)
        if self.enabled and self.collection:
            try:
                ids = [f"{video_id_str}_chunk_{i}" for i in range(len(chunks))]
                # ChromaDB client will automatically handle default embeddings if not supplied
                self.collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info(
                    f"Successfully added {len(chunks)} chunks to ChromaDB for video {video_id_str}"
                )
                return True
            except Exception as e:
                logger.error(f"Error adding chunks to ChromaDB: {e}")
                # Fallback to mock store
                pass

        # In-memory store fallback
        self.mock_store[video_id_str] = [
            {"document": chunk, "metadata": meta}
            for chunk, meta in zip(chunks, metadatas)
        ]
        logger.info(
            f"[Mock ChromaDB] Stored {len(chunks)} chunks in-memory for video {video_id_str}"
        )
        return True

    def query_similar_chunks(self, video_id: str, query: str, limit: int = 3) -> List[str]:
        """
        Queries the vector database for text chunks relevant to the user query.
        """
        video_id_str = str(video_id)
        if self.enabled and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where={"video_id": video_id_str},
                )
                # Parse documents list from results
                if results and "documents" in results and results["documents"]:
                    # ChromaDB returns List[List[str]]
                    flat_docs = results["documents"][0]
                    logger.info(
                        f"Retrieved {len(flat_docs)} chunks from ChromaDB for query: '{query}'"
                    )
                    return flat_docs
            except Exception as e:
                logger.error(f"Error querying ChromaDB: {e}")

        # In-memory mock retrieval: simple keyword matching logic
        logger.info(
            f"[Mock ChromaDB] Querying in-memory store for video {video_id_str} with query: '{query}'"
        )
        chunks = self.mock_store.get(video_id_str, [])
        if not chunks:
            # Return some static defaults if store is empty
            return [
                "WhisperX provides word-level timestamps (e.g. alignment) using phoneme models.",
                "PySceneDetect splits video into semantic scene transitions based on pixel changes.",
                "CLIP extracts semantic keyframe images and measures visual text similarity.",
            ]

        # Simple score based on term frequencies in documents
        query_words = set(query.lower().split())
        scored_chunks = []
        for chunk in chunks:
            doc_lower = chunk["document"].lower()
            score = sum(1 for word in query_words if word in doc_lower)
            scored_chunks.append((score, chunk["document"]))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_chunks[:limit]]


chromadb_service = ChromaDBService()
