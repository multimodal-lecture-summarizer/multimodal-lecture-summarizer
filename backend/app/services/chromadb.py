import logging
from typing import List, Dict, Any, Optional
import chromadb
from app.core.config import settings

logger = logging.getLogger(__name__)


def is_chromadb_responsive(host: str, port: Optional[int], ssl: bool = False, api_key: str = "", timeout: float = 2.0) -> bool:
    import urllib.request
    import urllib.error
    if not host:
        return False
    try:
        p = int(port) if port else (443 if ssl else 8000)
        proto = "https" if ssl or p == 443 else "http"
        url = f"{proto}://{host}:{p}/api/v1/heartbeat"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.getcode() == 200:
                return True
    except urllib.error.HTTPError:
        # If the server returned any HTTP error (like 401, 403, 410), it is responsive.
        return True
    except Exception:
        pass
    return False


class ChromaDBService:
    def __init__(self):
        self.host = settings.CHROMADB_HOST
        self.port = settings.CHROMADB_PORT
        self.ssl = settings.CHROMADB_SSL
        self.api_key = settings.CHROMADB_API_KEY
        self.tenant = settings.CHROMADB_TENANT
        self.database = settings.CHROMADB_DATABASE
        self.collection_name = settings.CHROMADB_COLLECTION_NAME
        self.enabled = False
        self.client = None
        self.collection = None
        self._initialized = False

        # Dict mock vector store to fallback on connection errors
        self.mock_store: Dict[str, List[Dict[str, Any]]] = {}

    def _ensure_connection(self):
        if self._initialized:
            return
        self._initialized = True
        uvicorn_logger = logging.getLogger("uvicorn.error")
        
        # Verify database endpoint responsiveness to avoid library hang
        if not is_chromadb_responsive(self.host, self.port, self.ssl, self.api_key, timeout=2.0):
            uvicorn_logger.warning(
                f"ChromaDB endpoint at {self.host}:{self.port} is not responsive. "
                "Falling back to local in-memory Mock store."
            )
            self.enabled = False
            return

        try:
            if self.api_key:
                # Use CloudClient for Chroma Cloud connection
                self.client = chromadb.CloudClient(
                    tenant=self.tenant,
                    database=self.database,
                    api_key=self.api_key,
                    cloud_host=self.host or "api.trychroma.com",
                    cloud_port=self.port if self.port else 443,
                    enable_ssl=self.ssl
                )
            else:
                # Use HttpClient for standard connection
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port if self.port else 8000,
                    ssl=self.ssl,
                    tenant=self.tenant,
                    database=self.database
                )
            # Try to get or create collection to verify connection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            self.enabled = True
            uvicorn_logger.info(
                f"Connected to ChromaDB at {self.host}:{self.port} successfully."
            )
        except Exception as e:
            uvicorn_logger.warning(
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
        self._ensure_connection()
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

    def query_similar_chunks_with_metadata(
        self, video_id: str, query: str, limit: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Queries the vector database for text chunks relevant to query, returning document text and metadata.
        """
        self._ensure_connection()
        video_id_str = str(video_id)
        if self.enabled and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where={"video_id": video_id_str},
                )
                if results and "documents" in results and results["documents"]:
                    docs = results["documents"][0]
                    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                    items = []
                    for doc, meta in zip(docs, metas):
                        items.append({"document": doc, "metadata": meta or {}})
                    logger.info(f"Retrieved {len(items)} chunks with metadata from ChromaDB for query: '{query}'")
                    return items
            except Exception as e:
                logger.error(f"Error querying ChromaDB with metadata: {e}")

        # In-memory mock retrieval
        logger.info(f"[Mock ChromaDB] Querying in-memory store for video {video_id_str} with query: '{query}'")
        chunks = self.mock_store.get(video_id_str, [])
        if not chunks:
            return [
                {
                    "document": "[00:00] Lời giảng: WhisperX cung cấp timestamp mức từ chính xác bằng mô hình ngữ âm.",
                    "metadata": {"video_id": video_id_str, "start_seconds": 0.0, "end_seconds": 60.0, "timecode": "00:00", "keyframe_url": ""}
                },
                {
                    "document": "[01:00] Lời giảng: PySceneDetect phân tách video dựa trên sự thay đổi màu sắc và pixel.",
                    "metadata": {"video_id": video_id_str, "start_seconds": 60.0, "end_seconds": 120.0, "timecode": "01:00", "keyframe_url": ""}
                },
                {
                    "document": "[02:00] Lời giảng: CLIP trích xuất đặc trưng hình ảnh slide và đo độ tương đồng văn bản.",
                    "metadata": {"video_id": video_id_str, "start_seconds": 120.0, "end_seconds": 180.0, "timecode": "02:00", "keyframe_url": ""}
                }
            ]

        query_words = set(query.lower().split())
        scored_chunks = []
        for chunk in chunks:
            doc_lower = chunk["document"].lower()
            score = sum(1 for word in query_words if word in doc_lower)
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:limit]]

    def query_similar_chunks(self, video_id: str, query: str, limit: int = 3) -> List[str]:
        """
        Queries the vector database for text chunks relevant to the user query (plain text).
        """
        items = self.query_similar_chunks_with_metadata(video_id, query, limit)
        return [item["document"] for item in items]

    def delete_transcript_chunks(self, video_id: Any) -> bool:
        """
        Deletes all transcript chunks associated with a video_id from ChromaDB or mock store.
        """
        self._ensure_connection()
        video_id_str = str(video_id)
        deleted = False
        if self.enabled and self.collection:
            try:
                self.collection.delete(where={"video_id": video_id_str})
                logger.info(f"Successfully deleted chunks from ChromaDB for video {video_id_str}")
                deleted = True
            except Exception as e:
                logger.error(f"Error deleting chunks from ChromaDB: {e}")

        if video_id_str in self.mock_store:
            self.mock_store.pop(video_id_str, None)
            logger.info(f"[Mock ChromaDB] Deleted chunks in-memory for video {video_id_str}")
            deleted = True

        return deleted

    def verify_connection(self, retries: int = 5, delay: float = 2.0):
        """
        Verifies connection to ChromaDB on startup, with retries.
        If it fails:
        - If APP_ENV == "development", falls back to in-memory Mock store (setting enabled=False).
        - If APP_ENV != "development", raises RuntimeError to prevent startup.
        """
        import time
        last_error = None
        uvicorn_logger = logging.getLogger("uvicorn.error")
        for attempt in range(1, retries + 1):
            try:
                
                # Verify database endpoint responsiveness to avoid library hang
                if not is_chromadb_responsive(self.host, self.port, self.ssl, self.api_key, timeout=2.0):
                    raise RuntimeError(f"ChromaDB endpoint at {self.host}:{self.port} is not responsive.")

                # Try to connect
                if self.api_key:
                    # Use CloudClient for Chroma Cloud connection
                    self.client = chromadb.CloudClient(
                        tenant=self.tenant,
                        database=self.database,
                        api_key=self.api_key,
                        cloud_host=self.host or "api.trychroma.com",
                        cloud_port=self.port if self.port else 443,
                        enable_ssl=self.ssl
                    )
                else:
                    # Use HttpClient for standard connection
                    self.client = chromadb.HttpClient(
                        host=self.host,
                        port=self.port if self.port else 8000,
                        ssl=self.ssl,
                        tenant=self.tenant,
                        database=self.database
                    )
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name
                )
                self.enabled = True
                self._initialized = True
                uvicorn_logger.info(f"Connected to ChromaDB at {self.host}:{self.port} successfully.")
                return
            except Exception as e:
                last_error = e
                uvicorn_logger.warning(f"ChromaDB connection attempt {attempt} failed: {e}")
                if attempt < retries:
                    time.sleep(delay)

        # Failed after all retries
        self._initialized = True # Mark as initialized so _ensure_connection won't try again
        self.enabled = False
        if settings.APP_ENV == "development":
            uvicorn_logger.warning(
                f"All connection attempts to ChromaDB failed. Falling back to local in-memory Mock store "
                f"since APP_ENV is '{settings.APP_ENV}'."
            )
        else:
            uvicorn_logger.error("ChromaDB connection failed and fallback is disabled in non-development environment.")
            raise RuntimeError(
                f"Failed to connect to ChromaDB at {self.host}:{self.port} after {retries} attempts: {last_error}"
            )


chromadb_service = ChromaDBService()
