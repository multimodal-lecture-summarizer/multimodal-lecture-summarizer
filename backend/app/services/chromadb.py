import logging
import os
from typing import List, Dict, Any, Optional
import chromadb
from app.core.config import settings

logger = logging.getLogger(__name__)


def is_chromadb_responsive(host: str, port: Optional[int], ssl: bool = False, api_key: str = "", timeout: float = 2.0) -> bool:
    import urllib.request
    import urllib.error
    if not host:
        return False
    p = int(port) if port else (443 if ssl else 8000)
    proto = "https" if ssl or p == 443 else "http"
    for endpoint in ("/api/v2/heartbeat", "/api/v1/heartbeat"):
        try:
            url = f"{proto}://{host}:{p}{endpoint}"
            req = urllib.request.Request(url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.getcode() == 200:
                    return True
        except urllib.error.HTTPError as e:
            # If the server returned auth/deprecation HTTP errors (like 401, 403, 410, 501), it is responsive.
            if e.code in (401, 403, 410, 501):
                return True
        except Exception:
            pass
    return False


def resolve_persist_dir(configured: Optional[str] = None) -> str:
    """Shared on-disk index used when the Chroma server is not running."""
    service_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(service_dir, "..", "..", ".."))
    default_dir = os.path.join(project_root, "cache", "chroma")
    raw = (configured or "").strip()
    if not raw or raw.replace("\\", "/") in ("./cache/chroma", "cache/chroma"):
        return default_dir
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(project_root, raw))


class ChromaDBService:
    def __init__(self):
        self.host = settings.CHROMADB_HOST
        self.port = settings.CHROMADB_PORT
        self.ssl = settings.CHROMADB_SSL
        self.api_key = settings.CHROMADB_API_KEY
        self.tenant = settings.CHROMADB_TENANT
        self.database = settings.CHROMADB_DATABASE
        self.collection_name = settings.CHROMADB_COLLECTION_NAME
        self.persist_dir = resolve_persist_dir(getattr(settings, "CHROMADB_PERSIST_DIR", ""))
        self.enabled = False
        self.client = None
        self.collection = None
        self._initialized = False
        self._embedding_function = None

    def _collection_kwargs(self) -> dict:
        kwargs = {"name": self.collection_name}
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function
        return kwargs

    def _open_remote(self) -> None:
        if self.api_key:
            self.client = chromadb.CloudClient(
                tenant=self.tenant,
                database=self.database,
                api_key=self.api_key,
                cloud_host=self.host or "api.trychroma.com",
                cloud_port=self.port if self.port else 443,
                enable_ssl=self.ssl,
            )
        else:
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port if self.port else 8000,
                ssl=self.ssl,
                tenant=self.tenant,
                database=self.database,
            )
        self.collection = self.client.get_or_create_collection(**self._collection_kwargs())
        self.enabled = True

    def _open_persistent(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(**self._collection_kwargs())
        self.enabled = True

    def _connect_once(self) -> bool:
        uvicorn_logger = logging.getLogger("uvicorn.error")
        if is_chromadb_responsive(self.host, self.port, self.ssl, self.api_key, timeout=2.0):
            try:
                self._open_remote()
                uvicorn_logger.info(f"Connected to ChromaDB at {self.host}:{self.port} successfully.")
                return True
            except Exception as exc:
                uvicorn_logger.warning(
                    f"ChromaDB at {self.host}:{self.port} responded but the client failed: {exc}"
                )

        try:
            self._open_persistent()
            uvicorn_logger.info(f"Using on-disk Chroma index at {self.persist_dir}.")
            return True
        except Exception as exc:
            self.enabled = False
            self.client = None
            self.collection = None
            uvicorn_logger.warning(
                f"ChromaDB is unavailable at {self.host}:{self.port} and {self.persist_dir}: {exc}. "
                "Index writes will fail instead of using an in-memory store."
            )
            return False

    def _ensure_connection(self):
        if self._initialized:
            return
        self._connect_once()
        self._initialized = True

    def _clean_metadata(self, video_id: str, metadata: dict) -> dict:
        clean: Dict[str, Any] = {"video_id": str(video_id)}
        for key, value in (metadata or {}).items():
            if key == "video_id":
                continue
            if isinstance(value, bool) or isinstance(value, (int, float, str)):
                clean[key] = value
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)
        clean.setdefault("start_seconds", 0.0)
        clean.setdefault("end_seconds", 0.0)
        clean.setdefault("timecode", "")
        clean.setdefault("keyframe_url", "")
        return clean

    def _chunk_id(self, video_id: str, metadata: dict, seen: Dict[str, int]) -> str:
        start = metadata.get("start_seconds") or 0.0
        try:
            start_ms = int(round(float(start) * 1000))
        except (TypeError, ValueError):
            start_ms = 0
        base = f"{video_id}:{start_ms}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        return base if count == 0 else f"{base}:{count}"

    def replace_transcript_chunks(
        self, video_id: str, chunks: List[str], metadatas: List[dict]
    ) -> bool:
        """Delete this video's chunks, write the new set, and require the stored count to match."""
        self._ensure_connection()
        video_id_str = str(video_id)
        if not self.enabled or self.collection is None:
            logger.error(f"ChromaDB is unavailable. Refusing to store chunks for video {video_id_str}.")
            return False
        if not chunks or len(chunks) != len(metadatas):
            logger.error(f"Refusing to index video {video_id_str}: chunk and metadata counts differ or are empty.")
            return False

        clean_metas = [self._clean_metadata(video_id_str, meta) for meta in metadatas]
        seen: Dict[str, int] = {}
        ids = [self._chunk_id(video_id_str, meta, seen) for meta in clean_metas]
        try:
            self.collection.delete(where={"video_id": video_id_str})
            self.collection.upsert(ids=ids, documents=list(chunks), metadatas=clean_metas)
            stored = self.collection.get(where={"video_id": video_id_str}, include=[])
            stored_count = len((stored or {}).get("ids") or [])
        except Exception as exc:
            logger.error(f"Error replacing Chroma chunks for video {video_id_str}: {exc}")
            return False

        if stored_count != len(chunks):
            logger.error(
                f"Chroma count mismatch for video {video_id_str}: stored {stored_count}, expected {len(chunks)}."
            )
            return False
        logger.info(f"Replaced {stored_count} Chroma chunks for video {video_id_str}.")
        return True

    def add_transcript_chunks(
        self, video_id: str, chunks: List[str], metadatas: List[dict]
    ) -> bool:
        """Replace the video index. A durable write is the only successful result."""
        return self.replace_transcript_chunks(video_id, chunks, metadatas)

    def query_similar_chunks_with_metadata(
        self, video_id: str, query: str, limit: int = 4
    ) -> List[Dict[str, Any]]:
        """Return stored chunks for this video. A missing index returns an empty list."""
        self._ensure_connection()
        video_id_str = str(video_id)
        if not self.enabled or self.collection is None:
            logger.error(f"ChromaDB is unavailable. No chunks returned for video {video_id_str}.")
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where={"video_id": video_id_str},
            )
        except Exception as exc:
            logger.error(f"Error querying ChromaDB with metadata: {exc}")
            return []

        documents = (results or {}).get("documents") or []
        metadatas = (results or {}).get("metadatas") or []
        docs = documents[0] if documents else []
        metas = metadatas[0] if metadatas else []
        items = []
        for doc, meta in zip(docs, metas):
            if not doc:
                continue
            items.append({"document": doc, "metadata": meta or {}})
        logger.info(f"Retrieved {len(items)} chunks for video {video_id_str}.")
        return items

    def query_similar_chunks(self, video_id: str, query: str, limit: int = 3) -> List[str]:
        """Queries the vector database for text chunks relevant to the user query (plain text)."""
        items = self.query_similar_chunks_with_metadata(video_id, query, limit)
        return [item["document"] for item in items]

    def count_transcript_chunks(self, video_id: Any) -> int:
        """Return the number of chunks stored for a video, or -1 when Chroma cannot be read."""
        self._ensure_connection()
        if not self.enabled or self.collection is None:
            return -1
        try:
            stored = self.collection.get(where={"video_id": str(video_id)}, include=[])
            return len((stored or {}).get("ids") or [])
        except Exception as exc:
            logger.error(f"Error counting Chroma chunks for video {video_id}: {exc}")
            return -1

    def delete_transcript_chunks(self, video_id: Any) -> bool:
        """Deletes all transcript chunks associated with a video_id from ChromaDB."""
        self._ensure_connection()
        video_id_str = str(video_id)
        if not self.enabled or self.collection is None:
            logger.error(f"ChromaDB is unavailable. Could not delete chunks for video {video_id_str}.")
            return False
        try:
            self.collection.delete(where={"video_id": video_id_str})
            logger.info(f"Successfully deleted chunks from ChromaDB for video {video_id_str}")
            return True
        except Exception as exc:
            logger.error(f"Error deleting chunks from ChromaDB: {exc}")
            return False

    def verify_connection(self, retries: int = 5, delay: float = 2.0):
        """
        Verifies a durable Chroma index on startup.
        Uses the Chroma server when it responds, otherwise the shared on-disk index.
        A non-development process stops if neither store can be opened.
        """
        import time
        last_error = None
        uvicorn_logger = logging.getLogger("uvicorn.error")
        for attempt in range(1, retries + 1):
            try:
                if self._connect_once():
                    self._initialized = True
                    return
                last_error = RuntimeError("ChromaDB server and on-disk index are both unavailable.")
            except Exception as exc:
                last_error = exc
                uvicorn_logger.warning(f"ChromaDB connection attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(delay)

        self._initialized = True
        self.enabled = False
        if settings.APP_ENV == "development":
            uvicorn_logger.warning(
                f"All ChromaDB connection attempts failed. Queries return no chunks and "
                f"index builds fail since APP_ENV is '{settings.APP_ENV}'."
            )
        else:
            uvicorn_logger.error("ChromaDB connection failed and fallback is disabled in non-development environment.")
            raise RuntimeError(
                f"Failed to connect to ChromaDB at {self.host}:{self.port} after {retries} attempts: {last_error}"
            )


chromadb_service = ChromaDBService()
