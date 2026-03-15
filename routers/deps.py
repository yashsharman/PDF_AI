import io
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType
from supabase import Client as SupabaseClient, create_client

load_dotenv()

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

COLLECTION_NAME = "pdf_rag"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1"
STORAGE_BUCKET = "useruploads"
DOC_ID_REGEX = re.compile(r"po[\s\-#]*([0-9]{3,})", re.IGNORECASE)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
supabase: SupabaseClient = create_client(
    SUPABASE_URL, SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
)
embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
openai_client = OpenAI()


def ensure_payload_indexes() -> None:
    for field in (
        "metadata.username",
        "metadata.file_name",
        "metadata.doc_id",
    ):
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Payload index on '%s' ensured.", field)
        except Exception as exc:
            logger.debug("Payload index creation skipped for '%s': %s", field, exc)


def get_vector_store() -> QdrantVectorStore | None:
    try:
        store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
        )
        logger.info("Connected to existing collection '%s'", COLLECTION_NAME)
        return store
    except Exception as exc:
        logger.warning("No existing collection '%s' found: %s", COLLECTION_NAME, exc)
        return None


def _extract_doc_id(text: str) -> str | None:
    match = DOC_ID_REGEX.search(text)
    if not match:
        return None
    return match.group(1)


def load_and_chunk(
    contents: bytes, filename: str, username: str, chunk_offset: int
) -> list[Document]:
    reader = PdfReader(io.BytesIO(contents))

    pages: list[Document] = []
    all_text_parts: list[str] = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": idx, "page_label": str(idx + 1), "source": filename},
                )
            )
            all_text_parts.append(text)

    if not pages:
        raise HTTPException(status_code=400, detail=f"No readable pages found in {filename}")

    doc_id = _extract_doc_id(" \n".join(all_text_parts))
    chunk_prefix = Path(filename).stem

    for idx, doc in enumerate(pages, start=chunk_offset):
        doc.metadata = {
            **doc.metadata,
            "username": username,
            "file_name": filename,
            "chunk_id": f"{chunk_prefix}-page-{idx}",
            "page_number": int(doc.metadata.get("page", 0)) + 1,
            "chunk_text": doc.page_content,
        }
        if doc_id:
            doc.metadata["doc_id"] = doc_id

    logger.info(
        "Prepared %s page chunks for file '%s' (user=%s, doc_id=%s)",
        len(pages),
        filename,
        username,
        doc_id,
    )
    return pages
