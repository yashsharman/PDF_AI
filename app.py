import io
import os
import logging
from pathlib import Path
from typing import List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client as SupabaseClient
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from openai import OpenAI
from pydantic import BaseModel

from chat import find_relevant_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# Use service role key on the backend to bypass RLS for storage and table operations
supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY)


app = FastAPI(title="PDF RAG API", version="0.1.0")


@app.on_event("startup")
def _ensure_payload_indexes():
    for field in ("metadata.username", "metadata.file_name"):
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Payload index on '%s' ensured.", field)
        except Exception as exc:
            # Index may already exist or collection may not exist yet — both are fine
            logger.debug("Payload index creation skipped for '%s': %s", field, exc)


# Open CORS so the API can be called from any origin (frontend/dev tools).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "pdf_rag"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1"
STORAGE_BUCKET = "useruploads"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
openai_client = OpenAI()


class ChatRequest(BaseModel):
    username: str
    query: str
    file_names: list[str] = []


class AuthRequest(BaseModel):
    email: str
    password: str


class TokenRequest(BaseModel):
    access_token: str



def _get_vector_store() -> QdrantVectorStore | None:
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


async def _load_and_chunk(contents: bytes, filename: str, username: str, chunk_offset: int) -> list:
    reader = PdfReader(io.BytesIO(contents))
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"page": i, "page_label": str(i + 1), "source": filename},
        )
        for i, page in enumerate(reader.pages)
    ]
    pages = [p for p in pages if p.page_content.strip()]

    if not pages:
        raise HTTPException(status_code=400, detail=f"No readable pages found in {filename}")

    chunks = text_splitter.split_documents(documents=pages)

    chunk_prefix = Path(filename).stem
    for idx, doc in enumerate(chunks, start=chunk_offset):
        doc.metadata = {
            **doc.metadata,
            "username": username,
            "file_name": filename,
            "chunk_id": f"{chunk_prefix}-chunk-{idx}",
        }

    logger.info(
        "Prepared %s chunks for file '%s' (user=%s)", len(chunks), filename, username
    )
    return chunks


@app.post("/upload-files")
async def upload_files(username: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    all_chunks = []
    chunk_counter = 0
    uploaded_file_meta = []

    for file in files:
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is empty")

        # ── Upload to Supabase Storage ───────────────────────────────────────
        storage_path = f"{username}/{file.filename}"
        try:
            supabase.storage.from_(STORAGE_BUCKET).upload(
                storage_path,
                contents,
                {"content-type": "application/pdf", "x-upsert": "true"},
            )
            logger.info("Uploaded '%s' to storage at '%s'", file.filename, storage_path)
        except Exception as exc:
            logger.error("Storage upload failed for '%s': %s", file.filename, exc)
            raise HTTPException(
                status_code=500, detail=f"Storage upload failed for {file.filename}: {exc}"
            )

        file_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)

        # ── Save file metadata ───────────────────────────────────────────────
        try:
            supabase.table("useruploads").upsert(
                {"username": username, "file_name": file.filename, "file_url": file_url},
                on_conflict="username,file_name",
            ).execute()
            logger.info("Saved metadata for '%s' (user=%s)", file.filename, username)
        except Exception as exc:
            logger.warning("Could not save file metadata for '%s': %s", file.filename, exc)

        uploaded_file_meta.append({"file_name": file.filename, "file_url": file_url})

        # ── Chunk and embed ──────────────────────────────────────────────────
        chunks = await _load_and_chunk(contents, file.filename, username, chunk_counter)
        chunk_counter += len(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks created from uploads")

    vector_store = _get_vector_store()
    if vector_store:
        vector_store.add_documents(documents=all_chunks)
        logger.info("Appended %s chunks to existing collection '%s'", len(all_chunks), COLLECTION_NAME)
    else:
        QdrantVectorStore.from_documents(
            documents=all_chunks,
            embedding=embeddings_model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
        )
        logger.info("Created new collection '%s' with %s chunks", COLLECTION_NAME, len(all_chunks))

    try:
        for field in ("metadata.username", "metadata.file_name"):
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Already exists
    except Exception:
        pass

    return {
        "indexed_chunks": len(all_chunks),
        "collection": COLLECTION_NAME,
        "files": uploaded_file_meta,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    vector_store = _get_vector_store()
    if not vector_store:
        raise HTTPException(status_code=404, detail="No collection found. Upload PDFs first.")
    # Ensure file_name index exists when a per-file filter will be applied
    if request.file_names:
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.file_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Already exists
    logger.info("Received query for user=%s: %s", request.username, request.query)
    results = await find_relevant_chunks(
        user_query=request.query,
        vector_db=vector_store,
        username=request.username,
        file_names=request.file_names or None,
    )
    logger.info("Similarity search returned %s results for user=%s", len(results), request.username)

    if not results:
        logger.warning("No matches found for user=%s query=%s", request.username, request.query)
        return {
            "answer": "No content found for this user. Please upload PDFs first.",
            "matches": [],
        }

    context_blocks = []
    for result in results:
        page_info = result.metadata.get("page_label") or result.metadata.get("page")
        context_blocks.append(
            f"Page Content: {result.page_content}\nPage Number: {page_info}\nFile: {result.metadata.get('file_name')}"
        )

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided PDF context. "
        "Use only the context to answer and reference the page numbers when possible."
        "If the context does not contain the information needed to answer the question, respond with 'I Think there is no relevant information to answer your question.'\n\n"
        f"CONTEXT:\n{context}"
    )

    response = openai_client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.query},
        ],
    )
    print(f"🤖 Response: {response.output_text}")
    logger.info(
        "Responded to user=%s with %s matches; first file=%s",
        request.username,
        len(results),
        results[0].metadata.get("file_name") if results else None,
    )
    return {
        "answer": response.output_text,
        "matches": [
            {
                "file_name": result.metadata.get("file_name"),
                # page_index: 0-based integer from PyPDFLoader — used by frontend for pdf.js (1-based) navigation
                "page_index": result.metadata.get("page"),
                # page_label: human-readable display label (falls back to 1-based string)
                "page_label": result.metadata.get("page_label")
                    or str((result.metadata.get("page") or 0) + 1),
                "chunk_id": result.metadata.get("chunk_id"),
                # chunk_text: the verbatim extracted text sent to the LLM — used for PDF highlight
                "chunk_text": result.page_content,
            }
            for result in results
        ],
    }


# ── Documents endpoint ────────────────────────────────────────────────────


@app.get("/user-documents")
async def user_documents(username: str):
    try:
        result = (
            supabase.table("useruploads")
            .select("file_name, file_url, uploaded_at")
            .eq("username", username)
            .order("uploaded_at", desc=True)
            .execute()
        )
        return result.data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Auth routes (Supabase) ─────────────────────────────────────────────────


@app.post("/auth/signup")
async def signup(req: AuthRequest):
    try:
        response = supabase.auth.sign_up({"email": req.email, "password": req.password})
        if response.user is None:
            raise HTTPException(status_code=400, detail="Signup failed. Please try again.")
        if response.session:
            return {
                "email": response.user.email,
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "message": "Account created successfully.",
            }
        # Email confirmation required
        return {
            "email": response.user.email,
            "access_token": None,
            "refresh_token": None,
            "message": "Please check your email to confirm your account before signing in.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/auth/signin")
async def signin(req: AuthRequest):
    try:
        response = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
        return {
            "email": response.user.email,
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
        }
    except Exception as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@app.post("/auth/signout")
async def signout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{SUPABASE_URL}/auth/v1/logout",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": SUPABASE_ANON_KEY,
                    },
                )
        except Exception:
            pass  # best-effort; client clears local token regardless
    return {"message": "Signed out successfully."}


@app.get("/auth/me")
async def get_me(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        result = supabase.auth.get_user(token)
        if not result.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        return {"email": result.user.email}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=str(exc))


# ── Delete document endpoint ───────────────────────────────────────────────

class DeleteDocumentRequest(BaseModel):
    username: str
    file_name: str


@app.delete("/delete-document")
async def delete_document(req: DeleteDocumentRequest):
    username = req.username
    file_name = req.file_name

    errors = []

    # 1. Delete from Supabase Storage
    storage_path = f"{username}/{file_name}"
    try:
        supabase.storage.from_(STORAGE_BUCKET).remove([storage_path])
        logger.info("Deleted '%s' from storage", storage_path)
    except Exception as exc:
        logger.error("Storage deletion failed for '%s': %s", storage_path, exc)
        errors.append(f"Storage deletion failed: {exc}")

    # 2. Delete all matching vectors from Qdrant
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.username",
                            match=MatchValue(value=username),
                        ),
                        FieldCondition(
                            key="metadata.file_name",
                            match=MatchValue(value=file_name),
                        ),
                    ]
                )
            ),
        )
        logger.info(
            "Deleted Qdrant vectors for file='%s' user='%s'", file_name, username
        )
    except Exception as exc:
        logger.error(
            "Qdrant deletion failed for file='%s' user='%s': %s", file_name, username, exc
        )
        errors.append(f"Vector deletion failed: {exc}")

    # 3. Delete metadata row from Supabase DB
    try:
        supabase.table("useruploads").delete().eq("username", username).eq(
            "file_name", file_name
        ).execute()
        logger.info(
            "Deleted metadata row for file='%s' user='%s'", file_name, username
        )
    except Exception as exc:
        logger.error(
            "Metadata deletion failed for file='%s' user='%s': %s", file_name, username, exc
        )
        errors.append(f"Metadata deletion failed: {exc}")

    if errors:
        # Partial failures — surface them so the caller knows what went wrong
        raise HTTPException(
            status_code=500,
            detail=f"Document deleted with errors: {'; '.join(errors)}",
        )

    return {"message": f"Document '{file_name}' deleted successfully."}
