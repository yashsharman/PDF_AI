import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI
from pydantic import BaseModel

from chat import find_relevant_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

QDRANT_URL = "https://7fe14b6f-1d8c-4f45-b837-9993dcd8a1a9.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rcdI89QMqiv8A3yaRA8VCazPKYcqWd4tyEwWHqGxW9U"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


app = FastAPI(title="PDF RAG API", version="0.1.0")

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
openai_client = OpenAI()


class ChatRequest(BaseModel):
    username: str
    query: str



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


async def _load_and_chunk(file: UploadFile, username: str, chunk_offset: int) -> list:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Each file must have a filename")

    with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".pdf") as tmp:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is empty")
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    loader = PyPDFLoader(str(tmp_path))
    pages = loader.load()
    tmp_path.unlink(missing_ok=True)

    if not pages:
        raise HTTPException(status_code=400, detail=f"No readable pages found in {file.filename}")

    chunks = text_splitter.split_documents(documents=pages)

    chunk_prefix = Path(file.filename).stem
    for idx, doc in enumerate(chunks, start=chunk_offset):
        doc.metadata = {
            **doc.metadata,
            "username": username,
            "file_name": file.filename,
            "chunk_id": f"{chunk_prefix}-chunk-{idx}",
        }

    logger.info(
        "Prepared %s chunks for file '%s' (user=%s)", len(chunks), file.filename, username
    )
    return chunks


@app.post("/upload-files")
async def upload_files(username: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    all_chunks = []
    chunk_counter = 0

    for file in files:
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
        chunks = await _load_and_chunk(file, username=username, chunk_offset=chunk_counter)
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

    return {"indexed_chunks": len(all_chunks), "collection": COLLECTION_NAME}


@app.post("/chat")
async def chat(request: ChatRequest):
    vector_store = _get_vector_store()
    if not vector_store:
        raise HTTPException(status_code=404, detail="No collection found. Upload PDFs first.")
    logger.info("Received query for user=%s: %s", request.username, request.query)
    results = await find_relevant_chunks(
        user_query=request.query, vector_db=vector_store, username=request.username
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
