import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import PayloadSchemaType

from routers.deps import (
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    STORAGE_BUCKET,
    embeddings_model,
    get_vector_store,
    load_and_chunk,
    qdrant_client,
    supabase,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload-files")
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

        try:
            supabase.table("useruploads").upsert(
                {"username": username, "file_name": file.filename, "file_url": file_url},
                on_conflict="username,file_name",
            ).execute()
            logger.info("Saved metadata for '%s' (user=%s)", file.filename, username)
        except Exception as exc:
            logger.warning("Could not save file metadata for '%s': %s", file.filename, exc)

        uploaded_file_meta.append({"file_name": file.filename, "file_url": file_url})

        chunks = load_and_chunk(contents, file.filename, username, chunk_counter)
        chunk_counter += len(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No chunks created from uploads")

    vector_store = get_vector_store()
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
        for field in ("metadata.username", "metadata.file_name", "metadata.doc_id"):
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
    except Exception:
        pass

    return {
        "indexed_chunks": len(all_chunks),
        "collection": COLLECTION_NAME,
        "files": uploaded_file_meta,
    }
