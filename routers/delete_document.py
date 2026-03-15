import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue

from routers.deps import COLLECTION_NAME, STORAGE_BUCKET, qdrant_client, supabase

router = APIRouter()
logger = logging.getLogger(__name__)


class DeleteDocumentRequest(BaseModel):
    username: str
    file_name: str


@router.delete("/delete-document")
async def delete_document(req: DeleteDocumentRequest):
    username = req.username
    file_name = req.file_name

    errors = []

    storage_path = f"{username}/{file_name}"
    try:
        supabase.storage.from_(STORAGE_BUCKET).remove([storage_path])
        logger.info("Deleted '%s' from storage", storage_path)
    except Exception as exc:
        logger.error("Storage deletion failed for '%s': %s", storage_path, exc)
        errors.append(f"Storage deletion failed: {exc}")

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
        logger.info("Deleted Qdrant vectors for file='%s' user='%s'", file_name, username)
    except Exception as exc:
        logger.error(
            "Qdrant deletion failed for file='%s' user='%s': %s", file_name, username, exc
        )
        errors.append(f"Vector deletion failed: {exc}")

    try:
        supabase.table("useruploads").delete().eq("username", username).eq(
            "file_name", file_name
        ).execute()
        logger.info("Deleted metadata row for file='%s' user='%s'", file_name, username)
    except Exception as exc:
        logger.error(
            "Metadata deletion failed for file='%s' user='%s': %s", file_name, username, exc
        )
        errors.append(f"Metadata deletion failed: {exc}")

    if errors:
        raise HTTPException(
            status_code=500,
            detail=f"Document deleted with errors: {'; '.join(errors)}",
        )

    return {"message": f"Document '{file_name}' deleted successfully."}
