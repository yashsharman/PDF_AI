from fastapi import APIRouter, HTTPException

from routers.deps import supabase

router = APIRouter()


@router.get("/user-documents")
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
