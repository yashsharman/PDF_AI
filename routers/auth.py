import logging

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from routers.deps import SUPABASE_ANON_KEY, SUPABASE_URL, supabase

router = APIRouter()
logger = logging.getLogger(__name__)


class AuthRequest(BaseModel):
    email: str
    password: str


@router.post("/auth/signup")
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


@router.post("/auth/signin")
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


@router.post("/auth/signout")
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
            pass
    return {"message": "Signed out successfully."}


@router.get("/auth/me")
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
