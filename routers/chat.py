import logging
import re
from collections import defaultdict
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client.http.models import PayloadSchemaType

from chat import find_relevant_chunks
from routers.deps import (
    COLLECTION_NAME,
    LLM_MODEL,
    get_vector_store,
    openai_client,
    qdrant_client,
    supabase,
)

router = APIRouter()
logger = logging.getLogger(__name__)

TOP_K_PER_DOCUMENT = 5
FALLBACK_TOP_K = 8
MAX_ROUTED_DOCUMENTS = 12

DOC_ID_PATTERN = re.compile(r"po[\s\-#]*([0-9]{3,})", re.IGNORECASE)


class ChatRequest(BaseModel):
    username: str
    query: str
    file_names: list[str] = []
    history: list[dict] = []


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _extract_doc_ids(query: str) -> List[str]:
    matches = DOC_ID_PATTERN.findall(query)
    return list(dict.fromkeys(matches)) if matches else []


def _extract_filename_hints(query: str) -> List[str]:
    """Pull filename-like tokens from the query (e.g., PO_1234.pdf or invoice123).

    This allows queries like "what's the total of PO_4702057395.pdf" to route
    without relying on PO regex extraction.
    """

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{3,}\.?", query)
    # Keep a small, deduped set to avoid over-routing
    uniq = []
    seen = set()
    for tok in tokens:
        norm = _normalize_token(tok)
        if norm and norm not in seen:
            uniq.append(tok)
            seen.add(norm)
    return uniq


def _get_user_files(username: str) -> List[str]:
    try:
        result = (
            supabase.table("useruploads")
            .select("file_name")
            .eq("username", username)
            .execute()
        )
        return [row.get("file_name") for row in result.data if row.get("file_name")]
    except Exception as exc:
        logger.warning("Could not fetch user files for %s: %s", username, exc)
        return []


def _route_candidates(
    username: str, doc_ids: List[str], filename_hints: List[str], explicit_files: List[str]
) -> List[str]:
    user_files = _get_user_files(username)
    candidates: List[str] = list(dict.fromkeys(explicit_files or []))

    search_tokens = doc_ids + filename_hints
    if search_tokens:
        normalized_tokens = [_normalize_token(t) for t in search_tokens if t]
        for fname in user_files:
            norm_name = _normalize_token(fname)
            if any(tok and tok in norm_name for tok in normalized_tokens):
                candidates.append(fname)

    # Deduplicate while preserving order and cap to avoid huge fan-out
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) > MAX_ROUTED_DOCUMENTS:
        candidates = candidates[:MAX_ROUTED_DOCUMENTS]
    return candidates


@router.post("/chat")
async def chat(request: ChatRequest):
    vector_store = get_vector_store()
    if not vector_store:
        raise HTTPException(status_code=404, detail="No collection found. Upload PDFs first.")

    doc_ids = _extract_doc_ids(request.query)
    filename_hints = _extract_filename_hints(request.query)
    candidate_files = _route_candidates(request.username, doc_ids, filename_hints, request.file_names)

    if candidate_files or doc_ids:
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.file_name",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    logger.info(
        "Received query for user=%s: %s | doc_ids=%s | filename_hints=%s | candidates=%s",
        request.username,
        request.query,
        doc_ids,
        filename_hints,
        candidate_files,
    )
    results = await find_relevant_chunks(
        user_query=request.query,
        vector_db=vector_store,
        username=request.username,
        file_names=candidate_files or None,
        doc_ids=doc_ids or None,
        per_doc_k=TOP_K_PER_DOCUMENT,
        fallback_k=FALLBACK_TOP_K,
    )
    if doc_ids and not results:
        logger.info("Doc-id filtered search returned 0; retrying without doc_id filter")
        results = await find_relevant_chunks(
            user_query=request.query,
            vector_db=vector_store,
            username=request.username,
            file_names=candidate_files or None,
            doc_ids=None,
            per_doc_k=TOP_K_PER_DOCUMENT,
            fallback_k=FALLBACK_TOP_K,
        )
    logger.info("Similarity search returned %s results for user=%s", len(results), request.username)

    if not results:
        logger.warning("No matches found for user=%s query=%s", request.username, request.query)
        return {
            "answer": "No content found for this user. Please upload PDFs first.",
            "matches": [],
        }

    grouped: Dict[str, list] = defaultdict(list)
    for result in results:
        grouped[result.metadata.get("file_name") or "Unknown Document"].append(result)

    context_lines: List[str] = []
    for file_name, docs in grouped.items():
        context_lines.append(f"Document: {file_name}")
        for doc in docs:
            page_number = doc.metadata.get("page_number") or (doc.metadata.get("page") or 0) + 1
            doc_id = doc.metadata.get("doc_id")
            prefix = f"- Document ID: {doc_id} | Page {page_number}" if doc_id else f"- Page {page_number}"
            context_lines.append(f"{prefix}: {doc.page_content}")

    structured_context = "\n".join(context_lines)

    system_prompt = (
        "You are an AI analyst for procurement PDFs (POs, invoices, revisions). "
        "Use only the provided context grouped by document and page. "
        "Return the answer or a concise table (e.g., PO Number, Delivery Date, Item, Value). "
        "Do NOT include source reference sections, citations, or file/page callouts in the final answer; keep it to the requested facts only. "
        "If data for a field is missing in the context, state that it is not found. Do not invent values.\n\n"
        f"Documents:\n{structured_context}"
    )

    prior_messages = []
    for msg in request.history or []:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            prior_messages.append({"role": role, "content": content})

    response = openai_client.responses.create(
        model=LLM_MODEL,
        input=[{"role": "system", "content": system_prompt}, *prior_messages, {"role": "user", "content": request.query}],
    )
    logger.info(
        "Responded to user=%s with %s matches; first file=%s",
        request.username,
        len(results),
        results[0].metadata.get("file_name") if results else None,
    )

    # Preserve candidate ordering when returning references so the UI can navigate predictably
    ordered_files = candidate_files or list(grouped.keys())
    match_payloads = []
    for fname in ordered_files:
        for result in grouped.get(fname, []):
            match_payloads.append(
                {
                    "file_name": result.metadata.get("file_name"),
                    "page_index": result.metadata.get("page"),
                    "page_label": result.metadata.get("page_label")
                    or str((result.metadata.get("page") or 0) + 1),
                    "page_number": result.metadata.get("page_number")
                    or (result.metadata.get("page") or 0) + 1,
                    "doc_id": result.metadata.get("doc_id"),
                    "chunk_id": result.metadata.get("chunk_id"),
                    "chunk_text": result.page_content,
                }
            )

    return {
        "answer": response.output_text,
        "matches": match_payloads,
    }
