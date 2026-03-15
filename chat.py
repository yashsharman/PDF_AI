from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

client = openai_client = OpenAI()

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


async def find_relevant_chunks(
    user_query: str,
    vector_db: QdrantVectorStore,
    username: str | None = None,
    file_names: list[str] | None = None,
    doc_ids: list[str] | None = None,
    per_doc_k: int = 5,
    fallback_k: int = 8,
):
    """Run filtered similarity search with optional per-document capping.

    Filters always scope to username and optionally to doc_ids and file_names.
    When file_names are provided, each document is searched independently with
    a fixed top-k so no single document dominates. Otherwise a broader search
    across the user's corpus (still constrained by doc_ids if provided) is performed.
    """

    base_filter = [
        {
            "key": "metadata.username",
            "match": {"value": username},
        }
    ]

    if doc_ids:
        base_filter.append({"key": "metadata.doc_id", "match": {"any": list(dict.fromkeys(doc_ids))}})

    if file_names:
        results: list = []
        for name in dict.fromkeys(file_names):  # preserve order, drop dupes
            filter_clause = {
                "must": base_filter
                + [
                    {
                        "key": "metadata.file_name",
                        "match": {"value": name},
                    }
                ]
            }
            doc_hits = vector_db.similarity_search(user_query, k=per_doc_k, filter=filter_clause)
            results.extend(doc_hits)
        return results

    return vector_db.similarity_search(
        user_query,
        k=fallback_k,
        filter={"must": base_filter},
    )