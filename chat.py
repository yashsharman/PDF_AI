from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

client = openai_client = OpenAI()

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


async def find_relevant_chunks(user_query: str, vector_db: QdrantVectorStore, k: int = 3, username: str = None) -> str:

    search_results = vector_db.similarity_search(user_query, k=3, filter={
    "must": [
      {
        "key": "metadata.username",
        "match": {
          "value": username
        }
      }
    ]
  })
    print(f"Search results: {search_results}")
    return search_results