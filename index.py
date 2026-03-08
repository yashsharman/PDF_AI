from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).parent /"resume.pdf"

#Loading the PDF document
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


#Splitting the text into chunks
chunks = text_splitter.split_documents(documents=pages)

#Vectorization

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    url="http://localhost:6333",
    collection_name="learnin g_rag",
)

print("Vector store created successfully!")