from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the support knowledge base (FAQs and manuals) for relevant information to answer a customer's question."""
    results = vectorstore.similarity_search(query, k=3)

    if not results:
        return "No relevant information found in the knowledge base."

    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)