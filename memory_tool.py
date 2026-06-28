# memory_tool.py
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import datetime

load_dotenv()

CHROMA_DIR = "chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Notice: collection_name is different from your FAQ one!
# This keeps customer facts SEPARATE from FAQ knowledge, in the same database
customer_memory = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="customer_facts"
)


@tool
def remember_fact(customer_id: str, fact: str) -> str:
    """Save an important fact about a customer for future conversations.
    Use this when the customer shares something worth remembering long-term,
    like a complaint, a preference, or an issue with an order."""

    customer_memory.add_texts(
        texts=[fact],
        metadatas=[{
            "customer_id": customer_id,
            "saved_at": datetime.datetime.now().isoformat()
        }]
    )
    return f"Noted and saved: {fact}"


@tool
def recall_facts(customer_id: str, question: str) -> str:
    """Search past saved facts about a specific customer to answer
    questions about their history, like 'has this customer complained before?'"""

    results = customer_memory.similarity_search(
        query=question,
        k=3,
        filter={"customer_id": customer_id}   # only this customer's facts
    )

    if not results:
        return "No previous facts found for this customer."

    facts = [doc.page_content for doc in results]
    return "\n".join(f"- {f}" for f in facts)