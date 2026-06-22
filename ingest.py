import os
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"

def load_documents():
    docs = []

    # Load all .txt and .md FAQ files
    for path in glob.glob("docs/faq/*.txt") + glob.glob("docs/faq/*.md"):
        loader = TextLoader(path)
        docs.extend(loader.load())
        print(f"Loaded: {path}")

    # Load all PDF manuals
    for path in glob.glob("docs/manuals/*.pdf"):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
        print(f"Loaded: {path}")

    return docs

def main():
    print("Loading documents...")
    docs = load_documents()

    if not docs:
        print("No documents found in docs/faq or docs/manuals!")
        return

    print(f"\nTotal documents loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\nCreating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"\nDone! Knowledge base saved to ./{CHROMA_DIR}")
    print(f"Total chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    main()