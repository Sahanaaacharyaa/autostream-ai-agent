import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_kb():
    with open("data/knowledge_base.json") as f:
        return json.load(f)


def build_docs(kb):
    docs = []

    for plan, desc in kb["pricing"].items():
        docs.append(Document(page_content=f"{plan} Plan: {desc}"))

    for k, v in kb["policies"].items():
        docs.append(Document(page_content=f"{k}: {v}"))

    return docs


def build_retriever():
    kb = load_kb()
    docs = build_docs(kb)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever(search_kwargs={"k": 3})