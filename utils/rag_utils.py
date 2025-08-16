# import os
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# # 1. Load and chunk PDF
# def process_pdf(file_path):
#     loader = PyPDFLoader(file_path)
#     documents = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     return splitter.split_documents(documents)

# # 2. Embed and save to FAISS DB
# def save_to_vector_db(docs, db_dir="rag_store/"):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.from_documents(docs, embeddings)
#     db.save_local(db_dir)

# # 3. Load vector DB and retrieve top-k similar docs
# def retrieve_context(query, k=5, db_dir="rag_store/"):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
#     return db.similarity_search(query, k=k)

# rag_utils.py

import os
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# Define where to store RAG data
VECTOR_DB_DIR = "rag_vector_store"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_and_store_pdf(pdf_path):
    """Loads and embeds PDF into FAISS vector DB"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()

    if not os.path.exists(VECTOR_DB_DIR):
        os.makedirs(VECTOR_DB_DIR)

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_DB_DIR)


def load_vectorstore():
    """Loads the stored FAISS vector DB"""
    return FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)


def query_context(query, k=3):
    """Performs a similarity search and returns top-k chunks"""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return "\n".join([res.page_content for res in results])
