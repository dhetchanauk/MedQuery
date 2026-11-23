
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import shutil


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vector_store(documents):

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    # Save vector database locally
    db_path = "vector_db/medical_index"

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    vectorstore.save_local(db_path)

    return vectorstore


def load_vector_store():

    db_path = "vector_db/medical_index"

    embeddings = get_embeddings()

    if not os.path.exists(db_path):
        raise FileNotFoundError("Please upload and process PDFs first.")

    vectorstore = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore
