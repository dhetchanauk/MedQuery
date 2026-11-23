
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.text_preprocessing import clean_text
import os
import tempfile


def process_pdf(uploaded_file):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        # Extract PDF pages
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Clean extracted text
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = uploaded_file.name

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        return chunks

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
