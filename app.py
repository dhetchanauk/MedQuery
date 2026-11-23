
import streamlit as st
from dotenv import load_dotenv
import os
import time
from collections import Counter

from src.pdf_processor import process_pdf
from src.vector_store import create_vector_store
from src.rag_pipeline import create_rag_chain, ask_question


# Load environment variables
load_dotenv()


# Check API key
if not os.getenv("GROQ_API_KEY"):
    st.error("Please configure your GROQ_API_KEY in the .env file.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="MedQuery",
    page_icon="🏥",
    layout="wide"
)


# Application title
st.title("🏥 MedQuery")
st.subheader("Healthcare Research & Document Assistant")

st.write(
    "Upload medical research papers and ask questions "
    "to explore their findings using AI."
)


# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload medical PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", use_container_width=True):

        if uploaded_files:

            with st.spinner("Processing documents..."):

                try:

                    all_chunks = []

                    # Process each uploaded PDF
                    for file in uploaded_files:

                        chunks = process_pdf(file)

                        print(f"\nPDF: {file.name}")
                        print(f"Number of chunks: {len(chunks)}")

                        all_chunks.extend(chunks)

                    # Check whether documents were extracted
                    if not all_chunks:

                        st.error(
                            "No readable text found in uploaded PDFs."
                        )

                    else:

                        # Document indexing report
                        print(
                            "\n========== DOCUMENT INDEXING REPORT =========="
                        )

                        document_counts = Counter(
                            doc.metadata.get("source", "Unknown")
                            for doc in all_chunks
                        )

                        for filename, count in document_counts.items():

                            print(f"{filename}: {count} chunks")

                        print("Total chunks:", len(all_chunks))

                        print(
                            "=============================================="
                        )

                        # Create FAISS vector database
                        vectorstore = create_vector_store(all_chunks)

                        # Initialize RAG chain
                        st.session_state.vectorstore = vectorstore

                        st.session_state.rag_chain = create_rag_chain(
                            vectorstore
                        )

                        # Clear previous conversation
                        st.session_state.messages = []

                        st.success(
                            f"Successfully processed "
                            f"{len(uploaded_files)} PDFs and created "
                            f"{len(all_chunks)} chunks!"
                        )

                        st.info(
                            "Your documents are now indexed "
                            "and ready for questions."
                        )

                except Exception as e:

                    st.error(f"Processing failed: {str(e)}")

                    print("PROCESSING ERROR:", str(e))

        else:

            st.warning("Please upload at least one PDF.")


    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# Main chat interface
st.divider()

if st.session_state.rag_chain is None:

    st.info(
        "👈 Upload your medical research PDFs and click "
        "'Process Documents' to get started."
    )

else:

    st.success(
        "Your medical document knowledge base is ready!"
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("View Sources"):

                    for source in message["sources"]:
                        st.write(source)


    # Chat input
    question = st.chat_input(
        "Ask something about your uploaded medical documents..."
    )

    if question:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)


        # Build previous conversation history
        history = []

        for message in st.session_state.messages[:-1][-6:]:

            history.append(
                (
                    "human" if message["role"] == "user" else "ai",
                    message["content"]
                )
            )


        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                try:

                    print("CURRENT QUESTION:", question)
                    print("CHAT HISTORY:", history)

                    # Measure RAG execution time
                    rag_start = time.time()

                    answer, source_docs = ask_question(
                        st.session_state.rag_chain,
                        question,
                        history
                    )

                    rag_end = time.time()

                    print(
                        f"RAG execution time: "
                        f"{rag_end - rag_start:.2f} seconds",
                        flush=True
                    )


                    # Measure answer rendering time
                    display_start = time.time()

                    print("GENERATED ANSWER:", answer[:200])

                    st.markdown(answer)

                    display_end = time.time()

                    print(
                        f"Answer display time: "
                        f"{display_end - display_start:.2f} seconds",
                        flush=True
                    )


                    # Display source citations
                    unique_sources = {}

                    for doc in source_docs:

                        filename = doc.metadata.get(
                            "source",
                            "Unknown document"
                        )

                        page = doc.metadata.get("page", 0)

                        if isinstance(page, int):
                            page += 1

                        key = (filename, page)

                        if key not in unique_sources:

                            unique_sources[key] = {
                                "filename": filename,
                                "page": page,
                                "content": doc.page_content
                            }

                    source_list = list(unique_sources.values())


                    # Measure source display time
                    source_start = time.time()

                    if source_list:

                        with st.expander("View Retrieved Sources"):

                            for i, source in enumerate(source_list):

                                st.markdown(
                                    f"**Source {i + 1}: "
                                    f"{source['filename']}**"
                                )

                                st.caption(
                                    f"Page {source['page']}"
                                )

                                st.write(source["content"])

                                st.divider()

                    source_end = time.time()

                    print(
                        f"Source display time: "
                        f"{source_end - source_start:.2f} seconds",
                        flush=True
                    )


                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [
                            f"📄 {source['filename']} — "
                            f"Page {source['page']}"
                            for source in source_list
                        ]
                    })


                except Exception as e:

                    st.error(
                        f"Could not generate answer: {str(e)}"
                    )

                    print("RAG ERROR:", str(e))
