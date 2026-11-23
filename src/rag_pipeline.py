
import time

from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from src.prompts import MEDICAL_PROMPT


def create_rag_chain(vectorstore):

    # Initialize Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_retries=0
    )

    # Get indexed documents
    documents = list(vectorstore.docstore._dict.values())

    if not documents:
        raise ValueError("No documents found in the FAISS vector store.")

    print(
        f"Total documents loaded for retrieval: {len(documents)}",
        flush=True
    )

    # 1. FAISS semantic retrieval
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    # 2. BM25 keyword retrieval
    bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=10
    )

    # 3. Hybrid retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            faiss_retriever,
            bm25_retriever
        ],
        weights=[0.5, 0.5]
    )

    # 4. Prompt for understanding follow-up questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Given the conversation history and the latest user question,
rewrite the latest question as a standalone question.

Instructions:
- Resolve references such as "it", "they", "this model" and "that paper".
- Use the conversation history only to understand the question.
- Do not answer the question.
- Preserve the original meaning.
- If the question is already standalone, return it unchanged.
- Return only the rewritten question."""
        ),
        (
            "human",
            """Conversation history:
{chat_history}

Latest question:
{input}"""
        )
    ])

    # 5. History-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        ensemble_retriever,
        contextualize_q_prompt
    )

    # 6. Retrieve, remove duplicates and keep top 5
    def retrieve_top_five(inputs):

        retrieved_docs = history_aware_retriever.invoke(inputs)

        unique_docs = []
        seen_chunks = set()

        for doc in retrieved_docs:

            chunk_key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content.strip()
            )

            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique_docs.append(doc)

        selected_docs = unique_docs[:5]

        print(
            f"Hybrid retrieval returned {len(retrieved_docs)} chunks.",
            flush=True
        )

        print(
            f"Unique chunks passed to Groq: {len(selected_docs)}",
            flush=True
        )

        return selected_docs

    top5_retriever = RunnableLambda(retrieve_top_five)

    # 7. Document answering chain
    document_chain = create_stuff_documents_chain(
        llm,
        MEDICAL_PROMPT
    )

    # 8. Final conversational RAG chain
    rag_chain = create_retrieval_chain(
        top5_retriever,
        document_chain
    )

    print("RAG chain initialized successfully.", flush=True)

    return rag_chain


def ask_question(rag_chain, question, chat_history=""):

    total_start = time.time()

    print("\n========== RAG EXECUTION ==========", flush=True)

    print("STEP 1: Starting RAG pipeline", flush=True)

    print("STEP 2: Calling rag_chain.invoke()", flush=True)

    try:

        response = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })

        print("STEP 3: RAG response received", flush=True)

    except Exception as e:

        print(
            f"RAG GENERATION ERROR: {type(e).__name__}: {str(e)}",
            flush=True
        )

        raise

    # Extract answer and retrieved documents
    answer = response["answer"]
    sources = response["context"]

    print("STEP 4: Extracting answer and sources", flush=True)

    print(
        f"Final source chunks: {len(sources)}",
        flush=True
    )

    # Display retrieved source information
    print(
        "\n========== CONVERSATIONAL HYBRID SEARCH ==========",
        flush=True
    )

    for i, doc in enumerate(sources):

        print(f"\nCHUNK {i + 1}", flush=True)

        print(
            "Source:",
            doc.metadata.get("source", "Unknown"),
            flush=True
        )

        page = doc.metadata.get("page", "Unknown")

        if isinstance(page, int):
            page += 1

        print("Page:", page, flush=True)

        print("Content:", flush=True)

        print(doc.page_content[:500], flush=True)

        print("----------------------------------------", flush=True)

    # Execution time
    total_time = time.time() - total_start

    print(
        f"\nTOTAL RAG EXECUTION TIME: {total_time:.2f} seconds",
        flush=True
    )

    print(
        "========== RAG EXECUTION COMPLETED ==========\n",
        flush=True
    )

    return answer, sources
