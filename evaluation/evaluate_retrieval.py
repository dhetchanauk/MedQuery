
import json
import os
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(PROJECT_ROOT)

from src.vector_store import load_vector_store
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# ---------------------------------------
# 1. Load evaluation dataset
# ---------------------------------------

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "eval_dataset.json"
)

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)


# ---------------------------------------
# 2. Load FAISS vector database
# ---------------------------------------

print("\nLoading FAISS vector database...")

vectorstore = load_vector_store()

documents = list(vectorstore.docstore._dict.values())

if not documents:
    raise ValueError("No documents found in FAISS database.")

print("Total indexed chunks:", len(documents))


# ---------------------------------------
# 3. Create FAISS retriever
# ---------------------------------------

faiss_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# ---------------------------------------
# 4. Create BM25 retriever
# ---------------------------------------

bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=10
)


# ---------------------------------------
# 5. Create Hybrid retriever
# ---------------------------------------

# Match the production RAG configuration:
# FAISS k=10, BM25 k=10, weights 0.5/0.5

production_faiss_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        production_faiss_retriever,
        bm25_retriever
    ],
    weights=[0.5, 0.5]
)


# ---------------------------------------
# 6. Remove duplicate chunks and keep top 5
# ---------------------------------------

def retrieve_top_five(retriever, question):

    retrieved_docs = retriever.invoke(question)

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

    return unique_docs[:5]


# ---------------------------------------
# 7. Evaluate retrieval methods
# ---------------------------------------

results = []

for item in evaluation_data:

    question = item["question"]
    expected_source = item["expected_source"]

    expected_concepts = [
        concept.lower()
        for concept in item["expected_concepts"]
    ]

    print("\n====================================")
    print("Question:", question)
    print("Expected source:", expected_source)
    print("====================================")

    # FAISS top 5
    faiss_docs = faiss_retriever.invoke(question)[:5]

    # Actual production-style Hybrid top 5
    hybrid_docs = retrieve_top_five(
        hybrid_retriever,
        question
    )

    for method, retrieved_docs in [
        ("FAISS", faiss_docs),
        ("Hybrid", hybrid_docs)
    ]:

        retrieved_count = len(retrieved_docs)

        relevant_source_count = sum(
            1
            for doc in retrieved_docs
            if doc.metadata.get("source", "") == expected_source
        )

        # Hit Rate@5:
        # Did at least one chunk come from the expected paper?
        hit_rate = (
            1 if relevant_source_count > 0 else 0
        )

        # Concept coverage:
        # What proportion of expected concepts appear
        # in the retrieved context?
        retrieved_text = " ".join(
            doc.page_content.lower()
            for doc in retrieved_docs
        )

        matched_concepts = [
            concept
            for concept in expected_concepts
            if concept in retrieved_text
        ]

        concept_coverage = (
            len(matched_concepts) / len(expected_concepts)
            if expected_concepts else 0
        )

        # Source match ratio:
        # Fraction of retrieved chunks from expected paper.
        source_precision = (
            relevant_source_count / retrieved_count
            if retrieved_count else 0
        )

        results.append({
            "Question ID": item["id"],
            "Question": question,
            "Method": method,
            "Expected Source": expected_source,
            "Relevant Source Chunks": relevant_source_count,
            "Retrieved Chunks": retrieved_count,
            "Source Precision@5": round(source_precision, 3),
            "Hit Rate@5": hit_rate,
            "Expected Concept Coverage": round(
                concept_coverage, 3
            )
        })

        print(f"\n{method} RESULTS")

        print(
            "Relevant source chunks:",
            relevant_source_count
        )

        print(
            "Retrieved chunks:",
            retrieved_count
        )

        print(
            "Hit Rate@5:",
            hit_rate
        )

        print(
            "Expected concept coverage:",
            f"{len(matched_concepts)}/{len(expected_concepts)}"
        )

        for i, doc in enumerate(retrieved_docs):

            print(f"\nChunk {i + 1}")

            print(
                "Source:",
                doc.metadata.get("source", "Unknown")
            )

            page = doc.metadata.get("page", "Unknown")

            if isinstance(page, int):
                page += 1

            print("Page:", page)

            print("Content:", doc.page_content[:200])


# ---------------------------------------
# 8. Save evaluation results
# ---------------------------------------

df = pd.DataFrame(results)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "retrieval_results.csv"
)

df.to_csv(OUTPUT_PATH, index=False)


# ---------------------------------------
# 9. Evaluation summary
# ---------------------------------------

summary = df.groupby("Method")[
    [
        "Source Precision@5",
        "Hit Rate@5",
        "Expected Concept Coverage"
    ]
].mean()

print("\n========== EVALUATION SUMMARY ==========")

print(summary.round(3))

print("\nResults saved to:", OUTPUT_PATH)
