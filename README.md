# MedQuery – Healthcare Research & Document Assistant

MedQuery is an AI-powered healthcare research assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded medical and research PDFs.

Built using LangChain, FAISS, HuggingFace embeddings, BM25 hybrid retrieval, and Groq LLMs, MedQuery retrieves relevant document passages and generates context-grounded answers with source references.

## Features

- **PDF Processing:** Upload and process healthcare research papers and medical documents.
- **Text Chunking:** Split documents into manageable chunks with overlapping context.
- **Semantic Search:** Retrieve relevant passages using FAISS vector similarity search.
- **Keyword Search:** Use BM25 to retrieve documents based on keyword relevance.
- **Hybrid Retrieval:** Combine FAISS and BM25 using an ensemble retriever.
- **Conversational RAG:** Maintain conversation context and understand follow-up questions.
- **Context-Grounded Answers:** Generate responses using retrieved document content.
- **Source References:** Display the source documents and page information used to generate answers.
- **Retrieval Evaluation:** Evaluate retrieval performance using source matching and expected concept coverage.

## Architecture

```text
             User
              |
              v
       Streamlit Interface
              |
              v
        PDF Processing
              |
              v
       Text Preprocessing
              |
              v
       Text Chunking
              |
              v
      HuggingFace Embeddings
              |
              v
         FAISS Vector Store
              |
              v
        Hybrid Retrieval
         /           \
        v             v
   FAISS Search    BM25 Search
        \             /
         v           v
       Ensemble Retriever
              |
              v
      Conversational Retrieval
              |
              v
         Top 5 Chunks
              |
              v
          Groq LLM
              |
              v
     Context-Grounded Answer
              |
              v
       Answer + Sources
```

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Embedding Model | all-MiniLM-L6-v2 |
| Keyword Retrieval | BM25 |
| Hybrid Retrieval | Ensemble Retriever |
| LLM Provider | Groq |
| PDF Processing | PyPDF |
| Data Evaluation | Pandas |

## Project Structure

```text
MedQuery/
│
├── data/
│
├── evaluation/
│   ├── eval_dataset.json
│   ├── evaluate_retrieval.py
│   └── retrieval_results.csv
│
├── src/
│   ├── pdf_processor.py
│   ├── prompts.py
│   ├── rag_pipeline.py
│   ├── text_preprocessing.py
│   └── vector_store.py
│
├── vector_db/
│   └── medical_index/
│       ├── index.faiss
│       └── index.pkl
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Retrieval Evaluation

The retrieval pipeline was evaluated using a dataset of 20 questions covering two research papers.

The evaluation compares FAISS-only retrieval against hybrid FAISS + BM25 retrieval.

| Metric | FAISS | Hybrid |
|---|---:|---:|
| Source Precision@5 | 1.00 | 1.00 |
| Hit Rate@5 | 1.00 | 1.00 |
| Expected Concept Coverage | 0.888 | 0.888 |

### Metric Definitions

- **Source Precision@5:** Proportion of the five retrieved chunks that belong to the expected source document.
- **Hit Rate@5:** Whether at least one chunk from the expected source appears in the top five results.
- **Expected Concept Coverage:** Proportion of expected concepts found in the retrieved text using keyword matching.

Both retrieval methods achieved identical results on this evaluation dataset. These metrics are preliminary retrieval indicators and do not independently measure answer correctness or clinical accuracy.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dhetchanauk/MedQuery.git
cd MedQuery
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own Groq API key.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Retrieval Evaluation

To run the retrieval evaluation independently:

```bash
python evaluation/evaluate_retrieval.py
```

The evaluation results are saved to:

```text
evaluation/retrieval_results.csv
```

## Important Notes

- A Groq API key is required to generate answers.
- API usage is subject to the provider's rate limits.
- The included FAISS index was generated from the research documents used during development.
- If using a different vector index, ensure it comes from a trusted source before loading it.
- MedQuery is a research-document assistant, not a substitute for professional medical advice, diagnosis, or treatment.

## Future Improvements

- Reranking retrieved passages to improve relevance.
- More comprehensive retrieval evaluation using additional metrics.
- Support for multiple document collections.
- Improved source attribution and citation accuracy.
- Expanded evaluation of generated answers.

## Author

**Dhetchana Uma Gowri Krishna**


GitHub: [@dhetchanauk](https://github.com/dhetchanauk)
