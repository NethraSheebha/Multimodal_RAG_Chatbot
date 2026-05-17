# Multi-Modal RAG Chatbot

An enterprise-grade, high-performance Retrieval-Augmented Generation (RAG) chatbot built to parse, index, and query uploaded documents in real-time. By leveraging a decoupled hybrid cloud API architecture, this system completely bypasses local hardware constraints (CPU/GPU/RAM limitations) and free-tier rate limits to deliver lightning-fast, highly accurate, context-aware responses.

Live Application Link: **https://multimodalragchatbot-2agirw4zegqn6zhsdws6xx.streamlit.app/**

---

## 🚀 System Architecture Overview

This application splits the heavy data engineering pipelines away from the local hosting server, utilizing optimized cloud endpoints for heavy mathematical computations while keeping the user interface fast and lightweight.

       ┌─────────────────────────────────────────────────────────┐
       │                 STREAMLIT FRONTEND UI                   │
       │  • Document Uploader (PDF/DOCX)  • Chat Interface     │
       └────────────────────────────┬────────────────────────────┘
                                    │
                        [1] File Upload & Chunking
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                LANGCHAIN ORCHESTRATION                  │
       │  • Recursive Text Splitting   • Guardrail Filtering     │
       └────────────────────────────┬────────────────────────────┘
                                    │
                   [2] Text Arrays     [5] Vector Query
                                    │               │
                                    ▼               ▼
    ┌──────────────────────────────────────┐ ┌──────────────────────────────────────┐
    │             COHERE API               │ │          CHROMADB ENGINE             │
    │  • Engine: embed-english-v3.0        │ │  • Isolated Local Storage Cache      │
    │  • Task: High-Speed Embedding Vectors│ │  • Semantic Vector Similarity Search │
    └──────────────────┬───────────────────┘ └──────────────────┬───────────────────┘
                       │                                        │
               [3] 1024-Dim Vectors                      [6] Top-K Context Matches
                       │                                        │
                       └─────────────────► ┌────────────────────┘
                                           │
                                    [4] Index / Save
                                           │
                                           ▼
           ┌─────────────────────────────────────────────────────────┐
           │                 GROQ INFERENCE CLUSTER                  │
           │  • Model: Llama-3.1-8b-instant                          │
           │  • Input: [Context Blocks] + [User System Prompt]       │
           └────────────────────────────┬────────────────────────────┘
                                        │
                             [7] Ultra-Fast Completion
                                        │
                                        ▼
           ┌─────────────────────────────────────────────────────────┐
           │                 STREAMLIT CHAT DISPLAY                  │
           │  • Stateful Session UI Render                           │
           └─────────────────────────────────────────────────────────┘

By separating the **Embedding Generation** from the **Text Inference Generation**, the application guarantees optimal resource utilization and prevents standard API rate-limit bottlenecks.

---

## 🎨 Frontend Architecture (Streamlit)

The user interface is engineered using **Streamlit**, designed for minimal friction and maximum scannability during live evaluations.

*   **Stateful Chat Interface:** Implements `st.session_state` to handle real-time message tracking, allowing users to have multi-turn conversations without losing context.
*   **Dynamic File Upload Pipeline:** Supports ad-hoc uploads of `PDF` and `DOCX` files. The UI changes dynamically based on processing states using loading spinners (`st.spinner`).
*   **Security & Secrets Management:** Uses Streamlit’s native production secrets management layer to securely inject API keys at runtime without exposing them in the codebase.

---

## ⚙️ Backend Pipeline (LangChain & ChromaDB)

The core data orchestration logic is driven by **LangChain**, managing document intake, chunking, database parsing, and context retrieval.

*   **Parsing:** Extracts raw text strings from binary document structures cleanly via `pypdfium2` and `docx2txt`.
*   **Chunking (Recursive Text Splitting):** Chunks text using overlapping intervals (`RecursiveCharacterTextSplitter`). It features a strict defensive wrapper that filters out empty padding lines or broken artifact strings to eliminate indexing errors.
*   **Vector Engine (ChromaDB):** An in-memory/disk-persisted vector database that maps the document embeddings, executing high-speed Cosine/Euclidean semantic searches to retrieve relevant text segments based on user queries.

---

## 🤖 Models & Cloud Infrastructure Used

To ensure zero-downtime, high request-per-minute (RPM) capacities, and complete immunity to PyTorch/Torchvision environment dependency clashes, the app uses a **Dual Cloud Service Layout**:

### 1. Vector Embeddings: `cohere.Embed-English-v3.0`
*   **Provider:** Cohere (Trial Developer Tier)
*   **Purpose:** Converts text chunks into mathematical vectors.
*   **Advantage:** Cohere's developer tier permits up to **2,000 text inputs per minute**, allowing entire documents to be vectorized instantly in one batch without hitting typical token/minute walls found in other APIs.

### 2. Large Language Model (Inference): `llama-3.1-8b-instant`
*   **Provider:** Groq Cloud 
*   **Purpose:** Processes the combined retrieved context and user prompt to generate the final natural language answer.
*   **Advantage:** Groq's custom LPU (Language Processing Unit) architecture processes tokens at ultra-high speeds, offering a generous limit of **30 Requests Per Minute (RPM)** and **14,400 Requests Per Day (RPD)**—perfect for live demonstration evaluations.

---

## ✨ Core Features

*   **Zero-Hardware Footprint:** Runs seamlessly on servers with less than 1GB of RAM since no neural networks are loaded locally.
*   **Robust Hallucination Guardrails:** The LLM is explicitly instructed via system prompts to answer *only* using retrieved document context. If a user asks a question outside the document's scope, the model politely declines to answer.
*   **Multi-Format Document Support:** Easily swap between native text PDFs, multi-page reports, or corporate Word documents.
*   **Auto-Cleaning Cache:** Regenerates clean vector schemas upon container reboots to prevent historical data cross-contamination.

---

## 🛠️ Quickstart: How to Run Locally

Follow these instructions to spin up the project locally on your machine for development or verification.

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd rag-project
```
### 2. Set Up a Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure Local Secrets
```bash
# .streamlit/secrets.toml
COHERE_API_KEY = "your_cohere_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"
```
### 5. Launch the Application
```bash
streamlit run app.py
```

## 📄 License

[MIT](LICENSE)
