# 🧠 Enterprise AI Knowledge Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://enterprise-rag-chatbot-4waqg6jq6pnbgna2qszccc.streamlit.app)
[![Backend Status](https://img.shields.io/badge/Render-API%20Online-46E3B7.svg?style=flat&logo=render&logoColor=black)](https://enterprise-rag-chatbot-ka-backend.onrender.com/health/ready)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL%20(Neon)-16-4169E1.svg?style=flat&logo=postgresql&logoColor=white)](https://neon.tech)
[![Redis](https://img.shields.io/badge/Redis%20(Upstash)-7-DC382D.svg?style=flat&logo=redis&logoColor=white)](https://upstash.com)
[![Qdrant](https://img.shields.io/badge/Qdrant%20Cloud-Vector%20DB-DC2626.svg?style=flat&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.18%20(BM25)-005A9C.svg?style=flat&logo=opensearch&logoColor=white)](https://opensearch.org)
[![Celery](https://img.shields.io/badge/Celery-5.4+-37814A.svg?style=flat&logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C.svg?style=flat&logo=prometheus&logoColor=white)](https://prometheus.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-000000.svg?style=flat&logo=langchain&logoColor=white)](https://smith.langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 **Live Production Deployment**:
> - 🌐 **Frontend Application (Streamlit Cloud):** [https://enterprise-rag-chatbot-4waqg6jq6pnbgna2qszccc.streamlit.app](https://enterprise-rag-chatbot-4waqg6jq6pnbgna2qszccc.streamlit.app)
> - ⚡ **Backend API Gateway (Render):** [https://enterprise-rag-chatbot-ka-backend.onrender.com](https://enterprise-rag-chatbot-ka-backend.onrender.com)
>   - 📚 **Swagger / OpenAPI Documentation:** [https://enterprise-rag-chatbot-ka-backend.onrender.com/docs](https://enterprise-rag-chatbot-ka-backend.onrender.com/docs)
>   - 🩺 **Health Readiness Probe:** [https://enterprise-rag-chatbot-ka-backend.onrender.com/health/ready](https://enterprise-rag-chatbot-ka-backend.onrender.com/health/ready)
>   - 📊 **Prometheus Metrics:** [https://enterprise-rag-chatbot-ka-backend.onrender.com/metrics](https://enterprise-rag-chatbot-ka-backend.onrender.com/metrics)

An enterprise-grade, multi-tenant, agentic Retrieval-Augmented Generation (**RAG**) knowledge assistant platform. Upload multi-format documents (PDF, DOCX, Markdown, TXT), perform hybrid semantic & lexical keyword search, and receive streaming answers with inline citations, verified confidence scores, and reasoning summaries.

Built from the ground up with **Domain-Driven Design (DDD) & Hexagonal Architecture** (Clean Architecture), distributed async ingestion, robust multi-tenancy, and production hardening for **1000+ concurrent users**.

---

## 📑 Table of Contents

- [🌐 Live Production Deployments](#-live-production-deployments)

- [1. Architectural Highlights](#1-architectural-highlights)
- [2. System Architecture & Diagrams](#2-system-architecture--diagrams)
  - [2.1 High-Level Component Topology](#21-high-level-component-topology)
  - [2.2 Clean Architecture / DDD Layers](#22-clean-architecture--ddd-layers)
  - [2.3 Agentic RAG Retrieval & Generation Flow](#23-agentic-rag-retrieval--generation-flow)
  - [2.4 Document Ingestion & Hierarchical Chunking](#24-document-ingestion--hierarchical-chunking)
  - [2.5 Database & Store Entity Relationship](#25-database--store-entity-relationship)
- [3. Key Features](#3-key-features)
- [4. Technology Stack](#4-technology-stack)
- [5. Repository Structure](#5-repository-structure)
- [6. API Endpoints](#6-api-endpoints)
- [7. Getting Started & Deployments](#7-getting-started--deployments)
  - [7.1 Live Production Endpoints](#71-live-production-endpoints)
  - [7.2 Quickstart with Docker Compose](#72-quickstart-with-docker-compose)
  - [7.3 Local Development](#73-local-development)
  - [7.4 Environment Variables](#74-environment-variables)
- [8. Testing & Quality Assurance](#8-testing--quality-assurance)
- [9. Production Hardening & Cloud Infrastructure](#9-production-hardening--cloud-infrastructure)
- [10. Evaluation & Benchmarking](#10-evaluation--benchmarking)

---

## 1. Architectural Highlights

- **Domain-Driven Design / Hexagonal Ports & Adapters**: Zero framework dependencies in domain entities and repository interfaces. The application use cases depend solely on abstract interfaces, allowing instant swapping of databases, vector stores, search engines, or LLM providers without touching business logic.
- **Hybrid Retrieval (Dense + Sparse)**: Combines **Qdrant** (cosine vector embeddings) with **OpenSearch / Elasticsearch** (BM25 lexical keyword search) fused via scale-invariant **Reciprocal Rank Fusion (RRF)**.
- **Hierarchical Parent-Child Chunking**: Searches small child chunks (~200 tokens) for high-precision retrieval while passing expanded parent chunks (~800 tokens) to the LLM to preserve complete surrounding context.
- **Agentic Multi-Turn RAG**:
  - **Planner LLM**: Classifies if retrieval is necessary and performs contextual query rewriting.
  - **Evaluator LLM**: Evaluates retrieved document sufficiency, computes confidence, and triggers corrective iterative retrieval if context is incomplete.
  - **Generator LLM**: Streams answers via Server-Sent Events (SSE) with strict grounded-only rules and verified inline citation brackets `[n]`.
- **Full-Stack Observability**: End-to-end tracing via **LangSmith**, error tracking via **Sentry**, real-time metric counters & histograms via **Prometheus** (`/metrics`), and correlation ID request auditing.

---

## 2. System Architecture & Diagrams

### 2.1 High-Level Component Topology

```mermaid
graph TB
    User(["👤 User / Client Browser"])

    subgraph Presentation ["Presentation Layer (Streamlit Frontend :8501)"]
        UI["Streamlit Multipage App"]
        ApiClient["ApiClient (HTTP / SSE Client)"]
        UI --> ApiClient
    end

    subgraph Gateway ["API Gateway (FastAPI Backend :8000)"]
        FastAPI["FastAPI App (Gunicorn + Uvicorn Async Workers)"]
        RateLimiter["Distributed Redis Sliding-Window Rate Limiter"]
        Audit["Audit Logger & Request ID Middleware"]
        DI["Composition Root (Dependency Injection)"]
        FastAPI --> RateLimiter
        FastAPI --> Audit
        FastAPI --> DI
    end

    subgraph Workers ["Async Processing (Celery Distributed Workers)"]
        CeleryWorker["Celery Worker Replicas"]
        Extractor["Text Extractors (PDF, DOCX, MD, TXT)"]
        Chunker["Parent-Child Hierarchical Chunker"]
        CeleryWorker --> Extractor --> Chunker
    end

    subgraph DataStores ["Data & Storage Layer"]
        Postgres[("Neon Serverless Postgres / PostgreSQL 16<br/>Metadata, Users, Chats, Chunks")]
        Redis[("Upstash Redis / Redis 7<br/>Task Queue, Cache & Rate Limiting")]
        QdrantDB[("Qdrant Cloud / Qdrant<br/>Dense Vector Embeddings (1536-dim)")]
        OpenSearchDB[("OpenSearch / Elasticsearch 2.x<br/>BM25 Lexical Keyword Search")]
    end

    subgraph AIProviders ["External AI Services"]
        LLMs["LLM Providers (Claude, OpenAI GPT-4o, Gemini)"]
        Embedder["Embedding Providers (OpenAI text-embedding-3-small)"]
        LangSmithCloud["LangSmith Tracing"]
        SentryCloud["Sentry Error Tracking"]
    end

    User -->|HTTP / UI| UI
    ApiClient -->|REST & SSE Stream| FastAPI
    FastAPI -->|Enqueue Ingestion Task| Redis
    Redis -->|Consume Task| CeleryWorker
    DI -->|Queries & Transactions| Postgres
    DI -->|Dense Vector Search| QdrantDB
    DI -->|BM25 Search| OpenSearchDB
    CeleryWorker -->|Upsert Chunks & Vectors| QdrantDB
    CeleryWorker -->|Bulk Index Tokens| OpenSearchDB
    CeleryWorker -->|Save Metadata| Postgres
    FastAPI -->|Generate Completion| LLMs
    FastAPI -->|Embed Queries| Embedder
    FastAPI -.->|Telemetry| LangSmithCloud
    FastAPI -.->|Errors| SentryCloud
```

---

### 2.2 Clean Architecture / DDD Layers

```mermaid
graph TD
    subgraph Domain ["1. Domain Layer (Zero External Dependencies)"]
        Entities["Entities: User, Document, Chunk, ChatSession, Message, Citation"]
        Ports["Repository Ports: UserRepository, VectorStore, KeywordSearchIndex, LLMClient"]
    end

    subgraph Application ["2. Application Layer (Use Cases & Business Logic)"]
        IngestUC["IngestDocumentUseCase"]
        AskUC["AskQuestionUseCase (Agentic Loop)"]
        RetrieveUC["HybridRetriever (RRF + Compression)"]
        EvalUC["EvaluateRAGPipelineUseCase"]
    end

    subgraph Infrastructure ["3. Infrastructure Layer (Concrete Adapters)"]
        SqlRepo["Sql*Repositories (SQLAlchemy + Asyncpg)"]
        QdrantAdapter["QdrantVectorStore"]
        OSAdapter["OpenSearchKeywordIndex"]
        LLMAdapter["OpenAI / Claude / Gemini Clients"]
        CeleryAdapter["Celery Ingestion Tasks"]
        ObsAdapter["Sentry, Prometheus & LangSmith Adapters"]
    end

    subgraph Interfaces ["4. Interfaces Layer (Delivery & API)"]
        Routers["FastAPI Routers: /auth, /collections, /documents, /chat"]
        Schemas["Pydantic v2 Request/Response Schemas"]
        CompRoot["Composition Root (dependencies.py)"]
    end

    Interfaces --> Application
    Infrastructure --> Domain
    Application --> Domain
    CompRoot --> Infrastructure
```

---

### 2.3 Agentic RAG Retrieval & Generation Flow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit as Streamlit Frontend
    participant API as FastAPI /chat/ask
    participant Planner as Planner LLM
    participant Qdrant as Qdrant (Vector)
    participant OpenSearch as OpenSearch (BM25)
    participant Evaluator as Evaluator LLM
    participant Generator as Generator LLM (SSE)

    User->>Streamlit: Asks question
    Streamlit->>API: POST /chat/ask (session_id, question)
    API-->>Streamlit: SSE Event: status ("Planning retrieval...")
    
    API->>Planner: Analyze query + chat history
    Planner-->>API: {needs_retrieval: true, queries: [q1, q2]}

    loop Up to 3 Iterations (if insufficient)
        API-->>Streamlit: SSE Event: status ("Executing hybrid search...")
        par Parallel Dense & Sparse Search
            API->>Qdrant: Query dense vectors (q1, q2)
            Qdrant-->>API: Top-K Vector Hits
        and
            API->>OpenSearch: Query BM25 keywords (q1, q2)
            OpenSearch-->>API: Top-K Lexical Hits
        end
        API->>API: Reciprocal Rank Fusion (RRF) & Parent Chunk Expansion
        API->>API: Reranking & Context-Budget Trimming
        
        API->>Evaluator: Judge context sufficiency & confidence
        Evaluator-->>API: {sufficient: true, confidence: 0.94, reasoning: "..."}
    end

    API-->>Streamlit: SSE Event: status ("Generating answer...")
    API->>Generator: Stream answer strictly grounded in citations [n]
    loop Token Streaming
        Generator-->>API: token chunk
        API-->>Streamlit: SSE Event: token (chunk)
    end

    API-->>Streamlit: SSE Event: final (citations, confidence, reasoning, follow-ups)
    API->>API: Log metrics to Prometheus & Audit Table
    Streamlit-->>User: Render formatted answer with interactive citations
```

---

### 2.4 Document Ingestion & Hierarchical Chunking

```mermaid
graph LR
    subgraph UploadStage ["1. Upload & Validation"]
        DocFile["Document File (PDF / DOCX / MD / TXT)"] --> HashCheck["SHA-256 Content Hash Check"]
        HashCheck -->|Duplicate| ConflictErr["HTTP 409 Conflict"]
        HashCheck -->|New| Enqueue["Enqueue Celery Ingestion Task"]
    end

    subgraph ExtractionStage ["2. Extraction & Chunking"]
        Enqueue --> Extract["Text & Metadata Extractor"]
        Extract --> HierarchicalChunker["Parent-Child Chunker"]
        HierarchicalChunker --> Parents["Parent Chunks (~800 tokens)<br/>(Context Containers)"]
        HierarchicalChunker --> Children["Child Chunks (~200 tokens)<br/>(Search Leaves)"]
    end

    subgraph IndexingStage ["3. Embed & Dual-Store Indexing"]
        Children --> Embed["OpenAI Embedding (1536-dim)"]
        Embed --> UpsertQdrant[("Qdrant Vector Store<br/>Child Vectors")]
        Children --> IndexOS[("OpenSearch Cluster<br/>Child BM25 Tokens")]
        Parents --> SavePostgres[("Postgres DB<br/>Metadata & Parents")]
        Children --> SavePostgres
    end
```

---

### 2.5 Database & Store Entity Relationship

```
+-----------------------------------------------------------------------------------+
|                                  PostgreSQL                                       |
+-----------------------------------------------------------------------------------+
| +-----------------+       1:N       +-------------------+                         |
| |      users      |---------------->|    collections    |                         |
| +-----------------+                 +-------------------+                         |
|         |                                     |                                   |
|         | 1:N                                 | 1:N                               |
|         v                                     v                                   |
| +-----------------+       1:N       +-------------------+                         |
| |  chat_sessions  |                 |     documents     |                         |
| +-----------------+                 +-------------------+                         |
|         |                                     |                                   |
|         | 1:N                                 | 1:N                               |
|         v                                     v                                   |
| +-----------------+                 +-------------------+                         |
| |    messages     |                 |      chunks       |                         |
| +-----------------+                 | (parent + child)  |                         |
|                                     +-------------------+                         |
|                                               |                                   |
+-----------------------------------------------|-----------------------------------+
                                                |
                         +----------------------+----------------------+
                         |                                             |
                         v                                             v
           +---------------------------+                 +---------------------------+
           |   Qdrant (Vectors)        |                 | OpenSearch (BM25 Lexical) |
           |   - id: chunk_id          |                 | - id: chunk_id            |
           |   - vector: 1536-dim float|                 | - text: analyzed string   |
           |   - user_id (keyword)     |                 | - user_id (keyword)       |
           |   - collection_id (kw)    |                 | - collection_id (keyword) |
           |   - level: 'child'        |                 | - level: 'child'          |
           +---------------------------+                 +---------------------------+
```

---

## 3. Key Features

- 🔒 **Multi-Tenant Security**: Strict user-level isolation enforced at database query level and vector/lexical index filters. No user can view or query another user's documents.
- ⚡ **Scale-Invariant Fusion**: Hybrid search fused via Reciprocal Rank Fusion ($RRF(d) = \sum \frac{1}{k + r(d)}$) to balance dense and lexical scores without score distortion.
- 💬 **Live SSE Streaming**: Word-by-word streaming generation with interactive citation references and confidence gauge.
- 📊 **Context-Budget Trimming**: Automatic token-budget-aware context compression preventing token overflows and reducing LLM inference costs.
- 🛡️ **Distributed Rate Limiting**: Redis sliding-window algorithm enforcing per-IP rate limits across multiple API replicas.
- 📈 **Full Observability**: Prometheus instrumentation (`/metrics`), LangSmith execution tracing, and Sentry exception capture.
- 📑 **Format Versatility**: Built-in support for `.pdf`, `.docx`, `.md`, and `.txt` files.

---

## 4. Technology Stack

| Layer / Role | Technology | Description |
| :--- | :--- | :--- |
| **API Backend** | FastAPI 0.115+, Gunicorn, Uvicorn | High-performance asynchronous REST & SSE backend |
| **Frontend UI** | Streamlit 1.38+ | Modern multi-page application talking to backend via HTTP |
| **Relational DB** | PostgreSQL 16 / Neon Serverless Postgres | SQL repository with Asyncpg & SQLAlchemy 2.0 |
| **Task Queue & Cache** | Celery 5.4+, Upstash Redis / Redis 7 | Distributed async background document processing |
| **Vector Database** | Qdrant Cloud / Qdrant 1.11+ | Dense vector indexing and filtered HNSW cosine search |
| **Lexical Search** | OpenSearch 2.18+ / Elasticsearch | Distributed multi-tenant BM25 keyword matching |
| **AI / LLM** | Anthropic Claude, OpenAI GPT-4o, Google Gemini | Pluggable LLM clients supporting generation & evaluation |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dimensional semantic dense representations |
| **Observability** | Prometheus, LangSmith, Sentry | Real-time metrics, trace visualizer, and APM error tracking |
| **Monorepo / Orchestration**| pnpm, Turborepo, Docker Compose | Unified monorepo scripts and multi-container deployment |

---

## 5. Repository Structure

```
Enterprise-RAG-main/
├── docker/
│   ├── backend.Dockerfile              # Production Gunicorn+Uvicorn FastAPI image
│   ├── frontend.Dockerfile             # Streamlit UI production image
│   └── worker.Dockerfile               # Celery async worker image
├── docker-compose.yml                  # Full multi-container environment (Postgres, Redis, Qdrant, OpenSearch, API, Worker, UI)
├── package.json                        # Turborepo task runner configuration
├── turbo.json                          # Monorepo build pipeline
├── .env.example                        # Environment configuration template
│
├── apps/
│   ├── backend/
│   │   ├── pyproject.toml              # Backend dependencies & pytest config
│   │   ├── src/knowledge_assistant/
│   │   │   ├── config.py               # Pydantic BaseSettings environment config
│   │   │   ├── domain/                 # Layer 1: Entities & Repository interfaces (Ports)
│   │   │   │   ├── entities.py         # User, Document, Chunk, ChatSession, Message, Citation
│   │   │   │   └── repositories.py     # UserRepository, VectorStore, KeywordSearchIndex, etc.
│   │   │   ├── application/            # Layer 2: Core Use Cases & Pipeline Logic
│   │   │   │   ├── use_cases/
│   │   │   │   │   ├── ask_question.py # Agentic Planner -> Retriever -> Evaluator loop
│   │   │   │   │   ├── ingest_document.py # Ingestion & chunking orchestrator
│   │   │   │   │   └── retrieve_context.py# Hybrid retriever, RRF, reranker, budget compression
│   │   │   │   ├── chunking/           # Parent-child chunking algorithms
│   │   │   │   ├── extractors/         # PDF, DOCX, Markdown, TXT text extractors
│   │   │   │   └── evaluation/         # Retrieval & generation benchmark metrics
│   │   │   ├── infrastructure/         # Layer 3: Concrete Adapters
│   │   │   │   ├── db/                 # Postgres models, session, repositories (Asyncpg)
│   │   │   │   ├── vector_store/       # QdrantVectorStore adapter
│   │   │   │   ├── search/             # OpenSearchKeywordIndex & InMemoryBM25Index adapters
│   │   │   │   ├── tasks/              # Celery app & async ingestion tasks
│   │   │   │   ├── llm/                # OpenAI, Claude, Gemini client implementations
│   │   │   │   ├── embeddings/         # OpenAI embedding client adapter
│   │   │   │   └── observability/      # Prometheus metrics, LangSmith tracing, Sentry APM
│   │   │   └── interfaces/api/         # Layer 4: FastAPI routers & composition root
│   │   │       ├── main.py             # FastAPI entrypoint, middleware, health endpoints
│   │   │       ├── dependencies.py     # Composition Root (Dependency Injection wiring)
│   │   │       ├── schemas.py          # Pydantic v2 schemas
│   │   │       └── routers/            # auth, collections, documents, chat
│   │   └── tests/                      # 39 Backend unit, integration & search tests
│   │
│   └── frontend/
│       ├── pyproject.toml              # Frontend dependencies & pytest config
│       ├── app.py                      # Streamlit entrypoint & authentication views
│       ├── api_client.py               # HTTP client translating Streamlit calls to FastAPI REST/SSE
│       ├── pages/                      # Multi-page Streamlit views
│       │   ├── 1_Collections.py        # Collection management
│       │   ├── 2_Upload_Documents.py   # Drag-and-drop document upload
│       │   ├── 3_My_Documents.py       # Document browser & deletion
│       │   ├── 4_AI_Chat.py            # Real-time streaming chat with citations
│       │   ├── 5_Conversation_History.py # Past chat sessions & message logs
│       │   └── 6_Profile_Settings.py   # User settings & session management
│       └── tests/                      # 20 Frontend unit & client tests
```

---

## 6. API Endpoints

| Category | Method | Path | Description | Auth Required |
| :--- | :--- | :--- | :--- | :---: |
| **Auth** | `POST` | `/auth/register` | Register new user account | ❌ |
| **Auth** | `POST` | `/auth/login` | Authenticate & retrieve JWT access token | ❌ |
| **Auth** | `GET` | `/auth/me` | Retrieve profile of authenticated user | ✅ |
| **Collections** | `GET` | `/collections` | List all collections for current user | ✅ |
| **Collections** | `POST` | `/collections` | Create a new document collection | ✅ |
| **Collections** | `DELETE`| `/collections/{id}` | Delete collection and cascade delete documents | ✅ |
| **Documents** | `GET` | `/documents` | List documents (optional `?collection_id=...`) | ✅ |
| **Documents** | `POST` | `/documents/upload` | Upload & enqueue file for background ingestion | ✅ |
| **Documents** | `DELETE`| `/documents/{id}` | Delete document and remove from Qdrant/OpenSearch | ✅ |
| **Chat** | `POST` | `/chat/sessions` | Create a new conversation session | ✅ |
| **Chat** | `GET` | `/chat/sessions` | List user conversation sessions | ✅ |
| **Chat** | `GET` | `/chat/sessions/{id}/messages` | Retrieve full message history with citations | ✅ |
| **Chat** | `POST` | `/chat/ask` | **Server-Sent Events (SSE)** streaming agentic RAG answer | ✅ |
| **Monitoring** | `GET` | `/health/live` | Liveness probe | ❌ |
| **Monitoring** | `GET` | `/health/ready`| Deep readiness probe (Postgres, Redis, Qdrant, OpenSearch)| ❌ |
| **Monitoring** | `GET` | `/metrics` | Prometheus metrics scrape endpoint | ❌ |

---

## 7. Getting Started & Deployments

### 7.1 Live Production Endpoints

The full system is deployed in production and immediately accessible:

| Service | Platform | Endpoint URL | Status / Notes |
| :--- | :--- | :--- | :--- |
| **Frontend Web App** | Streamlit Community Cloud | [https://enterprise-rag-chatbot-4waqg6jq6pnbgna2qszccc.streamlit.app](https://enterprise-rag-chatbot-4waqg6jq6pnbgna2qszccc.streamlit.app) | Live Interactive UI |
| **Backend API Gateway** | Render Cloud | [https://enterprise-rag-chatbot-ka-backend.onrender.com](https://enterprise-rag-chatbot-ka-backend.onrender.com) | FastAPI REST & SSE Gateway |
| **Interactive API Docs** | Render Cloud | [https://enterprise-rag-chatbot-ka-backend.onrender.com/docs](https://enterprise-rag-chatbot-ka-backend.onrender.com/docs) | Swagger UI / OpenAPI Schema |
| **Health Readiness Probe**| Render Cloud | [https://enterprise-rag-chatbot-ka-backend.onrender.com/health/ready](https://enterprise-rag-chatbot-ka-backend.onrender.com/health/ready) | Postgres, Redis, Vector, BM25 checks |
| **Prometheus Metrics** | Render Cloud | [https://enterprise-rag-chatbot-ka-backend.onrender.com/metrics](https://enterprise-rag-chatbot-ka-backend.onrender.com/metrics) | Scrape endpoint |

---

### 7.2 Quickstart with Docker Compose

The fastest way to launch the entire stack locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Enterprise-RAG.git
   cd Enterprise-RAG
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` and set your `OPENAI_API_KEY` (or Anthropic/Gemini keys).*

3. **Start all services**:
   ```bash
   docker compose up --build
   ```

4. **Access the application**:
   - **Frontend UI**: [http://localhost:8501](http://localhost:8501)
   - **API Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)
   - **OpenSearch Cluster**: [http://localhost:9200](http://localhost:9200)

---

### 7.3 Local Development

#### Prerequisites
- Python 3.11+
- Node.js & pnpm (optional, for monorepo tasks)

#### Step 1: Backend Setup
```bash
cd apps/backend
python -m venv .venv
# On Windows: .\.venv\Scripts\Activate.ps1
# On Linux/macOS: source .venv/bin/activate

pip install -e ".[dev]"
uvicorn knowledge_assistant.interfaces.api.main:app --reload --app-dir src --port 8000
```

#### Step 2: Celery Worker
```bash
# In a separate terminal with backend virtual environment active
celery -A knowledge_assistant.infrastructure.tasks.celery_app worker --loglevel=info
```

#### Step 3: Frontend Setup
```bash
cd apps/frontend
# Activate venv or install frontend dependencies
pip install -e ".[dev]"
streamlit run app.py --server.port 8501
```

---

### 7.4 Environment Variables

Key configuration variables in `.env`:

```ini
# Environment
ENVIRONMENT=production
JWT_SECRET=your-64-character-hex-secret
CORS_ORIGINS=*

# Database (Neon Serverless Postgres or Local Docker)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_assistant

# Redis (Upstash Redis or Local Docker)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Vector Store (Qdrant Cloud or Local Docker)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# OpenSearch (Elasticsearch / Local Docker)
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_INDEX=knowledge_chunks

# LLM & Embeddings
LLM_PROVIDER=openai              # "openai" | "claude" | "gemini"
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-...
EMBEDDING_DIM=1536

# Observability
PROMETHEUS_ENABLED=true
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
SENTRY_DSN=
```

---

## 8. Testing & Quality Assurance

The codebase includes an extensive suite of **59 unit and integration tests** across both backend and frontend.

### Running Backend Tests (39 Tests)
```bash
cd apps/backend
pytest
```
*Covers:* Parent-child chunking, citation scoring, evaluation metrics, document extractors (PDF, DOCX, MD, TXT), async document ingestion, OpenSearch BM25 indexing, lexical rerankers, Reciprocal Rank Fusion, and SSE serialization.

### Running Frontend Tests (20 Tests)
```bash
cd apps/frontend
pytest
```
*Covers:* API client error parsing, JWT header injection, collections/documents CRUD, SSE event stream parsing, dashboard metrics computation, and duplicate upload prevention.

---

## 9. Production Hardening & Cloud Infrastructure

The platform is engineered to handle **1000+ concurrent users** smoothly:

1. **Neon Serverless Postgres & PgBouncer**:
   - `prepared_statement_cache_size=0` configured on `asyncpg` to eliminate prepared statement collision errors across PgBouncer connection poolers.
   - Connection pool recycling (`pool_recycle=300`, `pool_timeout=30`, `max_overflow=20`, `pool_pre_ping=True`) to prune stale connections.
2. **Upstash Redis Resilience**:
   - Socket keepalives, socket timeouts (`30.0s`), and `retry_on_timeout=True` configured to survive idle connection terminations.
   - `worker_prefetch_multiplier=1` and `task_acks_late=True` ensuring fair task scheduling and no lost ingestion jobs across workers.
3. **Qdrant Cloud & OpenSearch Cloud**:
   - 30-second network timeouts and auto-indexing of payload keywords (`user_id`, `collection_id`, `document_id`, `level`) preventing unindexed linear scans across large corpora.
4. **Prometheus Metric Optimization**:
   - Low-cardinality label design preventing Prometheus OOM/memory leaks under thousands of unique users.
5. **Gunicorn / Uvicorn Concurrency**:
   - Production container configured with `--timeout 120`, `--keep-alive 5`, and `--graceful-timeout 30` to guarantee uninterrupted SSE chat streams.

---

## 10. Evaluation & Benchmarking

The platform includes an automated RAG evaluation framework measuring response accuracy against predefined benchmark questions:

```bash
cd apps/backend
python scripts/run_evaluation.py --email user@example.com --password secret
```

### Measured Metrics
- **Context Precision**: Ratio of relevant retrieved chunks to total retrieved chunks.
- **Context Recall**: Measure of whether all necessary facts were retrieved.
- **Groundedness / Faithfulness**: Verification that claims in the answer are strictly supported by retrieved citations.
- **Answer Relevance**: Semantic alignment between the user's question and generated answer.
- **Hallucination Rate**: Percentage of ungrounded statements.
- **Pipeline Latency**: Granular latency breakdown across Planner, Retrieval, Evaluator, and Generation phases.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
