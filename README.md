# Shadow CFO - Agentic RAG

An end-to-end "Shadow CFO" agent allowing users to upload 10-K PDFs and ask complex financial questions. Powered by **LangGraph**, **OpenSearch**, and **FastAPI**.

## Features
- **Agentic Workflow**: Cyclic graph with Retrieval, Analysis, Calculation, and Verification nodes.
- **Hybrid Search**: Combines BM25 and Vector Search using OpenSearch.
- **Thinking Process**: Real-time visibility into the agent's reasoning steps.
- **Faithfulness Check**: Self-correction mechanism to ensure answering is grounded in context.

## Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+
- OpenAI API Key (or Anthropic Key if configured)

## Setup Instructions

### 1. Infrastructure
Start OpenSearch and PostgreSQL:
```bash
cd infra
docker-compose up -d
```

### 2. Backend
Setup the Python environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set Environment Variables:
```bash
export OPENAI_API_KEY="sk-..."
export OPENSEARCH_URL="http://localhost:9200"
```

Start the Server:
```bash
uvicorn main:app --reload
```
API will be running at `http://localhost:8000`.

### 3. Frontend
Install dependencies and run Next.js:
```bash
cd frontend
npm install
npm run dev
```
UI will be running at `http://localhost:3000`.

## Usage
1. Open the Frontend (`http://localhost:3000`).
2. Upload a 10-K PDF using the sidebar uploader.
3. Wait for the success message (indexing complete).
4. Ask a question (e.g., "What was the revenue growth key driver in 2023?").
5. Expand the "Thinking Process" accordion to see the agent's steps.

## Architecture
- **Ingestion**: `UnstructuredPDFLoader` -> `RecursiveCharacterTextSplitter` -> `OpenAIEmbeddings` -> `OpenSearch`.
- **Agent**: `LangGraph` StateGraph.
- **UI**: `Next.js` + `Tailwind` + `Framer Motion` + `SSE`.
# Shadow-CFO
