import os
import json
import shutil
import asyncio
from typing import List, Dict, Any
from fastapi import UploadFile
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.messages import HumanMessage
from opensearchpy import OpenSearch, RequestsHttpConnection

# Configuration
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = "shadow_cfo_docs"
EMBEDDING_MODEL = "nomic-embed-text"


# Global executor for PDF parsing (CPU intensive)
executor = ThreadPoolExecutor(max_workers=8)

# Initialize the LLM (using Ollama for local processing)
llm = ChatOllama(model="llama3.1", temperature=0)

def get_opensearch_client():
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection
    )

async def generate_table_summary(table_content: str, table_title: str) -> str:
    """Creates a highly detailed and searchable textualization of a financial table."""
    prompt = f"""
    SYSTEM: You are an Expert Financial Data Analyst specializing in high-fidelity data extraction.
    
    TASK: Convert the provided table data (HTML or Text) into a detailed, prose-like textual representation.
    
    RULES:
    1. CONTEXT: Use the provided Section Title: "{table_title}" to anchor all metrics.
    2. DETAIL: For every row in the table, create a full sentence explaining the metric, its value, and the specific date or period it refers to.
    3. STRUCTURE: 
       - Start with a summary sentence: "This table titled '{table_title}' provides financial data for..."
       - Group data by period (year/quarter) if multiple columns exist.
       - Use the format: "The metric [Metric Name] was [Value] for the period ended [Date]."
    4. PRECISION:
       - Match names EXACTLY as they appear in the row headers.
       - Handle footnotes or parenthetical info if present (e.g., "including share-based compensation").
       - Preserve the sign of the numbers (e.g., "(1,000)" should be "negative 1,000" or stated as a loss/decrease).
    5. UNITS: Explicitly state "in millions" or "in thousands" if indicated in the table headers or metadata.
    6. NO HALLUCINATION: If a cell is empty or "—", state it as "zero" or "not applicable". If you are unsure, do not invent data.
    7. METRICS SIGNATURE: At the end, provide a "Searchable Metrics Key": [Comma-separated list of all metric row headers].
    
    TABLE CONTENT:
    {table_content}
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Error textualizing table: {str(e)}"

async def process_pdf(file: UploadFile):
    client = get_opensearch_client()

    # This deletes everything in the index so you start fresh
    if client.indices.exists(index=OPENSEARCH_INDEX):
        client.indices.delete(index=OPENSEARCH_INDEX)
        print(f"Index {OPENSEARCH_INDEX} cleared.")
        
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loop = asyncio.get_event_loop()
        # Using elements mode preserves table structure and metadata
        loader = UnstructuredPDFLoader(temp_path, mode="elements", strategy="ocr_only", hi_res_model_name="yolox")
        docs = await loop.run_in_executor(executor, loader.load)
        
        if not docs:
            return {"status": "error", "message": "No content extracted from PDF"}

        processed_docs = []
        table_tasks = []
        table_indices = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        current_section = "General Financial Report"

        for i, doc in enumerate(docs):
            category = doc.metadata.get('category')
            page = doc.metadata.get('page_number', 1)
            
            # Enrich basic metadata
            doc.metadata.update({
                "element_type": category,
                "table_title": current_section if category == 'Table' else None
            })


            if category == 'Title':
                current_section = doc.page_content
                # Keep titles as context for grounding
                doc.page_content = f"Section Header: {current_section}"
                processed_docs.append(doc)

            elif category == 'Table':
                html_content = doc.metadata.get('text_as_html')
                table_raw = html_content if html_content else doc.page_content
                # Queue the task for parallel execution
                table_tasks.append(generate_table_summary(table_raw, current_section))
                table_indices.append(len(processed_docs)) 
                
                # Placeholder metadata
                doc.metadata.update({"is_table": True, "table_title": current_section})
                processed_docs.append(doc)

            elif category in ['NarrativeText', 'UncategorizedText']:
                if len(doc.page_content) < 50: continue
                
                doc.page_content = f"[Section: {current_section} | Page {page}]\n" + doc.page_content
                if len(doc.page_content) > 1500:
                    processed_docs.extend(text_splitter.split_documents([doc]))
                else:
                    processed_docs.append(doc)

        # SPEED FIX 3: Execute all LLM table summaries in parallel
        if table_tasks:
            print(f"Summarizing {len(table_tasks)} tables in parallel...")
            summaries = await asyncio.gather(*table_tasks)
            for idx, summary in zip(table_indices, summaries):
                processed_docs[idx].page_content = summary

        # Persistence: Save processed content to a file for debugging/audit
        storage_dir = "data/ingested"
        os.makedirs(storage_dir, exist_ok=True)
        storage_path = os.path.join(storage_dir, f"{file.filename}_processed.json")
        
        serializable_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in processed_docs
        ]
        
        with open(storage_path, "w") as f:
            json.dump(serializable_docs, f, indent=2)
        print(f"Saved processed content to {storage_path}")

        # Indexing
        print(f"Indexing {len(processed_docs)} chunks...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Manual batching to avoid timeouts and provide progress updates
        batch_size = 100
        for i in range(0, len(processed_docs), batch_size):
            batch = processed_docs[i : i + batch_size]
            print(f"  Indexing batch {i//batch_size + 1}/{(len(processed_docs)-1)//batch_size + 1}...")
            
            if i == 0:
                # First batch creates/overwrites the index
                vector_store = OpenSearchVectorSearch.from_documents(
                    batch,
                    embeddings,
                    opensearch_url=OPENSEARCH_URL,
                    index_name=OPENSEARCH_INDEX,
                    engine="faiss",
                    space_type="l2"
                )
            else:
                # Subsequent batches add to existing index
                vector_store.add_documents(batch)
        
        print("Indexing complete.")
        return {"status": "success", "chunks": len(processed_docs)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name=OPENSEARCH_INDEX,
        embedding_function=embeddings
    )
    
    return vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 7, "fetch_k": 20}
    )
