from pandas.core.internals.construction import dataclasses_to_dicts
import os
from typing import List
from fastapi import UploadFile
import shutil
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configuration
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = "shadow_cfo_docs"
EMBEDDING_MODEL = "text-embedding-3-small"

# Use a thread pool for the heavy CPU lifting of PDF parsing
executor = ThreadPoolExecutor(max_workers=4)

def get_opensearch_client():
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection
    )

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Initialize the summarizer (Llama 3.1 is excellent for this)
summarizer_llm = ChatOllama(model="llama3.1", temperature=0)

async def get_global_context(docs: List) -> dict:
    """Extracts Company Name and Period from the cover page (first few elements)."""
    # Take first 10 elements to be safe
    cover_text = "\n".join([d.page_content for d in docs[:10]])
    
    prompt = f"""
    SYSTEM: You are a Financial Document Classifier.
    TASK: Extract the following from this document text:
    1. Exact Company Name
    2. Primary Reporting Date/Period (e.g. "December 27, 2025" or "Q3 2025")
    
    Return ONLY a JSON object like:
    {{"company": "Company Name", "period": "The Date"}}
    
    TEXT:
    {cover_text}
    """
    try:
        response = await summarizer_llm.ainvoke([HumanMessage(content=prompt)])
        # Simple cleanup in case LLM adds markdown blocks
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return {"company": "the company", "period": "the reporting period"}

async def generate_table_summary(table_content: str, table_title: str, global_context: dict) -> str:
    """Universal summarizer that anchors data to global context."""
    prompt = f"""
    SYSTEM: You are a Expert Financial Data Analyst. 
    GLOBAL CONTEXT:
    - Company: {global_context['company']}
    - Main Period: {global_context['period']}
    - Section: {table_title}

    TASK: Convert the HTML table into factual sentences. 
    
    RULES:
    1. SUBJECT: Every sentence must start with "{global_context['company']}".
    2. DATES: If the table says 'Current Year' or 'Period End', use "{global_context['period']}".
    3. MAPPING: Link every metric (row) to its value and the column date.
    4. NO PAGE NUMBERS: Do not treat small integers (like 18, 19, 20) as years or dates; those are likely page numbers.
    5. FORMAT: "[Company] [Metric] was [Value] for the period ended [Date]."
    
    DATA:
    {table_content}
    """
    response = await summarizer_llm.ainvoke([HumanMessage(content=prompt)])
    return response.content

# Use a thread pool for the heavy CPU lifting of PDF parsing
executor = ThreadPoolExecutor(max_workers=4)

async def process_pdf(file: UploadFile):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loop = asyncio.get_event_loop()
        loader = UnstructuredPDFLoader(temp_path, mode="elements", strategy="hi_res", infer_table_structure=True)
        docs = await loop.run_in_executor(executor, loader.load)
        
        # 1. EXTRACT GLOBAL CONTEXT (Who and When)
        global_meta = await get_global_context(docs)
        print(f"Processing document for: {global_meta['company']} - {global_meta['period']}")

        processed_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        # 2. TRACK SECTION TITLES
        current_title = "Financial Report"

        for doc in docs:
            category = doc.metadata.get('category')
            
            if category == 'Title':
                current_title = doc.page_content

            if category == 'Table':
                html_content = doc.metadata.get('text_as_html')
                content_to_summarize = html_content if html_content else doc.page_content
                
                # Use Global Context + Current Section Title
                summary = await generate_table_summary(content_to_summarize, current_title, global_meta)
                
                doc.page_content = f"Financial Statement Data for {global_meta['company']}: {summary}"
                doc.metadata["raw_table_content"] = html_content
                doc.metadata["is_table"] = True
                processed_docs.append(doc)
            
            elif category in ['NarrativeText', 'Title', 'UncategorizedText']:
                # Add company name to narrative chunks to improve metadata retrieval
                doc.page_content = f"Context: {global_meta['company']} {current_title}\n{doc.page_content}"
                
                if len(doc.page_content) > 2000:
                    processed_docs.extend(text_splitter.split_documents([doc]))
                else:
                    processed_docs.append(doc)
            else:
                continue

        if not processed_docs:
            return {"status": "error", "message": "No content extracted"}

        print(f"Indexing {len(processed_docs)} chunks to OpenSearch...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        docsearch = OpenSearchVectorSearch.from_documents(
            processed_docs,
            embeddings,
            opensearch_url=OPENSEARCH_URL,
            index_name=OPENSEARCH_INDEX,
            engine="faiss",
            space_type="l2",
            bulk_size=1000
        )
        
        return {"status": "success", "chunks_processed": len(processed_docs)}
        
    except Exception as e:
        print(f"Error: {e}")
        raise e
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
    
    # We increase 'k' because we want to catch both the summary and potentially 
    # the narrative text surrounding it.
    return vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 7, "fetch_k": 20} # Fetch 20, return the 7 most diverse
    )
