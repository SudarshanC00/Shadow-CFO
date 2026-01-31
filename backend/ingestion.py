import os
from typing import List
from fastapi import UploadFile
import shutil
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection

# Configuration
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = "shadow_cfo_docs"
EMBEDDING_MODEL = "nomic-embed-text"

def get_opensearch_client():
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection
    )

async def process_pdf(file: UploadFile):
    # Save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Load and Parse
        print(f"Parsing {file.filename}...")
        loader = UnstructuredPDFLoader(
            temp_path,
            mode="elements", # Preserves metadata like table structure
            strategy="hi_res"  # High-res for better table extraction - tesseract installed
        )
        docs = loader.load()
        
        # Chunking - increased size to preserve tables
        print(f"Chunking {len(docs)} elements...")
        
        # Separate tables from other content
        table_docs = [doc for doc in docs if doc.metadata.get('category') == 'Table']
        non_table_docs = [doc for doc in docs if doc.metadata.get('category') != 'Table']
        
        print(f"Found {len(table_docs)} table elements, {len(non_table_docs)} non-table elements")
        
        # Only split non-table content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased from 1000 to preserve more context
            chunk_overlap=200,  # Increased overlap
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
            is_separator_regex=False
        )
        split_non_tables = text_splitter.split_documents(non_table_docs)
        
        # Combine: keep tables intact, split everything else
        split_docs = table_docs + split_non_tables
        
        # Add enhanced metadata
        for doc in split_docs:
            doc.metadata["source"] = file.filename
            # Mark tables for prioritization in retrieval
            if doc.metadata.get('category') == 'Table':
                doc.metadata['is_table'] = True
                doc.metadata['preserve_formatting'] = True

        
        # Indexing
        print("Indexing to OpenSearch...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Process in batches of 500 to avoid payload size issues with large documents
        batch_size = 500
        total_chunks = len(split_docs)
        
        # Create index with first batch
        first_batch = split_docs[:batch_size]
        docsearch = OpenSearchVectorSearch.from_documents(
            first_batch,
            embeddings,
            opensearch_url=OPENSEARCH_URL,
            http_auth=None,
            use_ssl=False,
            verify_certs=False,
            index_name=OPENSEARCH_INDEX,
            engine="faiss",
            bulk_size=2000,
            validate_mapping=True
        )
        
        # Add remaining documents in batches
        for i in range(batch_size, total_chunks, batch_size):
            batch = split_docs[i:i + batch_size]
            docsearch.add_documents(batch)
            print(f"Indexed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")

        
        return {"status": "success", "chunks_processed": len(split_docs)}
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name=OPENSEARCH_INDEX,
        embedding_function=embeddings,
        http_auth=None,
        use_ssl=False,
        verify_certs=False
    )
    # Configure generic retriever, can be updated to specific Hybrid search params later
    return vector_store.as_retriever(search_kwargs={"k": 5})
