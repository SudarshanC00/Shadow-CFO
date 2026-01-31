import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from ingestion import process_pdf
# from agent import run_agent_stream # Will implement agent next

app = FastAPI(title="Shadow CFO API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

@app.get("/")
def read_root():
    return {"message": "Shadow CFO Backend logic is Running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        result = await process_pdf(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse
import json
from agent import app_graph
from langchain_core.messages import HumanMessage

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    async def event_generator():
        inputs = {"messages": [HumanMessage(content=request.message)]}
        # Optionally restore history into inputs['messages'] here if needed
        
        async for event in app_graph.astream_events(inputs, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            elif kind == "on_tool_start":
                 yield f"data: {json.dumps({'type': 'step', 'content': f'Starting tool: {event['name']}'})}\n\n"
            elif kind == "on_chain_end":
                # mapping to nodes
                if event['name'] == 'retrieve':
                     yield f"data: {json.dumps({'type': 'step', 'content': 'Retrieved documents'})}\n\n"
                elif event['name'] == 'verifier':
                     yield f"data: {json.dumps({'type': 'step', 'content': 'Verified answer'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
