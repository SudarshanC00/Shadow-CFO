import operator
import re
from typing import Annotated, Sequence, TypedDict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from ingestion import get_retriever

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    documents: List[str]
    steps: List[str]
    current_answer: str
    retry_count: int
    auditor_feedback: str

# --- Tools ---
@tool
def calculator(expression: str) -> str:
    """Calculates results using python syntax. Required for financial calculations."""
    try:
        # In production, replace eval with a safer math parser
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [calculator]
tool_node = ToolNode(tools)

# --- Nodes ---

def query_elaboration_node(state: AgentState):
    print("\n--- [NODE] QUERY ELABORATION ---")
    user_query = state['messages'][-1].content
    
    model = ChatOllama(model="llama3.1", temperature=0)
    
    elaboration_prompt = f"""
    You are a Financial Search Specialist. Your task is to expand the user's query into a precise set of search terms for a 10-Q filing.
    
    USER QUERY: {user_query}
    
    GOAL:
    1. Identify the likely Financial Statement needed (e.g., Statements of Operations, Balance Sheet).
    2. Identify exact technical line items (e.g., 'Total net sales', 'Cost of sales', 'Accounts receivable').
    3. Include relevant dates mentioned or implied (e.g., 'three months ended December 27, 2025').
    
    OUTPUT: 
    Return ONLY a single string of optimized search terms. Do not provide conversational filler.
    """
    
    response = model.invoke([HumanMessage(content=elaboration_prompt)])
    print(f"Expanded Query: {response.content}")
    
    # Store the expanded query in a new key in state
    return {"messages": [AIMessage(content=f"Expanded Query: {response.content}")], "steps": ["Elaborated user query"]}

def retrieve_node(state):
    print("\n--- [NODE] RETRIEVE & REHYDRATE ---")
    # Use the elaboration if available, otherwise fallback to the last user message
    if len(state['messages']) > 1 and "Expanded Query:" in state['messages'][-1].content:
        query = state['messages'][-1].content.replace("Expanded Query: ", "")
    else:
        query = state['messages'][-1].content

    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        rehydrated_context = []
        for doc in docs:
            is_table = doc.metadata.get('is_table', False)
            content = doc.metadata.get('raw_table_content', doc.page_content)
            prefix = "[TABLE DATA]" if is_table else "[TEXT DATA]"
            rehydrated_context.append(f"{prefix} Source: {doc.metadata.get('source')}\n{content}")
        return {"documents": rehydrated_context, "steps": ["Retrieved context chunks"]}
    except Exception as e:
        return {"documents": [], "steps": [f"Retrieval Error: {e}"]}

def analyst_node(state):
    print("\n--- [NODE] ANALYST (OLLAMA LLAMA 3.1) ---")
    context = "\n\n".join(state.get('documents', []))
    
    # Using Llama 3.1 for Analysis
    model = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)
    
    system_msg = f"""You are a Senior CFO Auditor. Your goal is 100% numerical accuracy.

            **STEP 1: IDENTIFY THE TARGET TABLE**
            Based on the user's query, determine which financial statement is needed:
            - Revenue/Net Sales -> "Statements of Operations"
            - Cash/Debt/Assets -> "Balance Sheets"
            - Cash Inflow/Outflow -> "Statements of Cash Flows"
            - Detailed Breakdown -> "Notes to Financial Statements"

            **STEP 2: SCAN CONTEXT**
            Check the provided context for headers matching these statements. 
            - If the required table is present, extract the Row x Column intersection.
            - If the table is NOT present, state: "The specific financial table for [Topic] was not found in the retrieved context. I can only see narrative text regarding [Summary of text]."

            **STEP 3: NO HALLUCINATIONS**
            - NEVER assume a number. 
            - NEVER use placeholders like $X.
            - If you see "In millions" in a header, ensure your final answer reflects that.

            **CONTEXT:**
        {context}
        """
    
    response = model.invoke([HumanMessage(content=system_msg)] + state['messages'])
    return {"messages": [response], "current_answer": response.content}

def verifier_node(state):
    print("\n--- [NODE] VERIFIER (OLLAMA LLAMA 3.1) ---")
    answer = state['current_answer']
    context = "\n\n".join(state.get('documents', []))
    
    model = ChatOllama(model="llama3.1", temperature=0)
    
    verify_prompt = f"""
    Role: Senior Forensic Auditor.
    Task: Audit the Analyst's Answer against the provided Context for absolute numerical integrity.
    
    CONTEXT DATA:
    {context}
    
    PROPOSED ANSWER:
    {answer}
    
    AUDIT CHECKLIST:
    1. METRIC ALIGNMENT: Does the name of the financial metric in the answer (e.g., 'Net Sales') match the EXACT row header in the context table? 
       - WARNING: Do not confuse 'Net Sales' with 'Deferred Revenue' or 'Comprehensive Income'.
    2. TEMPORAL ACCURACY: Does the value match the correct date column (e.g., Dec 27, 2025 vs Dec 28, 2024)?
    3. SCALE CHECK: If the table says "In millions", does the answer correctly reflect that (e.g., $143,756 million = $143.76 billion)?
    4. NO HALLUCINATION: Are there any numbers in the answer that do not exist in the context?
    
    REQUIRED OUTPUT FORMAT:
    Groundedness Score: [0.0 to 1.0]
    Violations: [List specific mismatches or "None"]
    Correction Instruction: [If score < 1.0, provide the exact metric and value the model should use instead]
    """
    
    response = model.invoke([HumanMessage(content=verify_prompt)])
    return {
        "messages": [AIMessage(content=response.content)],
        "auditor_feedback": response.content
    }

def regenerator_node(state):
    # Logic remains similar but using Llama 3.1
    print(f"\n--- [NODE] REGENERATOR (RETRY {state.get('retry_count', 0) + 1}) ---")
    feedback = state.get('auditor_feedback', "")
    context = "\n\n".join(state.get('documents', []))
    
    model = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)
    
    correction_prompt = f"""The previous response was REJECTED for hallucinations or inaccuracy.
    
    AUDITOR FEEDBACK: {feedback}
    
    NEW DATA CONTEXT:
    {context}
    
    RE-READ THE TABLES. Use actual values only. If a value is missing, state exactly which table you are looking for that is not there."""
    
    response = model.invoke([HumanMessage(content=correction_prompt)] + state['messages'])
    return {
        "messages": [response],
        "current_answer": response.content,
        "retry_count": state.get('retry_count', 0) + 1
    }

# --- Router Logic ---

def should_continue(state):
    last_msg = state['messages'][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tool"
    return "verifier"

def check_groundedness(state):
    feedback = state.get('auditor_feedback', "")
    print(f"DEBUG: Auditor says: {feedback[:100]}...")
    retries = state.get('retry_count', 0)
    
    # Extraction of the score
    score_match = re.search(r"Groundedness Score:\s*(\d?\.\d+|1\.0|1)", feedback)
    score = float(score_match.group(1)) if score_match else 0.0
    
    print(f"--- [EDGE] Final Score: {score} | Retry: {retries} ---")
    
    if score < 1.0:
        if retries < 2:
            print(">>> Rerouting to REGENERATOR for self-correction...")
            return "regenerator"
        else:
            print(">>> Max retries reached. Exiting.")
            return END
    
    print(">>> Accuracy Verified. Exiting.")
    return END

# --- Graph Assembly ---
workflow = StateGraph(AgentState)

workflow.add_node("elaborate", query_elaboration_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("regenerator", regenerator_node)
workflow.add_node("tool", tool_node)

workflow.set_entry_point("elaborate")
workflow.add_edge("elaborate", "retrieve")
workflow.add_edge("retrieve", "analyst")

# Transition from Analyst
workflow.add_conditional_edges("analyst", should_continue, {
    "tool": "tool",
    "verifier": "verifier"
})

# Transition from Tool back to Analyst
workflow.add_edge("tool", "analyst")

# Transition from Verifier (THE LOOP)
workflow.add_conditional_edges("verifier", check_groundedness, {
    "regenerator": "regenerator",
    END: END
})

# Transition from Regenerator back to Verifier to re-check
workflow.add_edge("regenerator", "verifier")

app_graph = workflow.compile()