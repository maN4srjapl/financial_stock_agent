from typing import List, Dict, Any, Union, Annotated, TypedDict
import uuid
import time
import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_neo4j import Neo4jGraph

from conversation_memory import ConversationMemory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "12345678")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Validate API key exists
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env file or environment variables.")

# Initialize Persistent Memory
persistent_memory = ConversationMemory()

# 1. Databases Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="financial_chunks",
    url=QDRANT_URL,
    content_payload_key="page_content", # Match the key used in chunking.py
    metadata_payload_key="metadata",
)

graph = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USER, 
    password=NEO4J_PASS,
    refresh_schema=False
)

# 2. Define Tools
@tool
def search_vector_db(query: str) -> str:
    """Search for deep historical context, qualitative explanations, and document snippets from financial reports."""
    docs = vector_store.similarity_search(query, k=3)
    print("\n🔍 [SEARCH PROOF: QDRANT]")
    for i, doc in enumerate(docs):
        print(f"   - Chunk from {doc.metadata.get('source', 'N/A')}")
    return "\n".join([d.page_content for d in docs])

@tool
def search_graph_db(entity_name: str) -> str:
    """Search for structural relationships, business segments, and organizational connections in the graph."""
    cypher = f"MATCH (n)-[r]->(m) WHERE n.name CONTAINS '{entity_name}' OR m.name CONTAINS '{entity_name}' RETURN n.name, type(r), m.name LIMIT 5"
    results = graph.query(cypher)
    print("\n🔗 [SEARCH PROOF: NEO4J]")
    return "\n".join([f"({r['n.name']})-[{r['type(r)']}]->({r['m.name']})" for r in results]) if results else "No graph data."

@tool
def get_latest_news(company: str) -> str:
    """Search for real-time market news, current events, and media updates regarding a specific company."""
    print(f"\n📰 [TOOL CALL: GET_NEWS for {company}]")
    # Placeholder for actual API call (e.g., NewsAPI or Tavily)
    return f"Latest news for {company}: Market sentiment is positive following expansion into green energy. Analyst ratings updated to 'Buy'."

@tool
def get_latest_earnings(company: str) -> str:
    """Fetch recent quantitative performance data, stock metrics, and raw financial figures (Revenue, EBITDA)."""
    print(f"\n📊 [TOOL CALL: GET_EARNINGS for {company}]")
    # Placeholder for financial API (e.g., Yahoo Finance)
    return f"Earnings for {company}: Revenue: $20B (+15% YoY), EBITDA Margin: 28%, Net Income: $3.5B."

tools = [search_vector_db, search_graph_db, get_latest_news, get_latest_earnings]

# 3. Error Handling for ToolNode
def handle_tool_error(e: Any) -> str:
    return f"Tool execution failed: {str(e)}. Please retry with different parameters."

tool_node = ToolNode(tools, handle_tool_errors=handle_tool_error)

# 4. Setup Models
basic_model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
advanced_model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0)

entity_cache = {}

def get_cached_entity(entity_name: str, max_age_hours: int = 24):
    """Retrieve cached entity data if fresh, else None."""
    if entity_name in entity_cache:
        cached = entity_cache[entity_name]
        if time.time() - cached["timestamp"] < max_age_hours * 3600:
            return cached["results"]
    return None

def cache_entity(entity_name: str, results: str):
    """Cache entity search results."""
    entity_cache[entity_name] = {
        "timestamp": time.time(),
        "results": results
    }

# 6. Extended State for Memory Context
class AgentState(MessagesState):
    """Extended state with memory context and user tracking."""
    user_id: str
    memory_context: str
    needs_tools: bool
    raw_response: str

# 7. LangGraph Agent Loop Functions

def retrieve_memory_context(state: AgentState):
    """Step 1: Retrieve relevant context from Neo4j memory."""
    messages = state["messages"]
    user_id = state.get("user_id", "default-user")
    
    if not messages:
        return {"memory_context": "", "needs_tools": True}
    
    last_user_message = messages[-1].content
    print(f"\n🔍 [STEP 1: SEARCHING MEMORY for user: {user_id}]")
    
    try:
        # Search persistent memory for relevant past context
        past_memories = persistent_memory.search_memory(
            query=last_user_message,
            user_id=user_id,
            limit=5,
            rerank=True
        )
        
        if past_memories:
            memory_context = "\n\nRELEVANT PAST CONTEXT FROM MEMORY:\n"
            for idx, mem in enumerate(past_memories, 1):
                memory_context += f"{idx}. {mem.get('memory', 'N/A')} (Score: {mem.get('score', 0):.2f})\n"
            print(f"   ✓ Found {len(past_memories)} relevant memories")
            return {"memory_context": memory_context, "needs_tools": False}
        else:
            print("   ⚠ No relevant memories found")
            return {"memory_context": "", "needs_tools": True}
    except Exception as e:
        print(f"   ✗ Memory search failed: {e}")
        return {"memory_context": "", "needs_tools": True}

def assess_context_sufficiency(state: AgentState):
    """Step 2: Determine if memory context is sufficient or if tools are needed."""
    messages = state["messages"]
    memory_context = state.get("memory_context", "")
    
    print(f"\n🤔 [STEP 2: ASSESSING CONTEXT SUFFICIENCY]")
    
    # Use LLM to determine if memory context is sufficient
    assessment_prompt = f"""You are a context assessor. Determine if the provided memory context is sufficient to answer the user's query.

User Query: {messages[-1].content}

Memory Context: {memory_context if memory_context else "No memory context available"}

Respond with ONLY 'SUFFICIENT' or 'INSUFFICIENT'.
Use 'SUFFICIENT' only if the memory context directly answers the query.
Use 'INSUFFICIENT' if you need current data, additional details, or the context is empty."""
    
    assessment = basic_model.invoke([{"role": "user", "content": assessment_prompt}])
    is_sufficient = "SUFFICIENT" in assessment.content.upper()
    
    if is_sufficient:
        print("   ✓ Memory context is SUFFICIENT")
        return {"needs_tools": False}
    else:
        print("   ⚠ Memory context is INSUFFICIENT - tools required")
        return {"needs_tools": True}

def call_model_with_context(state: AgentState):
    """Step 3a: Generate response using memory context only (no tools)."""
    messages = state["messages"]
    memory_context = state.get("memory_context", "")
    
    print(f"\n💬 [STEP 3a: GENERATING RESPONSE FROM MEMORY]")
    
    model = advanced_model
    
    system_prompt = f"""You are a Financial Copilot AI assistant. Answer the user's query using the provided memory context.

{memory_context}

Provide a clear, concise answer based on this context. Acknowledge that this is from previous conversations."""

    response = model.invoke([
        {"role": "system", "content": system_prompt},
        *messages
    ])
    
    return {"messages": [response], "raw_response": response.content}

def call_model_with_tools(state: AgentState):
    """Step 3b: Generate response using tools for fresh data."""
    messages = state["messages"]
    memory_context = state.get("memory_context", "")
    
    print(f"\n🛠️  [STEP 3b: GENERATING RESPONSE WITH TOOLS]")
    
    # Dynamic Model Selection
    last_message_content = messages[-1].content if messages else ""
    if len(last_message_content.split()) > 10 or "analyze" in last_message_content.lower() or "compare" in last_message_content.lower():
        model = advanced_model.bind_tools(tools)
    else:
        model = basic_model.bind_tools(tools)
    
    system_prompt = f"""You are a Financial Copilot AI assistant. You help analyze financial data, business segments, and company performance.

{memory_context}

IMPORTANT GUIDELINES:
1. Use tools to fetch current, accurate data
2. Combine memory context with fresh tool results
3. Be transparent about data sources
4. Prioritize recent data from tools over memory

Tools available:
- search_vector_db: For deep historical context and document snippets
- search_graph_db: For business relationships and structural data
- get_latest_news: For real-time market information
- get_latest_earnings: For quantitative financial metrics"""

    response = model.invoke([
        {"role": "system", "content": system_prompt},
        *messages
    ])
    
    return {"messages": [response]}

def enrich_and_refine_response(state: AgentState):
    """Step 4: Enrich and refine the final response before storing."""
    messages = state["messages"]
    user_id = state.get("user_id", "default-user")
    
    print(f"\n✨ [STEP 4: ENRICHING & REFINING RESPONSE]")
    
    last_response = messages[-1].content
    
    # Enrichment prompt
    enrichment_prompt = f"""You are a response enhancer. Take the following response and:
1. Add relevant context and explanations
2. Structure it clearly with bullet points or sections
3. Ensure accuracy and completeness
4. Keep it concise but informative

Original Response:
{last_response}

Provide the ENRICHED version:"""
    
    enriched = advanced_model.invoke([{"role": "user", "content": enrichment_prompt}])
    
    print("   ✓ Response enriched and refined")
    
    # Update the last message with enriched content
    messages[-1] = AIMessage(content=enriched.content)
    
    return {"messages": messages, "raw_response": enriched.content}

def store_in_memory(state: AgentState):
    """Step 5: Store the enriched conversation in persistent memory."""
    messages = state["messages"]
    user_id = state.get("user_id", "default-user")
    
    print(f"\n💾 [STEP 5: STORING IN MEMORY]")
    
    try:
        # Find the last user message and assistant response
        user_msg = None
        assistant_msg = None
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and assistant_msg is None:
                assistant_msg = msg.content
            elif isinstance(msg, HumanMessage) and user_msg is None:
                user_msg = msg.content
            
            if user_msg and assistant_msg:
                break
        
        if user_msg and assistant_msg:
            conversation_turn = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
            
            # Extract metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "used_tools": state.get("needs_tools", False)
            }
            
            # Add keywords if they exist
            keywords = []
            for keyword in ["Reliance", "segment", "earnings", "news", "analysis", "business"]:
                if keyword.lower() in user_msg.lower():
                    keywords.append(keyword)
            if keywords:
                metadata["keywords"] = keywords
            
            persistent_memory.add_memory(
                conversation_turn,
                user_id=user_id,
                metadata=metadata
            )
            print("   ✓ Conversation stored successfully")
    except Exception as e:
        print(f"   ✗ Failed to store in memory: {e}")
    
    return {"messages": messages}

def route_based_on_context(state: AgentState):
    """Router: Decide whether to use memory-only or tools-based response."""
    needs_tools = state.get("needs_tools", True)
    
    if needs_tools:
        return "call_with_tools"
    else:
        return "call_with_memory"

def should_continue(state: AgentState):
    """Determine if tool execution is needed or if we should proceed to enrichment."""
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, go to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, proceed to enrichment
    return "enrich"

# 8. Build the LangGraph with New Flow
workflow = StateGraph(AgentState)

# Add nodes for the new flow
workflow.add_node("retrieve_memory", retrieve_memory_context)
workflow.add_node("assess_context", assess_context_sufficiency)
workflow.add_node("call_with_memory", call_model_with_context)
workflow.add_node("call_with_tools", call_model_with_tools)
workflow.add_node("tools", tool_node)
workflow.add_node("enrich", enrich_and_refine_response)
workflow.add_node("store", store_in_memory)

# Define the flow
workflow.add_edge(START, "retrieve_memory")  # Step 1: Search memory
workflow.add_edge("retrieve_memory", "assess_context")  # Step 2: Assess sufficiency
workflow.add_conditional_edges("assess_context", route_based_on_context)  # Route based on needs_tools

# Memory-only path
workflow.add_edge("call_with_memory", "enrich")

# Tools path
workflow.add_conditional_edges("call_with_tools", should_continue)  # Check if tools needed
workflow.add_edge("tools", "call_with_tools")  # Loop back after tool execution

# Final steps (common path)
workflow.add_edge("enrich", "store")  # Step 4: Enrich then store
workflow.add_edge("store", END)  # Step 5: End after storing

# 9. Compile the graph
app = workflow.compile()

# 10. Main Execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("FINANCIAL COPILOT - MEMORY-FIRST AGENT")
    print("="*80)
    print("Flow: Search Memory → Assess → Route → Enrich → Store")
    print("="*80)
    
    # Set user ID for this session
    user_id = "analyst-001"
    
    # Single query example
    query = "What information do we have about Reliance business segments?"
    print(f"\n[User Query]: {query}")
    
    inputs = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "memory_context": "",
        "needs_tools": True,
        "raw_response": ""
    }
    
    final_state = app.invoke(inputs)
    
    # Extract and display the final answer
    last_message = final_state["messages"][-1]
    print(f"\n{'='*80}")
    print("[FINAL ENRICHED RESPONSE]:")
    print("="*80)
    print(last_message.content)
    print("\n" + "="*80)





