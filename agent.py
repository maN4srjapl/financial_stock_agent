from typing import List, Dict, Any, Union, Annotated
import uuid
import time
import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_neo4j import Neo4jGraph
import requests 

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "12345678")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Validate API key exists
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env file or environment variables.")


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

# 5. Entity Cache for Optimization
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

# 6. LangGraph Agent Loop Functions

def call_model(state: MessagesState):
    """Call the LLM with message history and tool bindings."""
    messages = state["messages"]
    
    # Dynamic Model Selection
    last_message_content = messages[-1].content if messages else ""
    if len(last_message_content.split()) > 10 or "analyze" in last_message_content.lower() or "compare" in last_message_content.lower():
        print("[System] Using Advanced Model (gpt-4o)")
        model = advanced_model.bind_tools(tools)
    else:
        print("[System] Using Basic Model (gpt-4o-mini)")
        model = basic_model.bind_tools(tools)
    
    # System prompt with memory context
    system_prompt = """You are a Financial Copilot AI assistant. You help analyze financial data, business segments, and company performance.

IMPORTANT GUIDELINES:
1. Use the conversation history to provide context-aware answers
2. Leverage previous tool calls to avoid redundant searches
3. When referring to past information, acknowledge it was discussed earlier
4. Always verify current data with fresh tool searches when needed
5. Be transparent about which tools you're using and why
6. Combine insights from multiple sources (Vector DB, Graph DB, News, Earnings)

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

def should_continue(state: MessagesState):
    """Determine if tool execution is needed or if we should finalize the answer."""
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, go to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, we're done
    return END

# 7. Build the LangGraph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.add_edge(START, "agent")

# Add conditional edges
workflow.add_conditional_edges("agent", should_continue)

# Tools always loop back to agent for next decision
workflow.add_edge("tools", "agent")

# 8. Setup Checkpointer for Memory
checkpointer = MemorySaver()

# 9. Compile the graph
app = workflow.compile(checkpointer=checkpointer)

# 10. Conversation Manager
class ConversationManager:
    """Manages multiple conversations with persistent memory."""
    
    def __init__(self):
        self.conversations = {}  # {thread_id: conversation_metadata}
    
    def start_conversation(self, user_id: str = None) -> str:
        """Start a new conversation and return thread ID."""
        thread_id = str(uuid.uuid4())
        self.conversations[thread_id] = {
            "user_id": user_id or "anonymous",
            "created_at": datetime.now(),
            "message_count": 0
        }
        print(f"\n[Conversation Started] Thread ID: {thread_id}")
        return thread_id
    
    def chat(self, thread_id: str, user_query: str) -> str:
        """Send a message in an existing conversation."""
        if thread_id not in self.conversations:
            raise ValueError(f"Thread {thread_id} not found. Start a new conversation first.")
        
        # Prepare input
        inputs = {"messages": [HumanMessage(content=user_query)]}
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the agent
        print(f"\n[User Query]: {user_query}")
        final_state = app.invoke(inputs, config=config)
        
        # Update conversation metadata
        self.conversations[thread_id]["message_count"] += 1
        
        # Extract and return the final answer
        last_message = final_state["messages"][-1]
        return last_message.content
    
    def get_conversation_history(self, thread_id: str) -> List[Dict]:
        """Retrieve full conversation history."""
        if thread_id not in self.conversations:
            raise ValueError(f"Thread {thread_id} not found.")
        
        # Retrieve state from checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)
        
        history = []
        for msg in state.values.get("messages", []):
            history.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            })
        return history
    
    def clear_conversation(self, thread_id: str):
        """Clear a conversation from memory."""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
            print(f"[Conversation Cleared] Thread ID: {thread_id}")

# 11. Main Execution
if __name__ == "__main__":
    # Create conversation manager
    manager = ConversationManager()
    
    # Start a new conversation
    thread_id = manager.start_conversation(user_id="demo_user")
    
    # Multi-turn conversation example
    print("\n" + "="*80)
    print("FINANCIAL COPILOT - MULTI-TURN CONVERSATION WITH MEMORY")
    print("="*80)
    
    # Query 1
    query1 = "What information do we have about Reliance business segments?"
    answer1 = manager.chat(thread_id, query1)
    print(f"\n[Assistant]: {answer1}")
    
    # Query 2 (uses context from Query 1)
    query2 = "Tell me more about their oil and hydrocarbon operations."
    answer2 = manager.chat(thread_id, query2)
    print(f"\n[Assistant]: {answer2}")
    
    # Query 3 (multi-tool reasoning)
    query3 = "How has their performance been recently? Analyze the trends."
    answer3 = manager.chat(thread_id, query3)
    print(f"\n[Assistant]: {answer3}")
    
    # Display conversation history
    print("\n" + "="*80)
    print("CONVERSATION HISTORY")
    print("="*80)
    history = manager.get_conversation_history(thread_id)
    for i, msg in enumerate(history):
        print(f"\n[Turn {i+1}] {msg['role'].upper()}: {msg['content'][:100]}...")





