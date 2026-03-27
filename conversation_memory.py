"""
Conversation Memory Manager
Stores and retrieves conversation history using Qdrant vector database.
Enables semantic search across past conversations.
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import hashlib


class ConversationMemoryManager:
    """
    Manages conversation history storage and retrieval in Qdrant.
    Each conversation exchange (user query + assistant response) is embedded and stored.
    """
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "conversation_history"):
        """
        Initialize the conversation memory manager.
        
        Args:
            qdrant_url: URL of Qdrant server
            collection_name: Name of collection to store conversations
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create or get collection
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        from qdrant_client.http.models import VectorParams, Distance
        
        try:
            self.client.get_collection(self.collection_name)
            print(f"✓ Collection '{self.collection_name}' already exists")
        except:
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding dimension
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Collection '{self.collection_name}' created successfully")
    
    def save_conversation_turn(
        self,
        thread_id: str,
        user_query: str,
        assistant_response: str,
        tools_used: List[str] = None
    ) -> str:
        """
        Save a single conversation turn (user query + assistant response) to Qdrant.
        
        Args:
            thread_id: Unique conversation thread identifier
            user_query: User's question/input
            assistant_response: Assistant's response
            tools_used: List of tools invoked (e.g., ["search_graph_db", "get_earnings"])
        
        Returns:
            Point ID (unique conversation turn identifier)
        """
        if tools_used is None:
            tools_used = []
        
        # Create combined text for embedding
        combined_text = f"Query: {user_query}\n\nResponse: {assistant_response}"
        
        # Generate embedding
        embedding = self.embeddings.embed_query(combined_text)
        
        # Create unique ID for this turn
        turn_id = int(hashlib.md5(f"{thread_id}{user_query}{datetime.now().isoformat()}".encode()).hexdigest(), 16) % (10**10)
        
        # Prepare metadata
        metadata = {
            "thread_id": thread_id,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "tools_used": json.dumps(tools_used),
            "timestamp": datetime.now().isoformat(),
            "turn_number": self._get_turn_count(thread_id) + 1
        }
        
        # Create and upload point
        point = PointStruct(
            id=turn_id,
            vector=embedding,
            payload=metadata
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        print(f"✓ Saved conversation turn {metadata['turn_number']} for thread {thread_id}")
        return str(turn_id)
    
    def search_conversations(
        self,
        query: str,
        thread_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search across conversation history using semantic similarity.
        
        Args:
            query: Search query (e.g., "What was discussed about Reliance's strategy?")
            thread_id: Optional - filter results to specific conversation thread
            limit: Maximum number of results to return
        
        Returns:
            List of relevant past conversation turns with scores
        """
        # Generate embedding for search query
        query_embedding = self.embeddings.embed_query(query)
        
        # Build filter if thread_id specified
        filter_condition = None
        if thread_id:
            filter_condition = {
                "must": [
                    {
                        "key": "thread_id",
                        "match": {"value": thread_id}
                    }
                ]
            }
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filter_condition,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "score": result.score,
                "turn_id": result.id,
                "query": result.payload.get("user_query", ""),
                "response": result.payload.get("assistant_response", ""),
                "tools_used": json.loads(result.payload.get("tools_used", "[]")),
                "timestamp": result.payload.get("timestamp", ""),
                "thread_id": result.payload.get("thread_id", "")
            })
        
        return formatted_results
    
    def get_thread_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all conversation turns for a specific thread in chronological order.
        
        Args:
            thread_id: Conversation thread identifier
        
        Returns:
            List of all conversation turns in order
        """
        # Use scroll to get all points for a thread
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,
            scroll_filter={
                "must": [
                    {
                        "key": "thread_id",
                        "match": {"value": thread_id}
                    }
                ]
            }
        )
        
        # Format and sort by turn number
        history = []
        for point in points:
            history.append({
                "turn_number": point.payload.get("turn_number", 0),
                "query": point.payload.get("user_query", ""),
                "response": point.payload.get("assistant_response", ""),
                "tools_used": json.loads(point.payload.get("tools_used", "[]")),
                "timestamp": point.payload.get("timestamp", "")
            })
        
        # Sort by turn number
        history.sort(key=lambda x: x["turn_number"])
        return history
    
    def get_turn_count(self, thread_id: str) -> int:
        """Get total number of turns in a conversation thread."""
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1,
            scroll_filter={
                "must": [
                    {
                        "key": "thread_id",
                        "match": {"value": thread_id}
                    }
                ]
            }
        )
        
        max_turn = 0
        for point in points:
            turn_num = point.payload.get("turn_number", 0)
            if turn_num > max_turn:
                max_turn = turn_num
        
        return max_turn
    
    def _get_turn_count(self, thread_id: str) -> int:
        """Internal helper to get turn count."""
        return self.get_turn_count(thread_id)
    
    def get_related_context(
        self,
        thread_id: str,
        current_query: str,
        limit: int = 3
    ) -> str:
        """
        Get related past conversation context for the current query.
        Useful for injecting relevant historical context into the LLM prompt.
        
        Args:
            thread_id: Current conversation thread
            current_query: Current user query
            limit: Number of past turns to retrieve
        
        Returns:
            Formatted string with related past context
        """
        # Search for similar past queries in same thread
        similar = self.search_conversations(current_query, thread_id=thread_id, limit=limit)
        
        if not similar:
            return ""
        
        context = "\n📚 Related Context from Past Discussions:\n"
        for i, item in enumerate(similar, 1):
            context += f"\n({i}) User asked: {item['query'][:100]}...\n"
            context += f"    Assistant responded: {item['response'][:100]}...\n"
        
        return context
    
    def export_thread(self, thread_id: str, format: str = "json") -> str:
        """
        Export a full conversation thread.
        
        Args:
            thread_id: Conversation thread to export
            format: Export format ('json' or 'markdown')
        
        Returns:
            Formatted conversation thread
        """
        history = self.get_thread_history(thread_id)
        
        if format == "json":
            return json.dumps({
                "thread_id": thread_id,
                "conversation": history,
                "exported_at": datetime.now().isoformat()
            }, indent=2)
        
        elif format == "markdown":
            md = f"# Conversation: {thread_id}\n\n"
            for turn in history:
                md += f"## Turn {turn['turn_number']}\n"
                md += f"**User:** {turn['query']}\n\n"
                md += f"**Assistant:** {turn['response']}\n\n"
                md += f"_Tools Used: {', '.join(turn['tools_used']) or 'None'}_\n\n"
                md += f"_Time: {turn['timestamp']}_\n\n---\n\n"
            return md
        
        return str(history)
    
    def clear_collection(self):
        """Delete all conversation history from the collection."""
        self.client.delete_collection(self.collection_name)
        print(f"✓ Cleared all data from collection '{self.collection_name}'")
        self._ensure_collection_exists()


# Example usage
if __name__ == "__main__":
    # Initialize memory manager
    memory = ConversationMemoryManager()
    
    # Simulate saving conversation turns
    thread_id = str(uuid.uuid4())
    print(f"\n--- Testing Conversation Memory (Thread: {thread_id[:8]}...) ---\n")
    
    # Turn 1
    memory.save_conversation_turn(
        thread_id=thread_id,
        user_query="What are Reliance's business segments?",
        assistant_response="Reliance operates in Oil, Hydrocarbon, Chemicals, and AI sectors.",
        tools_used=["search_graph_db"]
    )
    
    # Turn 2
    memory.save_conversation_turn(
        thread_id=thread_id,
        user_query="Tell me about their oil operations",
        assistant_response="Reliance's oil segment includes refining and production of various petroleum products.",
        tools_used=["search_vector_db"]
    )
    
    # Search test
    print("\n--- Semantic Search Results ---\n")
    search_results = memory.search_conversations("Reliance petroleum products", thread_id=thread_id)
    for i, result in enumerate(search_results, 1):
        print(f"Result {i} (Score: {result['score']:.3f})")
        print(f"Query: {result['query']}")
        print(f"Response: {result['response'][:100]}...\n")
    
    # Get history
    print("\n--- Full Thread History ---\n")
    history = memory.get_thread_history(thread_id)
    for turn in history:
        print(f"Turn {turn['turn_number']}: {turn['query']}")
    
    # Export
    print("\n--- Exported as Markdown ---\n")
    exported = memory.export_thread(thread_id, format="markdown")
    print(exported[:300] + "...")
