import os
from typing import List, Dict, Optional, Any
from mem0 import Memory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConversationMemory:
    """
    A conversation memory system using mem0 with Neo4j graph store.
    Provides functionality to add, search, and delete conversation memories.
    """
    
    def __init__(self):
        """Initialize the memory system with Neo4j configuration."""
        # Get API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in .env file or environment variables.")
        
        self.config = {
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
                    "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
                    "password": os.environ.get("NEO4J_PASSWORD", "12345678"),
                    "database": "neo4j",
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": openai_api_key,
                    "model": "text-embedding-3-small",
                    "embedding_dims": 1536
                }
            }
        }
        self.memory = Memory.from_config(self.config)
    
    def add_memory(
        self, 
        conversation: List[Dict[str, str]], 
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        
        try:
            result = self.memory.add(
                conversation, 
                user_id=user_id,
                metadata=metadata
            )
            print(f"✓ Memory added successfully for user: {user_id}")
            return result
        except Exception as e:
            print(f"✗ Error adding memory: {str(e)}")
            raise
    
    def search_memory(
        self, 
        query: str, 
        user_id: str,
        limit: int = 5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        try:
            results = self.memory.search(
                query,
                user_id=user_id,
                limit=limit,
                rerank=rerank
            )
            
            print(f"\n🔍 Found {len(results.get('results', []))} results for: '{query}'")
            return results.get("results", [])
        except Exception as e:
            print(f"✗ Error searching memory: {str(e)}")
            raise
    
    def delete_memory(
        self, 
        memory_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a specific memory by ID.
        
        Args:
            memory_id: The ID of the memory to delete
            user_id: Optional user ID for verification
            
        Returns:
            Result from the delete operation
            
        Example:
            >>> memory.delete_memory(memory_id="mem-123", user_id="user-123")
        """
        try:
            result = self.memory.delete(memory_id=memory_id)
            print(f"✓ Memory deleted successfully: {memory_id}")
            return result
        except Exception as e:
            print(f"✗ Error deleting memory: {str(e)}")
            raise
    
    def delete_all_memories(self, user_id: str) -> Dict[str, Any]:
        try:
            result = self.memory.delete_all(user_id=user_id)
            print(f"✓ All memories deleted for user: {user_id}")
            return result
        except Exception as e:
            print(f"✗ Error deleting all memories: {str(e)}")
            raise
    
    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:

        try:
            memories = self.memory.get_all(user_id=user_id)
            print(f" Retrieved {len(memories)} memories for user: {user_id}")
            return memories
        except Exception as e:
            print(f"✗ Error retrieving memories: {str(e)}")
            raise
    
    def update_memory(
        self, 
        memory_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            result = self.memory.update(memory_id=memory_id, data=data)
            print(f"✓ Memory updated successfully: {memory_id}")
            return result
        except Exception as e:
            print(f"✗ Error updating memory: {str(e)}")
            raise


def demo_usage():
    
    memory = ConversationMemory()
    
    # Example 1: Add a conversation
    print("\n1. Adding a conversation to memory...")
    conversation1 = [
        {"role": "user", "content": "Alice met Bob at GraphConf 2025 in San Francisco."},
        {"role": "assistant", "content": "Great! I've logged that connection."},
    ]
    memory.add_memory(conversation1, user_id="demo-user")
    
    # Example 2: Add another conversation
    print("\n2. Adding another conversation...")
    conversation2 = [
        {"role": "user", "content": "Bob is working on a graph database project."},
        {"role": "assistant", "content": "Interesting! I'll remember that about Bob."},
    ]
    memory.add_memory(conversation2, user_id="demo-user")
    
    # Example 3: Search for memories
    print("\n3. Searching for memories...")
    results = memory.search_memory(
        "Who did Alice meet at GraphConf?",
        user_id="demo-user",
        limit=3,
        rerank=True
    )
    
    print("\nSearch Results:")
    for idx, hit in enumerate(results, 1):
        print(f"{idx}. {hit.get('memory', 'N/A')}")
        print(f"   Score: {hit.get('score', 'N/A')}")
        print(f"   ID: {hit.get('id', 'N/A')}\n")
    
    # Example 4: Get all memories
    print("\n4. Getting all memories for user...")
    all_memories = memory.get_all_memories(user_id="demo-user")
    print(f"Total memories: {len(all_memories)}")


if __name__ == "__main__":
    # Run the demo
    demo_usage()
