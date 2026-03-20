import json
import os
from typing import List, Dict
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize Qdrant client (in-memory for now, can switch to persistent)
client = QdrantClient("http://localhost:6333")

# Load embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Chunking parameters
SIMILARITY_THRESHOLD = 0.5
MAX_CHUNK_SIZE = 1024  # Maximum tokens per chunk to avoid overly large chunks


def load_processed_data(filepath: str = "data/processed_data.json") -> List[Dict]:
    """Load the ingested financial data."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split by common sentence endings
    sentences = text.split('.')
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunking(text: str, similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[str]:
    """
    Perform semantic chunking by detecting topic shifts based on sentence similarity.
    Breaks chunks at low-similarity boundaries (indicating topic changes).
    """
    sentences = split_into_sentences(text)
    
    if len(sentences) <= 1:
        return [text]
    
    # Generate embeddings for all sentences
    embeddings = model.encode(sentences, convert_to_numpy=True)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_tokens = len(sentences[0].split())
    
    for i in range(1, len(sentences)):
        # Calculate cosine similarity between consecutive sentences
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-8
        )
        
        sentence_tokens = len(sentences[i].split())
        
        # Break if: low similarity (topic shift) OR chunk exceeds max size
        if (similarity < similarity_threshold and current_chunk) or (current_tokens + sentence_tokens > MAX_CHUNK_SIZE):
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentences[i]]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentences[i])
            current_tokens += sentence_tokens
    
    # Add remaining sentences
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Apply semantic chunking to all documents.
    Returns list of chunks with metadata.
    """
    chunked_data = []
    
    print("Performing semantic chunking on documents...")
    
    for doc in tqdm(documents):
        chunks = semantic_chunking(doc["text"])
        
        for chunk_idx, chunk_text in enumerate(chunks):
            chunked_data.append({
                "text": chunk_text,
                "company": doc.get("company", "unknown"),
                "source": doc.get("source", "unknown"),
                "date": doc.get("date", "unknown"),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "metadata": doc.get("metadata", {})
            })
    
    return chunked_data


def create_qdrant_collection(collection_name: str = "financial_chunks"):
    """Create a Qdrant collection for storing embeddings."""
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )
    print(f"Created Qdrant collection: {collection_name}")


def embed_and_store_chunks(chunks: List[Dict], collection_name: str = "financial_chunks"):
    """
    Generate embeddings for chunks and store in Qdrant.
    """
    print("Generating embeddings and storing in Qdrant...")
    
    points = []
    
    for idx, chunk in enumerate(tqdm(chunks)):
        # Generate embedding
        embedding = model.encode(chunk["text"])
        
        # Create unique ID
        point_id = idx + 1
        
        # Prepare payload (metadata)
        payload = {
            "text": chunk["text"],
            "company": chunk["company"],
            "source": chunk["source"],
            "date": chunk["date"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "metadata": json.dumps(chunk["metadata"])
        }
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )
        points.append(point)
    
    # Upsert points to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Stored {len(points)} chunks in Qdrant")


def save_chunked_data(chunks: List[Dict], filepath: str = "data/chunked_data.json"):
    """Save chunked data to JSON file for reference."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {filepath}")


def search_chunks(query: str, collection_name: str = "financial_chunks", limit: int = 5) -> List[Dict]:
    """
    Search for semantically similar chunks using Qdrant.
    """
    query_embedding = model.encode(query)
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=limit
    )
    
    matches = []
    for result in results:
        matches.append({
            "score": result.score,
            "text": result.payload["text"],
            "company": result.payload["company"],
            "source": result.payload["source"],
            "date": result.payload["date"],
            "metadata": json.loads(result.payload["metadata"])
        })
    
    return matches


def main():
    documents = load_processed_data()
    print(f"Loaded {len(documents)} documents")
    
    chunked_data = chunk_documents(documents)
    print(f"Created {len(chunked_data)} chunks")
    
    save_chunked_data(chunked_data)
    
    create_qdrant_collection()
    embed_and_store_chunks(chunked_data)
    
    print("\n--- Testing semantic search ---")
    test_query = "What is the revenue of Indian tech companies?"
    results = search_chunks(test_query)
    
    print(f"Top results for '{test_query}':")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['company']} ({result['source']})")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Text: {result['text'][:200]}...")


if __name__ == "__main__":
    main()