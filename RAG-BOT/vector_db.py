import json
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import os
import time

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, MatchValue, MatchAny, Range
    )
except ImportError:
    raise ImportError("Please install qdrant-client: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmritaQdrantClient:
    """Simplified Qdrant client that prioritizes finding results over strict filtering."""
    
    def __init__(self, url: str, api_key: str, collection_name: str = "amrita_documents", timeout: float = 60.0):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        self.collection_name = collection_name
        self.embedding_dim = None
        
        logger.info(f"Initialized Qdrant client for collection: {collection_name}")
        
        # Test connection
        try:
            collections = self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self, embedding_dimension: int, distance_metric: Distance = Distance.COSINE):
        """Create a new collection with the specified embedding dimension."""
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
                time.sleep(2)  # Wait for deletion to complete
            except Exception:
                pass  # Collection doesn't exist
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dimension,
                    distance=distance_metric
                )
            )
            
            self.embedding_dim = embedding_dimension
            logger.info(f"Created collection '{self.collection_name}' with dimension {embedding_dimension}")
            
            # Wait for collection to be ready
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def check_and_setup_collection(self, embedding_dim: int):
        """Check if collection exists, create if needed."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points")
            
        except Exception:
            logger.info(f"Creating new collection with dimension {embedding_dim}")
            self.create_collection(embedding_dim)

    def ingest_from_saved_files(self, output_dir: str = "output", batch_size: int = 50, max_retries: int = 3):
        """Ingest chunks and embeddings from saved files with minimal metadata."""
        # Load files
        chunks_file = os.path.join(output_dir, "processed_chunks.json")
        embeddings_file = os.path.join(output_dir, "embeddings.npy")
        embeddings_meta_file = os.path.join(output_dir, "embeddings_metadata.json")
        
        for file_path in [chunks_file, embeddings_file, embeddings_meta_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        with open(embeddings_meta_file, 'r', encoding='utf-8') as f:
            embedding_metadata = json.load(f)
        embedding_dim = embedding_metadata['embedding_dimension']
        
        # Check and setup collection
        self.check_and_setup_collection(embedding_dim)
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        embeddings = np.load(embeddings_file)
        logger.info(f"Loaded {len(chunks)} chunks and embeddings of shape {embeddings.shape}")

        # Progress Tracking
        progress_file = os.path.join(output_dir, "ingestion_progress.json")
        ingested_ids = set()
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    ingested_ids = set(json.load(f).get("ingested_ids", []))
                logger.info(f"Resuming ingestion. {len(ingested_ids)} points already uploaded.")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read progress file, starting from scratch. Error: {e}")
        
        # Prepare points with minimal payload - just content and source
        points_to_ingest = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if i in ingested_ids:
                continue
            
            # Minimal payload structure - just what's essential
            payload = {
                "content": chunk["content"],
                "source_url": chunk["metadata"].get("source_url", ""),
                "word_count": chunk["metadata"].get("word_count", 0)
            }
            
            points_to_ingest.append(PointStruct(id=i, vector=embedding.tolist(), payload=payload))
        
        if not points_to_ingest:
            logger.info("All points have already been ingested.")
            return

        # Upload in batches with retry logic
        logger.info(f"Uploading {len(points_to_ingest)} new points in batches of {batch_size}")
        total_batches = (len(points_to_ingest) + batch_size - 1) // batch_size
        
        for i in range(0, len(points_to_ingest), batch_size):
            batch = points_to_ingest[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            for attempt in range(max_retries):
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True
                    )
                    logger.info(f"Uploaded batch {batch_num}/{total_batches}")
                    
                    batch_ids = [p.id for p in batch]
                    ingested_ids.update(batch_ids)
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump({"ingested_ids": list(ingested_ids)}, f)
                    
                    break
                except Exception as e:
                    logger.warning(f"Failed to upload batch {batch_num} on attempt {attempt + 1}/{max_retries}. Error: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Permanently failed to upload batch {batch_num}. Skipping.")
                    else:
                        wait_time = 2 ** attempt
                        logger.info(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
        
        logger.info("Ingestion process complete.")
        collection_info = self.client.get_collection(self.collection_name)
        logger.info(f"Collection now contains {collection_info.points_count} points")

    def search(self, 
               query_vector: np.ndarray,
               limit: int = 10,
               **kwargs) -> List[Dict[str, Any]]:
        """
        Simple search without any filters - just find the most relevant content.
        """
        try:
            # Always do basic search without any filters
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results.points:
                formatted_result = {
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "metadata": {
                        "source_url": result.payload.get("source_url", ""),
                        "word_count": result.payload.get("word_count", 0)
                    }
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "config": collection_info.config.dict() if collection_info.config else None
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise


# Example usage and testing
def example_usage():
    """Test the vector database functionality."""
    QDRANT_CLOUD_URL = "https://d10bd071-e1e0-45d7-9d1d-86c01cdbe85e.europe-west3-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wqLRUZmcYzCVAP5ZlKwxi5EQ-OUagDAKp9GAoK-t9Lk"   
    
    client = AmritaQdrantClient(
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_API_KEY,
        collection_name="amrita_documents",
        timeout=120.0
    )
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_vector = embedding_model.encode("What are the upcoming events in Amrita University?")
    
    # Test basic search
    print("Testing basic search...")
    try:
        basic_results = client.search(query_vector=query_vector, limit=3)
        print(f"Found {len(basic_results)} results")
        for i, result in enumerate(basic_results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Content: {result['content'][:100]}...")
    except Exception as e:
        print(f"Basic search failed: {e}")


if __name__ == "__main__":
    example_usage()