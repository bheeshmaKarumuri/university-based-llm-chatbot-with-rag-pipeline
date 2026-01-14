import json
import numpy as np
from vector_db import AmritaQdrantClient
from sentence_transformers import SentenceTransformer

def reset_and_setup():
    """Reset and setup the vector database without filters."""
    
    # Configuration
    QDRANT_CLOUD_URL = "https://d10bd071-e1e0-45d7-9d1d-86c01cdbe85e.europe-west3-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wqLRUZmcYzCVAP5ZlKwxi5EQ-OUagDAKp9GAoK-t9Lk"
    
    print("Resetting and setting up vector database...")
    
    # Initialize client
    client = AmritaQdrantClient(
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_API_KEY,
        collection_name="amrita_documents",
        timeout=120.0
    )
    
    try:
        # Force re-ingestion
        client.ingest_from_saved_files(output_dir="output")
        print("Successfully reset and set up vector database!")
        
        # Get collection info
        info = client.get_collection_info()
        print(f"Collection: {info['name']}")
        print(f"Points: {info['points_count']}")
        
        # Test search
        print("\nTesting basic search...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_query = "Amrita University programs"
        query_vector = embedding_model.encode(test_query)
        results = client.search(query_vector=query_vector, limit=3)
        print(f"Test search found {len(results)} results")
        
        if results:
            print("Sample result:")
            print(f"Content: {results[0]['content'][:150]}...")
            print("Setup successful!")
        else:
            print("No results found - there might be an issue with the data.")
        
        return True
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    reset_and_setup()