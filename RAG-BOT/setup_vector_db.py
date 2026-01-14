import json
import numpy as np
from vector_db import AmritaQdrantClient
from sentence_transformers import SentenceTransformer

def setup_vector_database():
    """Setup and initialize the vector database with proper indexes."""
    
    # Configuration
    QDRANT_CLOUD_URL = "https://d10bd071-e1e0-45d7-9d1d-86c01cdbe85e.europe-west3-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wqLRUZmcYzCVAP5ZlKwxi5EQ-OUagDAKp9GAoK-t9Lk"
    
    print("Setting up vector database...")
    
    # Initialize client
    client = AmritaQdrantClient(
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_API_KEY,
        collection_name="amrita_documents",
        timeout=120.0
    )
    
    # Check if we have processed data
    try:
        # Try to ingest from saved files
        client.ingest_from_saved_files(output_dir="output")
        print("Successfully set up vector database!")
        
        # Get collection info
        info = client.get_collection_info()
        print(f"Collection: {info['name']}")
        print(f"Points: {info['points_count']}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"Required files not found: {e}")
        print("Please run data processing first: python data.py")
        return False
    except Exception as e:
        print(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    setup_vector_database()