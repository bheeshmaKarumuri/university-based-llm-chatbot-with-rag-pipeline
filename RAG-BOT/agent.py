import json
import re
import numpy as np
from agno.agent import Agent
from agno.models.google import Gemini
from agent_instructions import enhancer_instructions, response_instructions
from vector_db import AmritaQdrantClient
from sentence_transformers import SentenceTransformer
import logging
import warnings
import os

# Suppress all warnings and logs
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging to suppress INFO and WARNING messages
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('vector_db').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('google_genai').setLevel(logging.ERROR)
logging.getLogger('agno').setLevel(logging.ERROR)



enhancer_instructions = """
You are an expert AI assistant for Amrita University's academic search system. 
Your job is to improve user queries to find better information.

Return a JSON response with this structure:

{
  "enhanced_query": "improved version with academic terms and synonyms",
  "intent": "admission|course|career|research|events|general",
  "key_terms": ["important", "search", "terms"],
  "confidence": 0.0-1.0
}

ENHANCEMENT RULES:
- Add synonyms and expand abbreviations (AI→Artificial Intelligence, CS→Computer Science)
- Make queries more comprehensive and academic
- Intent is just for information, not for filtering
- Key terms are the most important search concepts

Examples:
- "CS admission" → "Computer Science admission requirements eligibility"
- "MBA jobs" → "MBA placement career opportunities employment"
- "AI research" → "Artificial Intelligence Machine Learning research opportunities"

Return ONLY valid JSON, no additional text.
"""

response_instructions = """
You are an intelligent assistant for Amrita University. 
Provide accurate answers based on retrieved document chunks about the university.

Return a JSON response with this structure:

{
  "answer": "comprehensive answer based on retrieved chunks",
  "confidence": 0.0-1.0,
  "answer_type": "complete|partial|insufficient|not_found",
  "key_points": ["point 1", "point 2", "point 3"],
  "suggested_questions": ["related question 1", "related question 2"]
}

GUIDELINES:
- Use ONLY information from the provided chunks
- Be comprehensive but accurate
- If information is limited, say so
- Include specific details when available
- Suggest related questions that might be helpful
- Always be helpful and informative about Amrita University

Return ONLY valid JSON, no additional text.
"""


def extract_json_from_response(response_content):
    """Extract JSON from response content with better error handling."""
    if not response_content:
        return None
    
    # Try to find JSON in code blocks first
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{.*\}'
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # Try the entire content
    try:
        return json.loads(response_content.strip())
    except json.JSONDecodeError:
        return None

def search_vector_db(enhanced_query, qdrant_client, embedding_model):
    """Search vector database with enhanced query - no filters."""
    try:
        # Generate embedding
        query_vector = embedding_model.encode(enhanced_query)
        
        print(f"Searching: '{enhanced_query}'")
        
        # Perform search without any filters
        results = qdrant_client.search(
            query_vector=query_vector,
            limit=5
        )
        
        print(f"Found {len(results)} results")
        return results
        
    except Exception as e:
        print(f"Search failed: {e}")
        return []

def generate_response(user_query, search_results, response_agent):
    """Generate response using search results."""
    if not search_results:
        return {
            "answer": "I couldn't find relevant information for your query. The database might be empty or there could be a connection issue.",
            "confidence": 0.0,
            "answer_type": "not_found",
            "key_points": [],
            "suggested_questions": [
                "What programs does Amrita University offer?",
                "Tell me about Amrita University campuses",
                "What are the admission requirements?"
            ]
        }
    
    # Prepare context with search results
    context_chunks = []
    for result in search_results:
        context_chunks.append({
            "content": result["content"],
            "score": result["score"],
            "source": result["metadata"].get("source_url", "unknown")
        })
    
    context_prompt = f"""
User Query: {user_query}

Retrieved Information:
{json.dumps(context_chunks, indent=2)}

Based on this information, provide a comprehensive answer about Amrita University.
"""
    
    try:
        response = response_agent.run(context_prompt)
        parsed_response = extract_json_from_response(response.content)
        
        if parsed_response and isinstance(parsed_response, dict):
            # Ensure all required fields exist
            default_response = {
                "answer": "I found some information but couldn't parse it properly.",
                "confidence": 0.5,
                "answer_type": "partial",
                "key_points": [],
                "suggested_questions": []
            }
            default_response.update(parsed_response)
            return default_response
        else:
            # If JSON parsing fails, use the raw response
            return {
                "answer": response.content,
                "confidence": 0.7,
                "answer_type": "partial",
                "key_points": [],
                "suggested_questions": []
            }
    except Exception as e:
        print(f"Response generation failed: {e}")
        return {
            "answer": "I found relevant information but encountered an error while generating the response.",
            "confidence": 0.3,
            "answer_type": "insufficient",
            "key_points": [],
            "suggested_questions": []
        }

def process_query(user_query, enhancer_agent, response_agent, qdrant_client, embedding_model):
    """Process a single user query end-to-end."""
    
    # Step 1: Enhance query (but don't use filters)
    enhanced_query = user_query
    intent = "general"
    
    try:
        enhancement_response = enhancer_agent.run(user_query)
        enhancement_data = extract_json_from_response(enhancement_response.content)
        
        if enhancement_data and isinstance(enhancement_data, dict):
            enhanced_query = enhancement_data.get("enhanced_query", user_query)
            intent = enhancement_data.get("intent", "general")
            print(f"Enhanced: {enhanced_query}")
            print(f"Intent: {intent}")
        else:
            print("Using original query (enhancement failed)")
        
    except Exception as e:
        print(f"Query enhancement failed, using original: {e}")
    
    # Step 2: Search vector database (no filters)
    search_results = search_vector_db(
        enhanced_query, 
        qdrant_client, 
        embedding_model
    )
    
    # Step 3: Generate response
    final_response = generate_response(user_query, search_results, response_agent)
    
    return {
        "query": user_query,
        "enhanced_query": enhanced_query,
        "intent": intent,
        "results_count": len(search_results),
        "response": final_response
    }

def initialize_components():
    """Initialize all components with error handling."""
    # Configuration
    QDRANT_CLOUD_URL = "https://d10bd071-e1e0-45d7-9d1d-86c01cdbe85e.europe-west3-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wqLRUZmcYzCVAP5ZlKwxi5EQ-OUagDAKp9GAoK-t9Lk"
    
    print("Initializing components...")
    
    try:
        # Initialize Qdrant client
        qdrant_client = AmritaQdrantClient(
            url=QDRANT_CLOUD_URL,
            api_key=QDRANT_API_KEY,
            collection_name="amrita_documents",
            timeout=120.0
        )
        
        # Check collection info
        try:
            info = qdrant_client.get_collection_info()
            print(f"Connected to collection: {info['name']} ({info['points_count']} documents)")
        except Exception as e:
            print(f"Warning: Could not get collection info: {e}")
            print("You may need to run setup first.")
        
        # Initialize embedding model
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize agents
        print("Initializing AI agents...")
        enhancer_agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key="AIzaSyBWlCMq91iKL0_XhxQhPU2MlKAr67lXLiM"),
            instructions=enhancer_instructions,
        )
        
        response_agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key="AIzaSyBWlCMq91iKL0_XhxQhPU2MlKAr67lXLiM"),
            instructions=response_instructions
        )
        
        print("All components initialized successfully!")
        return qdrant_client, embedding_model, enhancer_agent, response_agent
        
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        raise

def main():
    """Main execution with simplified workflow."""
    
    try:
        qdrant_client, embedding_model, enhancer_agent, response_agent = initialize_components()
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Please check your configuration and try again.")
        return
    
    # Test queries
    test_queries = [
        "What are the upcoming events?",
        "Computer science admission requirements",
        "MBA placement statistics",
        "Research opportunities in AI",
        "Tell me about Amrita University"
    ]
    
    print("\n" + "="*60)
    print("TESTING RAG SYSTEM")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        try:
            result = process_query(
                query, enhancer_agent, response_agent, 
                qdrant_client, embedding_model
            )
            
            if result:
                print(f"Answer: {result['response']['answer']}")
                # print(f"Confidence: {result['response']['confidence']}")
                # print(f"Results found: {result['results_count']}")
                
                # # Show some content snippets if available
                # if result['results_count'] > 0:
                #     print("Sample content found:")
                #     search_results = search_vector_db(result['enhanced_query'], qdrant_client, embedding_model)
                #     for i, res in enumerate(search_results[:2], 1):
                #         print(f"  {i}. {res['content'][:100]}...")
                
            else:
                print("Sorry, I couldn't process your query.")
        
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("-" * 40)

def interactive_mode():
    """Interactive mode for real-time testing."""
    
    try:
        qdrant_client, embedding_model, enhancer_agent, response_agent = initialize_components()
    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Please run setup first: python setup_vector_db.py")
        return
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE - Amrita University Assistant")
    print("="*60)
    print("Type 'quit' to exit, 'help' for sample questions.\n")
    
    sample_questions = [
        "What programs does Amrita University offer?",
        "Tell me about computer science admission",
        "What are the campus facilities?",
        "Research opportunities in engineering",
        "MBA program details"
    ]
    
    while True:
        user_query = input("Your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_query.lower() == 'help':
            print("\nSample questions you can ask:")
            for i, q in enumerate(sample_questions, 1):
                print(f"{i}. {q}")
            print()
            continue
        
        if not user_query:
            continue
        
        try:
            result = process_query(
                user_query, enhancer_agent, response_agent,
                qdrant_client, embedding_model
            )
            
            if result:
                print(f"\nAnswer: {result['response']['answer']}")
                if result['response']['key_points']:
                    print(f"\nKey Points:")
                    for point in result['response']['key_points']:
                        print(f"• {point}")
                if result['response']['suggested_questions']:
                    print(f"\nRelated Questions:")
                    for question in result['response']['suggested_questions']:
                        print(f"• {question}")
                print(f"\nConfidence: {result['response']['confidence']}")
                print(f"Results found: {result['results_count']}")
            else:
                print("\nSorry, I couldn't process your query.")
        
        except Exception as e:
            print(f"\nError occurred: {e}")
        
        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()