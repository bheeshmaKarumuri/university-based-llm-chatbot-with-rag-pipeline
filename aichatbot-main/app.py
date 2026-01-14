from flask import Flask, redirect, render_template, request, jsonify, session
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os
import requests
from werkzeug.utils import secure_filename
from sqlalchemy import or_
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import nltk
from dotenv import load_dotenv
import logging
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
import scipy.io.wavfile as wav
import tempfile
import soundfile as sf
import torchaudio
from pydub import AudioSegment
import sys
sys.path.append('../RAG-BOT')
from agent import initialize_components, process_query
from vector_db import AmritaQdrantClient

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize extensions
socketio = SocketIO(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Upload configuration
app.config['OUTPUTS_FOLDER'] = './outputs'
os.makedirs(app.config['OUTPUTS_FOLDER'], exist_ok=True)

# Pinecone configuration
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    # Hardcoded fallback if env var is missing
    PINECONE_API_KEY = 'pcsk_6fuRrj_8VXpa5xQhZJAZVkNybbyynLzYyR8mvoeHV6grepaw2Jbf5qZMSCFW6vHewn8hmM'
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = 'chatbot-index'
EMBED_DIM = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors
# Create index if it doesn't exist, or delete and recreate if dimension is wrong
index_names = pc.list_indexes().names()
if INDEX_NAME in index_names:
    desc = pc.describe_index(INDEX_NAME)
    if desc['dimension'] != EMBED_DIM:
        print(f"Deleting Pinecone index '{INDEX_NAME}' (wrong dimension {desc['dimension']}, expected {EMBED_DIM})...")
        pc.delete_index(INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
else:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# Groq API configuration (Assumed to be available)
try:
    from groq import Groq
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))  # Set GROQ_API_KEY environment variable
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)

# Add new model for RAG chats
class RAGChat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)

# Create tables
with app.app_context():
    db.create_all()

model = SentenceTransformer('all-MiniLM-L6-v2')
user_chunks = {}  # user_id -> [chunks]

# Download punkt for sentence splitting (only needs to run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load model once at startup
try:
    processor = AutoProcessor.from_pretrained("lamyer/Telugu-transcription", use_safetensors=True)
    model_speech = AutoModelForSpeechSeq2Seq.from_pretrained("lamyer/Telugu-transcription", use_safetensors=True)
    SPEECH_MODEL_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Telugu speech model could not be loaded: {e}")
    processor = None
    model_speech = None
    SPEECH_MODEL_AVAILABLE = False

# Load ASR pipeline once at startup, with fallback
ASR_MODEL_NAME = None
try:
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model="lamyer/Telugu-transcription",
        chunk_length_s=30,
        device=0 if torch.cuda.is_available() else -1
    )
    # Set to translate Telugu audio to English text
    asr_pipeline.model.config.forced_decoder_ids = asr_pipeline.tokenizer.get_decoder_prompt_ids(language="te", task="translate")
    ASR_PIPELINE_AVAILABLE = True
    ASR_MODEL_NAME = "lamyer/Telugu-transcription"
    print("DEBUG: Hugging Face ASR pipeline loaded successfully: lamyer/Telugu-transcription")
except Exception as e:
    print(f"[WARNING] Could not load lamyer/Telugu-transcription: {e}")
    try:
        asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device=0 if torch.cuda.is_available() else -1
        )
        # Set to translate Telugu audio to English text
        asr_pipeline.model.config.forced_decoder_ids = asr_pipeline.tokenizer.get_decoder_prompt_ids(language="te", task="translate")
        ASR_PIPELINE_AVAILABLE = True
        ASR_MODEL_NAME = "openai/whisper-small"
        print("DEBUG: Fallback ASR pipeline loaded successfully: openai/whisper-small")
    except Exception as e2:
        print(f"[WARNING] Could not load fallback ASR model: {e2}")
        asr_pipeline = None
        ASR_PIPELINE_AVAILABLE = False
        ASR_MODEL_NAME = None

# --- Amrita University RAG System globals ---
qdrant_client = None
embedding_model = None
enhancer_agent = None
response_agent = None

def initialize_rag_components():
    global qdrant_client, embedding_model, enhancer_agent, response_agent
    if not all([qdrant_client, embedding_model, enhancer_agent, response_agent]):
        try:
            qdrant_client, embedding_model, enhancer_agent, response_agent = initialize_components()
            print("[RAG] Amrita University RAG components initialized.")
        except Exception as e:
            print(f"[RAG] Failed to initialize components: {e}")

def generate_request(user_input):
    """Generates a response based on user input using Groq API."""
    if not client:
        return "Error: Groq client not initialized."
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "Respond to the input with relevant content in any appropriate format."},
                  {"role": "user", "content": user_input}],
        model="llama3-70b-8192"
    )
    return chat_completion.choices[0].message.content

def extract_pdf_text(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks by sentences, with overlap.
    chunk_size: target number of words per chunk
    overlap: number of words to overlap between chunks
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sent in sentences:
        words = sent.split()
        if current_length + len(words) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Start new chunk with overlap
            if overlap > 0:
                overlap_words = []
                count = 0
                # Add last 'overlap' words from current_chunk
                for w in reversed(current_chunk):
                    overlap_words.insert(0, w)
                    count += 1
                    if count >= overlap:
                        break
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
            else:
                current_chunk = words
                current_length = len(words)
        else:
            current_chunk.extend(words)
            current_length += len(words)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/index')
def main_page():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.json.get('username')
    email = request.json.get('email')
    password = request.json.get('password')
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered!"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken!"}), 400
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User created successfully!"}), 200

@app.route('/login', methods=['POST'])
def login():
    identifier = request.json.get('email')  # could be email or username
    password = request.json.get('password')
    user = User.query.filter(or_(User.email == identifier, User.username == identifier)).first()
    print(f"Login attempt: {identifier}, user found: {user is not None}")
    if user:
        print(f"Stored hash: {user.password}, password: {password}")
        print(f"Check: {bcrypt.check_password_hash(user.password, password)}")
    if user and bcrypt.check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({"message": "Login successful!"}), 200
    return jsonify({"error": "Invalid credentials!"}), 400

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect('/')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    print('DEBUG: upload_file called')
    if 'user_id' not in session:
        print('DEBUG: No user_id in session')
        logging.error('Upload attempt without login/session.')
        return jsonify({"error": "Unauthorized: Please log in before uploading files."}), 401
    if 'file' not in request.files:
        print('DEBUG: No file part in request.files')
        logging.error('No file part in request.')
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    print(f'DEBUG: file.filename={file.filename}')
    if file.filename == '':
        print('DEBUG: No selected file')
        logging.error('No selected file.')
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['OUTPUTS_FOLDER'], filename)
    print(f'DEBUG: Saving file to {filepath}')
    file.save(filepath)
    if filename.lower().endswith('.pdf'):
        try:
            print('DEBUG: Extracting PDF text')
            text = extract_pdf_text(filepath)
            print(f'DEBUG: Extracted text length: {len(text)}')
            chunks = chunk_text(text)
            print(f'DEBUG: Number of chunks: {len(chunks)}')
            user_id = str(session['user_id'])
            user_chunks[user_id] = chunks
            print('DEBUG: Deleting old vectors from Pinecone')
            namespace = "default"
            try:
                index.delete(ids=[f"{user_id}-{i}" for i in range(len(chunks))], namespace=namespace)
            except Exception as e:
                print(f'DEBUG: Exception during Pinecone delete (ignored if NotFound): {e}')
            print('DEBUG: Encoding chunks')
            embeddings = model.encode(chunks).astype('float32')
            print('DEBUG: Upserting to Pinecone')
            pinecone_vectors = [(f"{user_id}-{i}", emb.tolist(), {"chunk": chunk, "user_id": user_id}) for i, (emb, chunk) in enumerate(zip(embeddings, chunks))]
            index.upsert(vectors=pinecone_vectors, namespace=namespace)
            print('DEBUG: Upload and upsert successful')
        except Exception as e:
            print('DEBUG: Exception during PDF processing or Pinecone upsert:', e)
            logging.exception('Error during PDF processing or Pinecone upsert:')
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    print('DEBUG: File upload completed successfully')
    return jsonify({"message": f"File '{filename}' uploaded successfully!", "path": filepath}), 200

@app.route('/edit_message', methods=['PUT'])
def edit_message():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    message_id = data.get('message_id')
    new_message = data.get('new_message')
    if not message_id or not new_message:
        return jsonify({"error": "Invalid data provided."}), 400
    try:
        chat = Chat.query.filter_by(id=message_id, user_id=session['user_id']).first()
        if not chat:
            return jsonify({"error": "Message not found."}), 404
        db.session.delete(chat)
        db.session.commit()
        new_response = generate_request(new_message)
        new_chat = Chat(user_id=session['user_id'], prompt=new_message, response=new_response)
        db.session.add(new_chat)
        db.session.commit()
        return jsonify({"new_response": new_response}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to edit message: {str(e)}"}), 500

@app.route('/get_response', methods=['POST'])
def get_response():
    # This is for Groq chat only
    user_input = request.json.get('user_input')
    incognito = request.json.get('incognito', False)
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    try:
        response = generate_request(user_input)
        # Store in Chat table only
        chat = Chat(user_id=user_id, prompt=user_input, response=response)
        db.session.add(chat)
        db.session.commit()
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Failed to process request: {e}"}), 500

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    chats = Chat.query.filter_by(user_id=user_id).paginate(page=page, per_page=per_page)
    chat_history = [{"prompt": chat.prompt, "response": chat.response} for chat in chats.items]
    return jsonify({
        "chats": chat_history,
        "total_pages": chats.pages,
        "current_page": chats.page
    }), 200

@app.route('/clear_chat_history', methods=['DELETE'])
def clear_chat_history():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    try:
        Chat.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        return jsonify({"message": "Chat history cleared successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_chat_history', methods=['GET'])
def search_chat_history():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    query = request.args.get('query', '', type=str)
    try:
        if not query:
            chats = Chat.query.filter_by(user_id=user_id).all()
        else:
            chats = Chat.query.filter(
                Chat.user_id == user_id,
                (Chat.prompt.ilike(f"%{query}%") | Chat.response.ilike(f"%{query}%"))
            ).all()
        if not chats:
            message = "No matching chats found." if query else "No chats available."
            return jsonify({"chats": [], "message": message}), 200
        chat_history = [{"prompt": chat.prompt, "response": chat.response} for chat in chats]
        return jsonify({"chats": chat_history}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/delete_chat', methods=['DELETE'])
def delete_chat():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    chat_id = request.json.get('chat_id')
    try:
        chat = Chat.query.filter_by(id=chat_id, user_id=session['user_id']).first()
        if chat:
            db.session.delete(chat)
            db.session.commit()
            return jsonify({"message": "Chat deleted successfully!"}), 200
        else:
            return jsonify({"error": "Chat not found!"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_rag_chat_history', methods=['GET'])
def get_rag_chat_history():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    chats = RAGChat.query.filter_by(user_id=user_id).paginate(page=page, per_page=per_page)
    chat_history = [{"question": chat.question, "answer": chat.answer} for chat in chats.items]
    return jsonify({
        "chats": chat_history,
        "total_pages": chats.pages,
        "current_page": chats.page
    }), 200

@app.route('/ask', methods=['POST'])
def ask():
    # This is for RAG/document Q&A only
    question = request.json.get('question')
    user_id = str(session['user_id'])
    if user_id not in user_chunks:
        return jsonify({"error": "No document uploaded."}), 400
    # Embed the question
    q_emb = model.encode([question]).astype('float32')[0]
    # Query Pinecone
    namespace = "default"
    query_results = index.query(vector=q_emb.tolist(), top_k=3, include_metadata=True, namespace=namespace, filter={"user_id": {"$eq": user_id}})
    context = ' '.join([match['metadata']['chunk'] for match in query_results['matches']])
    print(f'DEBUG: Context sent to LLM: {context[:500]}...')  # Print first 500 chars for brevity
    answer = generate_request(f"Context: {context}\n\nQuestion: {question}")
    # Store in RAGChat table only
    rag_chat = RAGChat(user_id=user_id, question=question, answer=answer)
    db.session.add(rag_chat)
    db.session.commit()
    return jsonify({"answer": answer})

@app.route('/transcribe_telugu', methods=['POST'])
def transcribe_telugu():
    print("DEBUG: Entered /transcribe_telugu")
    if not ASR_PIPELINE_AVAILABLE:
        print("DEBUG: ASR pipeline not available")
        return jsonify({'error': 'Telugu ASR pipeline is not available on this system.'}), 503
    if 'audio' not in request.files:
        print("DEBUG: No audio in request.files")
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    # Save to persistent location
    audio_dir = os.path.join(app.config['OUTPUTS_FOLDER'], 'audio_uploads')
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, secure_filename(file.filename))
    file.save(audio_path)
    print(f"DEBUG: Saved audio file to {audio_path}")
    try:
        # Use pydub to load and (optionally) re-export as standard PCM WAV
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # duration in seconds
        print(f"DEBUG: Audio duration: {duration:.2f} seconds")
        # Re-export as standard PCM WAV for compatibility
        pcm_path = audio_path.rsplit('.', 1)[0] + '_pcm.wav'
        audio.export(pcm_path, format='wav')
        print(f"DEBUG: Exported PCM WAV to {pcm_path}")
        # Use Hugging Face pipeline for transcription
        result = asr_pipeline(pcm_path)
        text = result["text"]
        print(f"DEBUG: Transcription: {text}")
        return jsonify({'text': text, 'audio_path': pcm_path, 'duration': duration, 'model': ASR_MODEL_NAME})
    except Exception as e:
        print(f"DEBUG: Exception during ASR pipeline transcription: {e}")
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc(), 'audio_path': audio_path, 'model': ASR_MODEL_NAME}), 500

@app.route('/amrita_ask', methods=['POST'])
def amrita_ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    global qdrant_client, embedding_model, enhancer_agent, response_agent
    if not all([qdrant_client, embedding_model, enhancer_agent, response_agent]):
        try:
            qdrant_client, embedding_model, enhancer_agent, response_agent = initialize_components()
        except Exception as e:
            return jsonify({"error": f"Failed to initialize RAG system: {str(e)}"}), 500
    try:
        result = process_query(
            question,
            enhancer_agent,
            response_agent,
            qdrant_client,
            embedding_model
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

@app.route('/amrita_health', methods=['GET'])
def amrita_health():
    global qdrant_client, embedding_model, enhancer_agent, response_agent
    components_status = {
        "qdrant_client": qdrant_client is not None,
        "embedding_model": embedding_model is not None,
        "enhancer_agent": enhancer_agent is not None,
        "response_agent": response_agent is not None
    }
    if qdrant_client:
        try:
            info = qdrant_client.get_collection_info()
            components_status["database_connection"] = True
            components_status["documents_count"] = info.get("points_count", 0)
        except Exception as e:
            components_status["database_connection"] = False
            components_status["database_error"] = str(e)
    all_healthy = all(components_status.values())
    return jsonify({
        "status": "healthy" if all_healthy else "degraded",
        "message": "All systems operational" if all_healthy else "Some components may have issues",
        "components": components_status
    })

@app.route('/amrita_collection_info', methods=['GET'])
def amrita_collection_info():
    global qdrant_client
    if not qdrant_client:
        return jsonify({"error": "Database client not initialized"}), 503
    try:
        info = qdrant_client.get_collection_info()
        return jsonify({
            "collection_name": info.get("name", "unknown"),
            "points_count": info.get("points_count", 0),
            "status": "connected"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get collection info: {str(e)}"}), 500

@app.route('/amrita_sample_questions', methods=['GET'])
def amrita_sample_questions():
    return jsonify({
        "sample_questions": [
            "What programs does Amrita University offer?",
            "Tell me about computer science admission requirements",
            "What are the campus facilities available?",
            "Research opportunities in engineering",
            "MBA program details and placement statistics",
            "What are the upcoming events at Amrita University?",
            "Tell me about student life at Amrita",
            "What scholarships are available?",
            "How to apply for PhD programs?",
            "Campus locations and contact information"
        ]
    })

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({
        "error": str(e),
        "trace": traceback.format_exc()
    }), 500

if __name__ == '__main__':
    initialize_rag_components()
    socketio.run(app, debug=True)
