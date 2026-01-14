import json
import re
import spacy
import nltk
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging
import os
import pickle
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessorWithEmbeddings:
    """
    Document processor with semantic chunking, metadata extraction, and embedding generation.
    """
    
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the processor with embedding capabilities."""
        
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Using NLTK fallback.")
            self.nlp = None
        
        # Load sentence transformer for embeddings
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.sentence_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Metadata mappings
        self.level_map = {
            "Undergraduate": [
                "undergraduate", "ug programmes", "b.tech", "b.sc", "b.a.", "b.com", 
                "bba", "bds", "bams", "mbbs", "b.c.a.", "b.des.", "bachelor", "4-year",
                "b. tech", "b. sc", "b. a.", "b. com", "b. des"
            ],
            "Postgraduate": [
                "postgraduate", "pg programmes", "m.tech", "m.sc", "m.a.", "mba", 
                "m.d.", "m.s.", "mds", "master", "2-year master", "post graduate",
                "m. tech", "m. sc", "m. a."
            ],
            "Doctoral": [
                "doctoral", "ph.d", "d.m.", "doctorate", "phd", "research degree",
                "ph. d"
            ],
            "Integrated": [
                "integrated degree", "5 year integrated", "dual degree", "5-year",
                "integrated programme", "integrated msc", "integrated m.sc"
            ],
            "Fellowship": ["fellowship", "post doctoral", "postdoc"],
            "Certificate": ["certificate", "diploma", "certification course"]
        }
        
        self.discipline_map = {
            "Engineering": [
                "engineering", "computer science", "mechanical", "electrical", 
                "electronics", "civil", "chemical", "aerospace", "robotics",
                "artificial intelligence", "machine learning", "data science",
                "cyber security", "information technology", "biotechnology"
            ],
            "Medicine": [
                "medicine", "medical", "mbbs", "m.d.", "clinical", "surgery",
                "pathology", "anatomy", "physiology", "pharmacology", "dentistry"
            ],
            "Business": [
                "business", "management", "mba", "finance", "marketing", "economics",
                "operations", "strategy", "entrepreneurship"
            ],
            "Arts": [
                "arts", "humanities", "literature", "history", "philosophy", 
                "languages", "sociology", "psychology", "mass communication"
            ],
            "Sciences": [
                "physics", "chemistry", "mathematics", "biology", "biotechnology",
                "life sciences", "physical sciences", "nanoscience"
            ],
            "Health Sciences": [
                "nursing", "pharmacy", "allied health", "health sciences",
                "ayurveda", "physiotherapy"
            ]
        }
        
        self.campuses = [
            'amaravati', 'amritapuri', 'bengaluru', 'chennai', 'coimbatore', 
            'faridabad', 'kochi', 'mysuru', 'nagercoil', 'haridwar'
        ]
        
        # Academic content indicators
        self.academic_keywords = [
            'curriculum', 'course', 'program', 'degree', 'admission', 'eligibility',
            'semester', 'syllabus', 'faculty', 'research', 'laboratory', 'project',
            'internship', 'placement', 'career', 'skill', 'learning', 'education',
            'academic', 'university', 'college', 'department', 'specialization'
        ]
        
        # For duplicate detection
        self.seen_chunks = set()

    def is_quality_content(self, text: str) -> bool:
        """Check if text is meaningful academic content."""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Minimum length check
        if len(text_clean) < 150:
            return False
        
        # Skip navigation/UI elements
        ui_patterns = [
            r'^(back|close|home|about|contact|apply|read more|submit|search|top)$',
            r'^(privacy policy|terms|conditions|jobs|events|news|blog)$',
            r'^\d+$',  # Just numbers
            r'^[^\w\s]+$',  # Just punctuation
            r'^(admissions|rankings|accreditation|explore)+$'
        ]
        
        for pattern in ui_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return False
        
        # Check for academic content
        academic_score = sum(1 for keyword in self.academic_keywords 
                           if keyword in text_lower)
        
        # Must have academic indicators or be substantial content
        return academic_score >= 2 or len(text_clean) > 400

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences using spaCy or NLTK."""
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents 
                        if len(sent.text.strip()) > 30]
        else:
            sentences = nltk.sent_tokenize(text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        return sentences

    def create_semantic_chunks(self, text: str, 
                             min_chunk_size: int = 400, 
                             max_chunk_size: int = 800) -> List[str]:
        """
        Create chunks with semantic boundaries and optimal size.
        """
        sentences = self.extract_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed max size and we have enough content
            if (current_length + sentence_length > max_chunk_size and 
                current_length >= min_chunk_size):
                
                chunk_text = ' '.join(current_chunk).strip()
                if self.is_quality_content(chunk_text):
                    chunks.append(chunk_text)
                
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk if it meets criteria
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if current_length >= min_chunk_size and self.is_quality_content(chunk_text):
                chunks.append(chunk_text)
        
        return chunks

    def extract_metadata(self, text: str, source_url: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from text."""
        metadata = {
            'source_url': source_url,
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(self.extract_sentences(text))
        }
        
        text_lower = text.lower()
        
        # Extract program level
        level_scores = {}
        for level, keywords in self.level_map.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                level_scores[level] = score
        
        if level_scores:
            metadata['level'] = max(level_scores, key=level_scores.get)
            metadata['level_confidence'] = level_scores[metadata['level']]
        
        # Extract discipline
        discipline_scores = {}
        for discipline, keywords in self.discipline_map.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                discipline_scores[discipline] = score
        
        if discipline_scores:
            metadata['discipline'] = max(discipline_scores, key=discipline_scores.get)
            metadata['discipline_confidence'] = discipline_scores[metadata['discipline']]
        
        # Extract campuses
        found_campuses = [campus.capitalize() for campus in self.campuses 
                         if campus in text_lower]
        if found_campuses:
            metadata['campus'] = sorted(list(set(found_campuses)))
        
        # Extract additional features
        features = {}
        if any(word in text_lower for word in ['admission', 'eligibility', 'requirement']):
            features['has_admission_info'] = True
        
        if any(word in text_lower for word in ['career', 'job', 'placement', 'opportunity']):
            features['has_career_info'] = True
        
        if any(word in text_lower for word in ['curriculum', 'syllabus', 'course', 'subject']):
            features['has_curriculum_info'] = True
        
        if any(word in text_lower for word in ['research', 'project', 'thesis', 'publication']):
            features['has_research_info'] = True
        
        if features:
            metadata['features'] = features
        
        return metadata

    def is_duplicate(self, text: str) -> bool:
        """Check if chunk is duplicate using content hash."""
        # Create hash of normalized text
        normalized_text = re.sub(r'\s+', ' ', text.strip().lower())
        text_hash = hash(normalized_text[:200])  # Use first 200 chars for hash
        
        if text_hash in self.seen_chunks:
            return True
        
        self.seen_chunks.add(text_hash)
        return False

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Create embeddings for all chunks and return updated chunks with embedding info.
        """
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Extract text content for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.sentence_model.encode(
                batch_texts,
                show_progress_bar=True,
                batch_size=batch_size
            )
            all_embeddings.append(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Combine all embeddings
        embeddings_matrix = np.vstack(all_embeddings)
        
        # Add embedding info to chunks
        updated_chunks = []
        for i, chunk in enumerate(chunks):
            updated_chunk = chunk.copy()
            updated_chunk['metadata']['embedding_model'] = self.sentence_model.get_sentence_embedding_dimension()
            updated_chunk['metadata']['embedding_index'] = i
            updated_chunks.append(updated_chunk)
        
        logger.info(f"Created embeddings matrix of shape: {embeddings_matrix.shape}")
        return updated_chunks, embeddings_matrix

    def process_document(self, raw_text: str, source_url: str) -> List[Dict[str, Any]]:
        """Process a document into chunks with metadata."""
        logger.info(f"Processing document from {source_url}")
        
        # Clean text
        cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(cleaned_text)
        
        # Process each chunk
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Skip duplicates
            if self.is_duplicate(chunk_text):
                continue
            
            # Extract metadata
            metadata = self.extract_metadata(chunk_text, source_url)
            metadata['chunk_id'] = f"{self._sanitize_url(source_url)}_{i}"
            metadata['created_at'] = datetime.now().isoformat()
            
            processed_chunks.append({
                "content": chunk_text,
                "metadata": metadata
            })
        
        logger.info(f"Created {len(processed_chunks)} unique chunks from {source_url}")
        return processed_chunks

    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL for use in IDs."""
        sanitized = re.sub(r'[^\w\-_.]', '_', url)
        return sanitized[:50]  # Limit length

    def save_chunks_and_embeddings(self, chunks: List[Dict[str, Any]], 
                                  embeddings: np.ndarray,
                                  output_dir: str = "output") -> Dict[str, str]:
        """
        Save chunks and embeddings to files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # File paths
        chunks_file = os.path.join(output_dir, "processed_chunks.json")
        embeddings_file = os.path.join(output_dir, "embeddings.npy")
        embeddings_meta_file = os.path.join(output_dir, "embeddings_metadata.json")
        
        # Save chunks as JSON
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Save embeddings as numpy array
        np.save(embeddings_file, embeddings)
        
        # Save embedding metadata
        embedding_metadata = {
            "model_name": str(self.sentence_model),
            "embedding_dimension": self.embedding_dim,
            "num_chunks": len(chunks),
            "embedding_shape": embeddings.shape,
            "created_at": datetime.now().isoformat()
        }
        
        with open(embeddings_meta_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_metadata, f, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")
        logger.info(f"Saved embeddings to {embeddings_file}")
        logger.info(f"Saved embedding metadata to {embeddings_meta_file}")
        
        return {
            "chunks_file": chunks_file,
            "embeddings_file": embeddings_file,
            "embeddings_meta_file": embeddings_meta_file
        }

    def load_chunks_and_embeddings(self, output_dir: str = "output") -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Load previously saved chunks and embeddings.
        """
        chunks_file = os.path.join(output_dir, "processed_chunks.json")
        embeddings_file = os.path.join(output_dir, "embeddings.npy")
        
        # Load chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Load embeddings
        embeddings = np.load(embeddings_file)
        
        logger.info(f"Loaded {len(chunks)} chunks and embeddings of shape {embeddings.shape}")
        return chunks, embeddings

    def get_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get comprehensive statistics about processed chunks."""
        if not chunks:
            return {}
        
        stats = {
            'total_chunks': len(chunks),
            'avg_char_count': sum(c['metadata']['char_count'] for c in chunks) / len(chunks),
            'avg_word_count': sum(c['metadata']['word_count'] for c in chunks) / len(chunks),
            'avg_sentence_count': sum(c['metadata']['sentence_count'] for c in chunks) / len(chunks),
        }
        
        # Count distributions
        levels = [c['metadata'].get('level') for c in chunks if c['metadata'].get('level')]
        disciplines = [c['metadata'].get('discipline') for c in chunks if c['metadata'].get('discipline')]
        
        if levels:
            stats['level_distribution'] = dict(Counter(levels))
        if disciplines:
            stats['discipline_distribution'] = dict(Counter(disciplines))
        
        # Campus distribution
        all_campuses = []
        for chunk in chunks:
            if 'campus' in chunk['metadata']:
                all_campuses.extend(chunk['metadata']['campus'])
        
        if all_campuses:
            stats['campus_distribution'] = dict(Counter(all_campuses))
        
        # Feature distribution
        feature_counts = Counter()
        for chunk in chunks:
            if 'features' in chunk['metadata']:
                for feature in chunk['metadata']['features']:
                    feature_counts[feature] += 1
        
        if feature_counts:
            stats['feature_distribution'] = dict(feature_counts)
        
        return stats

    def get_statistics_from_files(self, output_dir: str = "output") -> Dict[str, Any]:
        """Get comprehensive statistics about processed chunks from saved files."""
        try:
            # Load chunks from saved file
            chunks_file = os.path.join(output_dir, "processed_chunks.json")
            embeddings_meta_file = os.path.join(output_dir, "embeddings_metadata.json")
            
            if not os.path.exists(chunks_file):
                logger.error(f"Chunks file not found: {chunks_file}")
                return {}
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not chunks:
                return {}
            
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
            
            # Basic statistics
            stats = {
                'total_chunks': len(chunks),
                'avg_char_count': sum(c['metadata']['char_count'] for c in chunks) / len(chunks),
                'avg_word_count': sum(c['metadata']['word_count'] for c in chunks) / len(chunks),
                'avg_sentence_count': sum(c['metadata']['sentence_count'] for c in chunks) / len(chunks),
            }
            
            # Count distributions
            levels = [c['metadata'].get('level') for c in chunks if c['metadata'].get('level')]
            disciplines = [c['metadata'].get('discipline') for c in chunks if c['metadata'].get('discipline')]
            
            if levels:
                stats['level_distribution'] = dict(Counter(levels))
            if disciplines:
                stats['discipline_distribution'] = dict(Counter(disciplines))
            
            # Campus distribution
            all_campuses = []
            for chunk in chunks:
                if 'campus' in chunk['metadata']:
                    all_campuses.extend(chunk['metadata']['campus'])
            
            if all_campuses:
                stats['campus_distribution'] = dict(Counter(all_campuses))
            
            # Feature distribution
            feature_counts = Counter()
            for chunk in chunks:
                if 'features' in chunk['metadata']:
                    for feature in chunk['metadata']['features']:
                        feature_counts[feature] += 1
            
            if feature_counts:
                stats['feature_distribution'] = dict(feature_counts)
            
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics from files: {e}")
            return {}


def main():
    """Main processing function with embedding generation."""
    # Initialize processor
    processor = DocumentProcessorWithEmbeddings()
    
    # Configuration
    input_file = 'amrita.json'
    output_dir = 'output'
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{input_file}'.")
        return
    
    # Process all documents
    all_chunks = []
    
    print("Processing documents...")
    for i, (url, content) in enumerate(data.items(), 1):
        if isinstance(content, str):
            try:
                chunks = processor.process_document(content, url)
                all_chunks.extend(chunks)
                print(f"Processed {i}/{len(data)}: {len(chunks)} chunks from {url}")
            except Exception as e:
                print(f"Error processing {url}: {e}")
        else:
            print(f"Skipping {url}: content is not text")
    
    if not all_chunks:
        print("No chunks were created. Check your input data.")
        return
    
    # Create embeddings
    print("\nCreating embeddings...")
    chunks_with_embeddings, embeddings_matrix = processor.create_embeddings(all_chunks)
    
    # Save chunks and embeddings
    print("\nSaving results...")
    file_paths = processor.save_chunks_and_embeddings(
        chunks_with_embeddings, 
        embeddings_matrix, 
        output_dir
    )
    
    # Generate and display statistics
    stats = processor.get_statistics(chunks_with_embeddings)
    
    # Generate and display statistics from saved files
    stats_from_files = processor.get_statistics_from_files(output_dir)
    
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total chunks created: {stats.get('total_chunks', 0)}")
    print(f"Average characters per chunk: {stats.get('avg_char_count', 0):.0f}")
    print(f"Average words per chunk: {stats.get('avg_word_count', 0):.0f}")
    print(f"Average sentences per chunk: {stats.get('avg_sentence_count', 0):.0f}")
    print(f"Embedding dimension: {processor.embedding_dim}")
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    
    print(f"\nFiles saved:")
    for key, path in file_paths.items():
        print(f"  {key}: {path}")
    
    # Display in-memory statistics
    print(f"\n" + "="*40)
    print("IN-MEMORY STATISTICS")
    print("="*40)
    
    if 'level_distribution' in stats:
        print(f"\nLevel distribution:")
        for level, count in stats['level_distribution'].items():
            print(f"  {level}: {count}")
    
    if 'discipline_distribution' in stats:
        print(f"\nDiscipline distribution:")
        for discipline, count in stats['discipline_distribution'].items():
            print(f"  {discipline}: {count}")
    
    if 'campus_distribution' in stats:
        print(f"\nCampus distribution:")
        for campus, count in stats['campus_distribution'].items():
            print(f"  {campus}: {count}")
    
    if 'feature_distribution' in stats:
        print(f"\nContent features:")
        for feature, count in stats['feature_distribution'].items():
            feature_name = feature.replace('has_', '').replace('_', ' ').title()
            print(f"  {feature_name}: {count}")
    
    # Display file-based statistics
    print(f"\n" + "="*40)
    print("FILE-BASED STATISTICS")
    print("="*40)
    
    if stats_from_files:
        print(f"Total chunks in files: {stats_from_files.get('total_chunks', 0)}")
        print(f"Average characters per chunk: {stats_from_files.get('avg_char_count', 0):.0f}")
        print(f"Average words per chunk: {stats_from_files.get('avg_word_count', 0):.0f}")
        print(f"Average sentences per chunk: {stats_from_files.get('avg_sentence_count', 0):.0f}")
        
        if 'level_distribution' in stats_from_files:
            print(f"\nLevel distribution (from files):")
            for level, count in stats_from_files['level_distribution'].items():
                print(f"  {level}: {count}")
        
        if 'discipline_distribution' in stats_from_files:
            print(f"\nDiscipline distribution (from files):")
            for discipline, count in stats_from_files['discipline_distribution'].items():
                print(f"  {discipline}: {count}")
        
        if 'campus_distribution' in stats_from_files:
            print(f"\nCampus distribution (from files):")
            for campus, count in stats_from_files['campus_distribution'].items():
                print(f"  {campus}: {count}")
        
        if 'feature_distribution' in stats_from_files:
            print(f"\nContent features (from files):")
            for feature, count in stats_from_files['feature_distribution'].items():
                feature_name = feature.replace('has_', '').replace('_', ' ').title()
                print(f"  {feature_name}: {count}")
    else:
        print("No file-based statistics available.")
    
    print(f"\n" + "="*60)


if __name__ == "__main__":
    main()