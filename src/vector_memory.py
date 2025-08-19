# Vector Memory - Lightweight Embedding System for Semantic Search
# Local storage and retrieval for grounding AI agents with context

import os
import json
import logging
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Database
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class MemoryType(Enum):
    DOCUMENT = "document"
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    CONTEXT = "context"
    FACTUAL = "factual"

class EmbeddingModel(Enum):
    SENTENCE_BERT = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_BERT_LARGE = "sentence-transformers/all-mpnet-base-v2"
    OPENAI_ADA = "text-embedding-ada-002"
    CUSTOM = "custom"

# Database Models
class VectorMemory(Base):
    __tablename__ = "vector_memories"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Content identification
    content_hash = Column(String, unique=True, index=True)
    content = Column(Text)
    title = Column(String)
    source = Column(String)
    
    # Vector embedding
    embedding = Column(LargeBinary)  # Stored as pickled numpy array
    embedding_model = Column(String)
    embedding_dim = Column(Integer)
    
    # Metadata
    memory_type = Column(String)
    tags = Column(JSON)
    metadata = Column(JSON)
    
    # Retrieval information
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    relevance_score = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

@dataclass
class MemoryItem:
    """Individual memory item for vector storage"""
    content: str
    title: str = ""
    source: str = "unknown"
    memory_type: MemoryType = MemoryType.KNOWLEDGE
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    expires_hours: Optional[int] = None

@dataclass
class SearchResult:
    """Search result with similarity score"""
    content: str
    title: str
    source: str
    similarity_score: float
    memory_type: str
    tags: List[str]
    metadata: Dict[str, Any]
    access_count: int
    last_accessed: Optional[datetime]

@dataclass
class VectorConfig:
    """Configuration for vector memory system"""
    embedding_model: EmbeddingModel = EmbeddingModel.SENTENCE_BERT
    embedding_dim: int = 384
    max_memories: int = 10000
    similarity_threshold: float = 0.5
    cache_embeddings: bool = True
    use_faiss_index: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50

class VectorMemorySystem:
    """Lightweight vector memory system for semantic search and context grounding"""
    
    def __init__(self, db_session: Session, config: VectorConfig = None):
        self.db = db_session
        self.config = config or VectorConfig()
        
        # Initialize embedding model
        self.embedding_model = None
        self.tokenizer = None
        self._setup_embedding_model()
        
        # Initialize FAISS index if available
        self.faiss_index = None
        self.memory_id_map = {}  # Maps FAISS index position to memory ID
        self._setup_faiss_index()
        
        # Cache for frequently accessed embeddings
        self.embedding_cache = {}
        self.cache_timestamps = {}
    
    def _setup_embedding_model(self):
        """Setup embedding model based on configuration"""
        
        try:
            if self.config.embedding_model == EmbeddingModel.SENTENCE_BERT:
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.config.embedding_dim = 384
                    logger.info("Initialized Sentence-BERT model (MiniLM)")
                else:
                    logger.warning("SentenceTransformers not available, falling back to simple embeddings")
                    self._setup_simple_embeddings()
            
            elif self.config.embedding_model == EmbeddingModel.SENTENCE_BERT_LARGE:
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
                    self.config.embedding_dim = 768
                    logger.info("Initialized Sentence-BERT model (MPNet)")
                else:
                    logger.warning("SentenceTransformers not available, falling back to simple embeddings")
                    self._setup_simple_embeddings()
            
            else:
                self._setup_simple_embeddings()
                
        except Exception as e:
            logger.error(f"Failed to setup embedding model: {e}")
            self._setup_simple_embeddings()
    
    def _setup_simple_embeddings(self):
        """Setup simple TF-IDF style embeddings as fallback"""
        
        self.embedding_model = "simple"
        self.config.embedding_dim = 300
        self.vocab = {}
        self.idf_scores = {}
        logger.info("Initialized simple embedding model")
    
    def _setup_faiss_index(self):
        """Setup FAISS index for fast similarity search"""
        
        if FAISS_AVAILABLE and self.config.use_faiss_index:
            try:
                # Create FAISS index for cosine similarity
                self.faiss_index = faiss.IndexFlatIP(self.config.embedding_dim)
                logger.info(f"Initialized FAISS index with dimension {self.config.embedding_dim}")
                
                # Load existing memories into FAISS
                self._rebuild_faiss_index()
                
            except Exception as e:
                logger.error(f"Failed to setup FAISS index: {e}")
                self.faiss_index = None
        else:
            logger.info("FAISS not available or disabled, using simple similarity search")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from database"""
        
        if not self.faiss_index:
            return
        
        try:
            memories = self.db.query(VectorMemory).all()
            
            if memories:
                embeddings = []
                memory_ids = []
                
                for memory in memories:
                    try:
                        embedding = pickle.loads(memory.embedding)
                        embeddings.append(embedding)
                        memory_ids.append(memory.id)
                    except Exception as e:
                        logger.warning(f"Failed to load embedding for memory {memory.id}: {e}")
                        continue
                
                if embeddings:
                    embeddings_array = np.vstack(embeddings).astype('float32')
                    
                    # Normalize for cosine similarity
                    faiss.normalize_L2(embeddings_array)
                    
                    # Add to index
                    self.faiss_index.add(embeddings_array)
                    
                    # Update mapping
                    self.memory_id_map = {i: memory_id for i, memory_id in enumerate(memory_ids)}
                    
                    logger.info(f"Rebuilt FAISS index with {len(embeddings)} memories")
        
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text"""
        
        try:
            # Check cache first
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if self.config.cache_embeddings and text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
            
            # Compute embedding based on model type
            if self.embedding_model == "simple":
                embedding = self._compute_simple_embedding(text)
            elif hasattr(self.embedding_model, 'encode'):
                # Sentence transformers
                embedding = self.embedding_model.encode([text])[0]
            else:
                # Fallback to simple embedding
                embedding = self._compute_simple_embedding(text)
            
            # Cache embedding
            if self.config.cache_embeddings:
                self.embedding_cache[text_hash] = embedding
                self.cache_timestamps[text_hash] = datetime.utcnow()
                
                # Clean old cache entries
                self._clean_embedding_cache()
            
            return embedding.astype('float32')
            
        except Exception as e:
            logger.error(f"Failed to compute embedding: {e}")
            # Return random embedding as fallback
            return np.random.randn(self.config.embedding_dim).astype('float32')
    
    def _compute_simple_embedding(self, text: str) -> np.ndarray:
        """Compute simple TF-IDF style embedding"""
        
        # Tokenize text
        tokens = text.lower().split()
        
        # Simple word frequency
        word_freq = defaultdict(int)
        for token in tokens:
            word_freq[token] += 1
        
        # Create embedding vector
        embedding = np.zeros(self.config.embedding_dim)
        
        for i, (word, freq) in enumerate(word_freq.items()):
            if i >= self.config.embedding_dim:
                break
            # Simple hash-based position + frequency
            pos = hash(word) % self.config.embedding_dim
            embedding[pos] += freq
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _clean_embedding_cache(self):
        """Clean old cache entries"""
        
        try:
            current_time = datetime.utcnow()
            cache_ttl = timedelta(hours=24)
            
            expired_keys = [
                key for key, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > cache_ttl
            ]
            
            for key in expired_keys:
                self.embedding_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
                
        except Exception as e:
            logger.error(f"Failed to clean embedding cache: {e}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        
        try:
            # Simple word-based chunking
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
                chunk_words = words[i:i + self.config.chunk_size]
                chunk = ' '.join(chunk_words)
                if chunk.strip():
                    chunks.append(chunk)
            
            return chunks if chunks else [text]
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            return [text]
    
    def add_memory(self, memory_item: MemoryItem) -> List[int]:
        """Add memory item to vector storage"""
        
        try:
            # Chunk large content
            chunks = self._chunk_text(memory_item.content)
            memory_ids = []
            
            for i, chunk in enumerate(chunks):
                # Create content hash
                content_hash = hashlib.sha256(
                    f"{chunk}{memory_item.source}{memory_item.title}".encode()
                ).hexdigest()
                
                # Check if already exists
                existing = self.db.query(VectorMemory).filter(
                    VectorMemory.content_hash == content_hash
                ).first()
                
                if existing:
                    logger.info(f"Memory already exists: {content_hash[:8]}")
                    memory_ids.append(existing.id)
                    continue
                
                # Compute embedding
                embedding = self._compute_embedding(chunk)
                
                # Calculate expiration
                expires_at = None
                if memory_item.expires_hours:
                    expires_at = datetime.utcnow() + timedelta(hours=memory_item.expires_hours)
                
                # Create memory record
                title = memory_item.title
                if len(chunks) > 1:
                    title = f"{memory_item.title} (Part {i+1})"
                
                memory = VectorMemory(
                    content_hash=content_hash,
                    content=chunk,
                    title=title,
                    source=memory_item.source,
                    embedding=pickle.dumps(embedding),
                    embedding_model=self.config.embedding_model.value,
                    embedding_dim=self.config.embedding_dim,
                    memory_type=memory_item.memory_type.value,
                    tags=memory_item.tags or [],
                    metadata=memory_item.metadata or {},
                    expires_at=expires_at
                )
                
                self.db.add(memory)
                self.db.commit()
                
                memory_ids.append(memory.id)
                
                # Add to FAISS index
                if self.faiss_index:
                    try:
                        # Normalize for cosine similarity
                        normalized_embedding = embedding.copy()
                        faiss.normalize_L2(normalized_embedding.reshape(1, -1))
                        
                        # Add to index
                        self.faiss_index.add(normalized_embedding.reshape(1, -1))
                        
                        # Update mapping
                        next_index = len(self.memory_id_map)
                        self.memory_id_map[next_index] = memory.id
                        
                    except Exception as e:
                        logger.error(f"Failed to add to FAISS index: {e}")
                
                logger.info(f"Added memory: {title} (ID: {memory.id})")
            
            # Clean up old memories if over limit
            self._cleanup_old_memories()
            
            return memory_ids
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            self.db.rollback()
            return []
    
    def search_memories(self, 
                       query: str,
                       limit: int = 10,
                       memory_type: Optional[MemoryType] = None,
                       tags: Optional[List[str]] = None,
                       source: Optional[str] = None,
                       min_similarity: float = None) -> List[SearchResult]:
        """Search for similar memories"""
        
        try:
            # Compute query embedding
            query_embedding = self._compute_embedding(query)
            min_similarity = min_similarity or self.config.similarity_threshold
            
            results = []
            
            if self.faiss_index and len(self.memory_id_map) > 0:
                # Use FAISS for fast search
                results = self._search_with_faiss(
                    query_embedding, limit * 2, min_similarity, 
                    memory_type, tags, source
                )
            else:
                # Use database search
                results = self._search_with_database(
                    query_embedding, limit * 2, min_similarity,
                    memory_type, tags, source
                )
            
            # Sort by similarity and apply limit
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            results = results[:limit]
            
            # Update access statistics
            for result in results:
                self._update_access_stats(result)
            
            logger.info(f"Found {len(results)} memories for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def _search_with_faiss(self, 
                          query_embedding: np.ndarray,
                          limit: int,
                          min_similarity: float,
                          memory_type: Optional[MemoryType],
                          tags: Optional[List[str]],
                          source: Optional[str]) -> List[SearchResult]:
        """Search using FAISS index"""
        
        try:
            # Normalize query embedding
            normalized_query = query_embedding.copy()
            faiss.normalize_L2(normalized_query.reshape(1, -1))
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(
                normalized_query.reshape(1, -1), 
                min(limit, len(self.memory_id_map))
            )
            
            results = []
            
            for similarity, index in zip(similarities[0], indices[0]):
                if index == -1:  # No more results
                    break
                
                if similarity < min_similarity:
                    continue
                
                memory_id = self.memory_id_map.get(index)
                if not memory_id:
                    continue
                
                # Get memory from database
                memory = self.db.query(VectorMemory).filter(
                    VectorMemory.id == memory_id
                ).first()
                
                if not memory:
                    continue
                
                # Apply filters
                if memory_type and memory.memory_type != memory_type.value:
                    continue
                
                if tags and not any(tag in memory.tags for tag in tags):
                    continue
                
                if source and memory.source != source:
                    continue
                
                # Check expiration
                if memory.expires_at and datetime.utcnow() > memory.expires_at:
                    continue
                
                results.append(SearchResult(
                    content=memory.content,
                    title=memory.title,
                    source=memory.source,
                    similarity_score=float(similarity),
                    memory_type=memory.memory_type,
                    tags=memory.tags,
                    metadata=memory.metadata,
                    access_count=memory.access_count,
                    last_accessed=memory.last_accessed
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _search_with_database(self,
                             query_embedding: np.ndarray,
                             limit: int,
                             min_similarity: float,
                             memory_type: Optional[MemoryType],
                             tags: Optional[List[str]],
                             source: Optional[str]) -> List[SearchResult]:
        """Search using database similarity computation"""
        
        try:
            # Build query filters
            query = self.db.query(VectorMemory)
            
            if memory_type:
                query = query.filter(VectorMemory.memory_type == memory_type.value)
            
            if source:
                query = query.filter(VectorMemory.source == source)
            
            # Filter expired memories
            query = query.filter(
                (VectorMemory.expires_at.is_(None)) |
                (VectorMemory.expires_at > datetime.utcnow())
            )
            
            memories = query.all()
            
            results = []
            
            for memory in memories:
                try:
                    # Load embedding
                    memory_embedding = pickle.loads(memory.embedding)
                    
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, memory_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                    )
                    
                    if similarity < min_similarity:
                        continue
                    
                    # Apply tag filter
                    if tags and not any(tag in memory.tags for tag in tags):
                        continue
                    
                    results.append(SearchResult(
                        content=memory.content,
                        title=memory.title,
                        source=memory.source,
                        similarity_score=float(similarity),
                        memory_type=memory.memory_type,
                        tags=memory.tags,
                        metadata=memory.metadata,
                        access_count=memory.access_count,
                        last_accessed=memory.last_accessed
                    ))
                    
                except Exception as e:
                    logger.warning(f"Failed to process memory {memory.id}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return []
    
    def _update_access_stats(self, result: SearchResult):
        """Update access statistics for memory"""
        
        try:
            memory = self.db.query(VectorMemory).filter(
                VectorMemory.content == result.content,
                VectorMemory.title == result.title
            ).first()
            
            if memory:
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
                memory.relevance_score = (memory.relevance_score + result.similarity_score) / 2
                self.db.commit()
                
        except Exception as e:
            logger.error(f"Failed to update access stats: {e}")
    
    def get_context_for_query(self, 
                             query: str,
                             max_context_length: int = 2000,
                             memory_types: List[MemoryType] = None) -> str:
        """Get relevant context for a query"""
        
        try:
            # Search for relevant memories
            memory_types = memory_types or [MemoryType.KNOWLEDGE, MemoryType.FACTUAL, MemoryType.CONTEXT]
            
            all_results = []
            for memory_type in memory_types:
                results = self.search_memories(
                    query=query,
                    limit=5,
                    memory_type=memory_type
                )
                all_results.extend(results)
            
            # Sort by relevance
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Build context string
            context_parts = []
            current_length = 0
            
            for result in all_results:
                content = f"[{result.source}] {result.title}: {result.content}"
                
                if current_length + len(content) > max_context_length:
                    break
                
                context_parts.append(content)
                current_length += len(content) + 2  # +2 for newlines
            
            context = "\n\n".join(context_parts)
            
            logger.info(f"Generated context of {len(context)} characters from {len(context_parts)} memories")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return ""
    
    def _cleanup_old_memories(self):
        """Clean up old or expired memories"""
        
        try:
            # Remove expired memories
            expired_count = self.db.query(VectorMemory).filter(
                VectorMemory.expires_at.isnot(None),
                VectorMemory.expires_at < datetime.utcnow()
            ).delete()
            
            # Check if over memory limit
            total_count = self.db.query(VectorMemory).count()
            
            if total_count > self.config.max_memories:
                # Remove oldest, least accessed memories
                excess_count = total_count - self.config.max_memories
                
                old_memories = self.db.query(VectorMemory).order_by(
                    VectorMemory.access_count.asc(),
                    VectorMemory.last_accessed.asc(),
                    VectorMemory.created_at.asc()
                ).limit(excess_count).all()
                
                for memory in old_memories:
                    self.db.delete(memory)
                
                logger.info(f"Removed {excess_count} old memories")
            
            if expired_count > 0:
                logger.info(f"Removed {expired_count} expired memories")
            
            self.db.commit()
            
            # Rebuild FAISS index if memories were removed
            if expired_count > 0 or total_count > self.config.max_memories:
                self._rebuild_faiss_index()
                
        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}")
            self.db.rollback()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        
        try:
            total_memories = self.db.query(VectorMemory).count()
            
            # Count by type
            type_counts = {}
            for memory_type in MemoryType:
                count = self.db.query(VectorMemory).filter(
                    VectorMemory.memory_type == memory_type.value
                ).count()
                type_counts[memory_type.value] = count
            
            # Count by source
            source_counts = {}
            sources = self.db.query(VectorMemory.source).distinct().all()
            for (source,) in sources:
                count = self.db.query(VectorMemory).filter(
                    VectorMemory.source == source
                ).count()
                source_counts[source] = count
            
            # Most accessed memories
            top_memories = self.db.query(VectorMemory).order_by(
                VectorMemory.access_count.desc()
            ).limit(5).all()
            
            top_accessed = [
                {
                    "title": memory.title,
                    "access_count": memory.access_count,
                    "relevance_score": memory.relevance_score
                }
                for memory in top_memories
            ]
            
            return {
                "total_memories": total_memories,
                "by_type": type_counts,
                "by_source": source_counts,
                "top_accessed": top_accessed,
                "faiss_enabled": self.faiss_index is not None,
                "embedding_model": self.config.embedding_model.value,
                "embedding_dimension": self.config.embedding_dim,
                "cache_size": len(self.embedding_cache),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

# Helper functions for common memory operations
def create_document_memory(title: str, content: str, source: str = "document", tags: List[str] = None) -> MemoryItem:
    """Create memory item for document content"""
    return MemoryItem(
        content=content,
        title=title,
        source=source,
        memory_type=MemoryType.DOCUMENT,
        tags=tags or [],
        metadata={"document_type": "text"}
    )

def create_conversation_memory(conversation: str, participants: List[str], source: str = "chat") -> MemoryItem:
    """Create memory item for conversation"""
    return MemoryItem(
        content=conversation,
        title=f"Conversation with {', '.join(participants)}",
        source=source,
        memory_type=MemoryType.CONVERSATION,
        tags=participants,
        metadata={"participants": participants, "conversation_type": "chat"}
    )

def create_knowledge_memory(fact: str, category: str, source: str = "knowledge") -> MemoryItem:
    """Create memory item for factual knowledge"""
    return MemoryItem(
        content=fact,
        title=f"Knowledge: {category}",
        source=source,
        memory_type=MemoryType.FACTUAL,
        tags=[category],
        metadata={"category": category, "fact_type": "general"}
    )

# Background job for memory maintenance
def run_memory_maintenance(db_session: Session, config: VectorConfig = None):
    """Run memory system maintenance tasks"""
    
    logger.info("Starting memory maintenance")
    
    try:
        memory_system = VectorMemorySystem(db_session, config)
        
        # Clean up old memories
        memory_system._cleanup_old_memories()
        
        # Clean embedding cache
        memory_system._clean_embedding_cache()
        
        # Rebuild FAISS index if needed
        if memory_system.faiss_index:
            memory_system._rebuild_faiss_index()
        
        logger.info("Memory maintenance completed successfully")
        
    except Exception as e:
        logger.error(f"Memory maintenance failed: {e}")
        raise

# Example usage
if __name__ == "__main__":
    print("Vector Memory System initialized for semantic search and context grounding!")