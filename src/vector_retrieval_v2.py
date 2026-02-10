"""
Vector Retrieval Module for REALM (Robust Version)
Implements semantic search using sentence-transformers and FAISS
"""

import os
import numpy as np
from typing import List, Dict, Optional

# Set Hugging Face mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Global faiss reference (import at module level)
try:
    import faiss
    FAISS_AVAILABLE = True
    print(f"✓ FAISS imported successfully")
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    print(f"⚠ FAISS not available, using fallback")


class VectorRetriever:
    """
    Semantic retrieval using sentence embeddings and FAISS.
    Replaces simple keyword matching with dense vector search.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda:2",
        index_path: str = None
    ):
        self.model_name = model_name
        self.device = device
        self.index_path = index_path
        
        self.model = None
        self.index = None
        self.documents = []
        self.dimension = None
        
        self._load_model()
        
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"✓ Model loaded. Dimension: {self.dimension}")
            
        except ImportError:
            print("✗ sentence-transformers not available, using fallback")
            self.model = None
            self.dimension = 768
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model = None
            self.dimension = 768
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors"""
        if self.model is None:
            # Fallback: random vectors for testing
            return np.random.randn(len(texts), self.dimension).astype('float32')
        
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to index.
        
        Args:
            documents: List of dicts with 'text' and optional metadata
        """
        if not documents:
            return
        
        self.documents.extend(documents)
        texts = [doc['text'] for doc in documents]
        
        # Encode documents
        embeddings = self.encode(texts)
        
        # Create or update FAISS index
        self._update_index(embeddings)
    
    def _update_index(self, embeddings: np.ndarray):
        """Update FAISS index with new embeddings"""
        if not FAISS_AVAILABLE:
            # Use simple numpy-based similarity
            self._update_simple_index(embeddings)
            return
        
        # Use FAISS
        try:
            if self.index is None:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)
                
                # Try to use GPU if available
                if self.device.startswith('cuda'):
                    try:
                        gpu_id = int(self.device.split(':')[1])
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
                        print(f"✓ FAISS index moved to GPU {gpu_id}")
                    except Exception as e:
                        print(f"⚠ Could not use GPU for FAISS: {e}")
            else:
                # Use GPU if already created
                pass
            
            # Add vectors (L2 normalize for cosine similarity)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            print(f"✓ Added {len(embeddings)} vectors to index (total: {self.index.ntotal})")
            
        except Exception as e:
            print(f"✗ FAISS error: {e}")
            # Fall back to simple index
            self._update_simple_index(embeddings)
    
    def _update_simple_index(self, embeddings: np.ndarray):
        """Update simple numpy-based index (fallback)"""
        if not hasattr(self, '_simple_index') or self._simple_index is None:
            self._simple_index = SimpleIndex(self.dimension)
        
        self._simple_index.add(embeddings)
        print(f"✓ Added {len(embeddings)} vectors to simple index")
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        state_vector: np.ndarray = None
    ) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            state_vector: Optional state vector for state-conditioned retrieval
            
        Returns:
            List of relevant documents with scores
        """
        if not self.documents:
            return []
        
        # State-conditioned query expansion
        if state_vector is not None:
            query = self._expand_query_with_state(query, state_vector)
        
        # Encode query
        query_embedding = self.encode([query])
        
        # Search
        if FAISS_AVAILABLE and self.index is not None:
            # Use FAISS
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        else:
            # Use simple index
            scores, indices = self._simple_index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def _expand_query_with_state(self, query: str, state_vector: np.ndarray) -> str:
        """Expand query based on current state (Motivated Retrieval)"""
        if len(state_vector) > 0:
            mood_val = state_vector[0]
            
            # Add state context to query
            if mood_val > 0.7:
                state_prefix = "positive mood"
            elif mood_val < 0.3:
                state_prefix = "concerned mood"
            else:
                state_prefix = "neutral mood"
            
            return f"[{state_prefix}] {query}"
        
        return query
    
    def save(self, filepath: str):
        """Save index and documents"""
        import json
        
        # Save documents
        with open(f"{filepath}_docs.json", 'w') as f:
            json.dump(self.documents, f)
        
        # Save FAISS index if available
        if FAISS_AVAILABLE and self.index is not None:
            try:
                faiss.write_index(self.index, f"{filepath}.faiss")
            except:
                pass
    
    def load(self, filepath: str):
        """Load index and documents"""
        import json
        
        # Load documents
        try:
            with open(f"{filepath}_docs.json", 'r') as f:
                self.documents = json.load(f)
        except:
            pass


class SimpleIndex:
    """Fallback index when FAISS is not available"""
    
    def __init__(self, dimension: int):
        self.vectors = None
        self.dimension = dimension
    
    def add(self, vectors: np.ndarray):
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
    
    def search(self, query: np.ndarray, k: int):
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([[]]), np.array([[]])
        
        # Compute cosine similarity
        similarities = np.dot(self.vectors, query.T).flatten()
        
        # Get top k
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        top_k_scores = similarities[top_k_idx]
        
        return top_k_scores.reshape(1, -1), top_k_idx.reshape(1, -1)
    
    @property
    def ntotal(self):
        return len(self.vectors) if self.vectors is not None else 0
