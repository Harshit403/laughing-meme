#!/usr/bin/env python3
"""
Optimized Ultra-Efficient In-Memory Vector Search with Semantic Chunking + Z.AI GLM Answer Generation
Optimizations:
1. Quantized embeddings for reduced memory usage
2. Asynchronous processing for concurrent requests
3. Improved hybrid search with better keyword scoring
4. Dynamic batching for embedding generation
5. Optimized FAISS index for faster search
6. Memory-efficient chunking with overlap optimization
7. Smart caching with LRU eviction
8. CPU-optimized processing
9. Z.AI GLM integration for human-readable answers
Required packages:
pip install numpy nltk transformers sentence-transformers torch faiss-cpu aiohttp openai fastapi uvicorn python-multipart
Usage:
python3 main.py
"""
import numpy as np
import hashlib
import re
import nltk
from sentence_transformers import SentenceTransformer, util
import torch
import faiss
from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import asyncio
import aiohttp
from functools import lru_cache
from collections import OrderedDict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
# Import the OpenAI class for the new SDK
from openai import OpenAI
# Download NLTK data only if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
# Configuration - Optimized for CPU performance and multilingual support
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual model
EMBEDDING_DIMENSION = 384  # Dimension for the multilingual model
CHUNK_MAX_TOKENS = 256
CHUNK_OVERLAP = 25
TOP_K_RESULTS = 5
MAX_CACHE_SIZE = 1000  # LRU cache size
BATCH_SIZE = 16  # Reduced batch size for better CPU performance
N_WORKERS = min(4, os.cpu_count() or 1)  # Worker threads
USE_GPU = False  # Force CPU usage
QUANTIZE_EMBEDDINGS = True  # Reduce memory usage
# --- Z.AI GLM Configuration ---
# IMPORTANT: Replace 'your-Z.AI-api-key' with your actual Z.AI API key
Z_AI_API_KEY = os.getenv("Z_AI_API_KEY") or "ebc1621fb09741cba9c82e06b9e94ed8.keDzS2msXG0BOvY2"
Z_AI_BASE_URL = "https://api.z.ai/api/paas/v4/"  # Ensure correct URL format
Z_AI_MODEL = "GLM-4.5-Flash"
# Global system prompt for the AI
SYSTEM_PROMPT = (
    "You are a highly professional and helpful assistant. "
    "Your primary goal is to provide accurate, concise, and professional responses based on the provided context. "
    "Respond directly and naturally as a human expert would, without mentioning phrases like 'Based on the context provided' "
    "or 'According to the given information'. If the context does not contain sufficient information to answer the user's query, "
    "you must state clearly: 'I cannot find the relevant information to answer your query. Please contact the admin for more information.' "
    "Do not attempt to fabricate an answer if the information is missing."
)
# Initialize the Z.AI client
client = OpenAI(
    api_key=Z_AI_API_KEY,
    base_url=Z_AI_BASE_URL
)
# Initialize the embedding model with optimizations
print("Loading multilingual SentenceTransformer model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded on CPU")
def simple_sentence_tokenize(text: str) -> List[str]:
    """Optimized sentence tokenizer"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
def chunk_text(text: str, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Memory-efficient text chunking with overlap optimization"""
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        sentences = simple_sentence_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_length + word_count > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Optimized overlap handling
            if overlap > 0:
                overlap_words = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_words + s_words <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_words
            else:
                current_chunk = []
                current_length = 0
        current_chunk.append(sentence)
        current_length += word_count
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
class LRUCache:
    """Thread-safe LRU cache implementation"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()
    def get(self, key: str) -> Optional[np.ndarray]:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
    def put(self, key: str, value: np.ndarray) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
class OptimizedVectorStore:
    """High-performance vector store with optimizations"""
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Use IVF index for better performance
        self.nlist = min(100, max(1, dimension // 4))  # Store as instance variable
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = min(10, self.nlist)
        self.chunks = []
        self.embeddings = None
        self.query_cache = LRUCache(MAX_CACHE_SIZE)
        self.is_trained = False
        self.embedding_lock = threading.Lock()
    def add_documents(self, documents: List[str]) -> None:
        """Add documents with optimized processing"""
        all_chunks = []
        chunk_metadata = []
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            chunk_results = list(executor.map(chunk_text, documents))
        for doc_idx, chunks in enumerate(chunk_results):
            doc_id = hashlib.md5(documents[doc_idx].encode()).hexdigest()
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "content": chunk
                })
        # Generate embeddings in batches
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = []
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i+BATCH_SIZE]
            with self.embedding_lock:
                batch_embeddings = embedding_model.encode(
                    batch,
                    convert_to_tensor=True,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False
                )
            embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = util.normalize_embeddings(embeddings)
        # Quantize embeddings if enabled
        if QUANTIZE_EMBEDDINGS:
            embeddings = (embeddings * 127).to(torch.int8).to(torch.float32) / 127
        self.embeddings = embeddings.cpu().numpy()
        # Train and add to FAISS index
        if not self.is_trained and len(self.embeddings) > self.nlist * 10:  # Use instance variable
            print("Training FAISS index...")
            self.index.train(self.embeddings)
            self.is_trained = True
        if self.is_trained:
            self.index.add(self.embeddings)
        else:
            # Fallback to flat index
            fallback_index = faiss.IndexFlatIP(self.dimension)
            fallback_index.add(self.embeddings)
            self.index = fallback_index
        self.chunks = chunk_metadata
        print(f"Added {len(all_chunks)} chunks to vector store")
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get cached query embedding"""
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        query_embedding = util.normalize_embeddings(query_embedding.unsqueeze(0)).cpu().numpy()
        return query_embedding
    def search(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Optimized search with improved hybrid scoring"""
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        # Extract keywords with scoring
        keywords = re.findall(r'\b\w+\b', query.lower())
        keyword_set = set(keywords)
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                content = chunk["content"]
                # Enhanced keyword matching
                content_words = set(re.findall(r'\b\w+\b', content.lower()))
                keyword_matches = keyword_set & content_words
                keyword_score = len(keyword_matches) / max(len(keyword_set), 1)
                # Improved hybrid scoring
                hybrid_score = (0.7 * float(score)) + (0.3 * keyword_score)
                results.append({
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "content": content,
                    "vector_similarity": float(score),
                    "keyword_match": bool(keyword_matches),
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score
                })
        # Sort by hybrid score
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results
async def async_search(vector_store: OptimizedVectorStore, query: str) -> List[Dict[str, Any]]:
    """Async wrapper for search"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, vector_store.search, query)
async def handle_concurrent_requests(vector_store: OptimizedVectorStore, queries: List[str]) -> List[List[Dict[str, Any]]]:
    """Handle multiple concurrent search requests"""
    tasks = [async_search(vector_store, query) for query in queries]
    return await asyncio.gather(*tasks)
async def generate_answer(query: str, vector_store: OptimizedVectorStore, chat_history: List[Dict[str, str]] = None) -> str:
    """Generate a human-readable answer using Z.AI GLM."""
    # Step 1: Retrieve top-k relevant chunks
    results = vector_store.search(query, k=TOP_K_RESULTS)
    
    # Step 2: Prepare context from retrieved chunks
    if not results:
        context = "No relevant information found in the document store."
    else:
        context = "\n\n".join([res["content"] for res in results])
    
    # Step 3: Format chat history if provided
    formatted_history = ""
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_history += f"{role}: {content}\n"
    
    # Step 4: Construct messages for Z.AI GLM, including the global system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nChat History:\n{formatted_history}\n\nQuestion: {query}"}
    ]
    
    # Step 5: Call Z.AI GLM API asynchronously
    try:
        # Run the synchronous OpenAI call in a thread executor to keep it async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=Z_AI_MODEL,
                messages=messages,
                max_tokens=500, # Increased slightly for more detailed answers
                temperature=0.2 # Lower temperature for more deterministic/professional responses
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = f"Error generating answer: {type(e).__name__}: {str(e)}"
        print(error_msg) # Log the error for debugging
        return "An error occurred while generating the answer. Please try again or contact the admin for more information."
def run_async_task(coro):
    """Run an async task, handling both environments with and without a running event loop"""
    try:
        # Check if there's already an event loop running (Python 3.7+)
        loop = asyncio.get_running_loop()
        # If we get here, there's a running loop.
        # Schedule the coroutine and wait for its completion within the existing loop.
        # This is the correct way to run async code from sync code when a loop is running.
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        # Block until the future is done and return the result.
        return future.result()
    except RuntimeError:
        # No running event loop, so we can use asyncio.run()
        return asyncio.run(coro)