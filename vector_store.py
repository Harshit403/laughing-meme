#!/usr/bin/env python3
"""
Optimized Ultra-Efficient In-Memory Vector Search with Semantic Chunking + Z.AI GLM Answer Generation
Now powered by Jina-API embeddings (jina-embeddings-v3) instead of local Sentence-Transformers.
"""
import numpy as np
import hashlib
import re
import nltk
import asyncio
import aiohttp
import faiss
import os
import sys
import time
import pickle
import requests
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from collections import OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# ------------------------------------------------------------------
# 1.  NLTK bootstrap
# ------------------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ------------------------------------------------------------------
# 2.  CONFIG – Jina + Z.AI
# ------------------------------------------------------------------
JINA_API_KEY  = "jina_bf0e1a310b0845849f4f339569af6bef2MFTDUSxcGgO1U4xYEv9CD2O54eK"
JINA_API_URL  = "https://api.jina.ai/v1/embeddings"
JINA_MODEL    = "jina-embeddings-v3"
JINA_TASK     = "retrieval.query"
EMBEDDING_DIMENSION = 1024          # jina-embeddings-v3 output size
QUANTIZE_EMBEDDINGS = True          # or False if you don’t want quantization

Z_AI_API_KEY  = os.getenv("Z_AI_API_KEY") or "ebc1621fb09741cba9c82e06b9e94ed8.keDzS2msXG0BOvY2"
Z_AI_BASE_URL = "https://api.z.ai/api/paas/v4/"
Z_AI_MODEL    = "GLM-4.5-Flash"
SYSTEM_PROMPT = (
    "You are a highly professional and helpful assistant. "
    "Your primary goal is to provide accurate, concise, and professional responses based on the provided context. "
    "Respond directly and naturally as a human expert would, without mentioning phrases like 'Based on the context provided' "
    "or 'According to the given information'. If the context does not contain sufficient information to answer the user's query, "
    "you must state clearly: 'I cannot find the relevant information to answer your query. Please contact the admin for more information.' "
    "Do not attempt to fabricate an answer if the information is missing."
)

# ------------------------------------------------------------------
# 3.  Jina client (sync + async wrappers)
# ------------------------------------------------------------------
def jina_embed(texts: List[str]) -> np.ndarray:
    """Return **normalised** np.float32 embeddings for a list of strings."""
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {"model": JINA_MODEL, "task": JINA_TASK, "input": texts}
    resp = requests.post(JINA_API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Jina API error: {resp.text}")
    embs = [item["embedding"] for item in resp.json()["data"]]
    embs = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(embs)          # cosine similarity
    return embs

async def jina_embed_async(texts: List[str]) -> np.ndarray:
    """Async version of the above."""
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {"model": JINA_MODEL, "task": JINA_TASK, "input": texts}
    async with aiohttp.ClientSession() as session:
        async with session.post(JINA_API_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(await resp.text())
            data = await resp.json()
            embs = [item["embedding"] for item in data["data"]]
            embs = np.array(embs, dtype=np.float32)
            faiss.normalize_L2(embs)
            return embs

# ------------------------------------------------------------------
# 4.  Text chunking (unchanged)
# ------------------------------------------------------------------
def simple_sentence_tokenize(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, max_tokens: int = 256, overlap: int = 25) -> List[str]:
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = simple_sentence_tokenize(text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        words = sent.split()
        word_cnt = len(words)
        if current_len + word_cnt > max_tokens and current:
            chunks.append(" ".join(current))
            if overlap:
                overlap_words, overlap_sents = 0, []
                for s in reversed(current):
                    s_words = len(s.split())
                    if overlap_words + s_words <= overlap:
                        overlap_sents.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                current, current_len = overlap_sents, overlap_words
            else:
                current, current_len = [], 0
        current.append(sent)
        current_len += word_cnt
    if current:
        chunks.append(" ".join(current))
    return chunks

# ------------------------------------------------------------------
# 5.  LRU cache (thread-safe)
# ------------------------------------------------------------------
class LRUCache:
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

# ------------------------------------------------------------------
# 6.  Optimised FAISS store (IVF index, quantisation flag, etc.)
# ------------------------------------------------------------------
class OptimizedVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.nlist = min(100, max(1, dimension // 4))
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = min(10, self.nlist)
        self.chunks = []
        self.embeddings = None
        self.query_cache = LRUCache(1000)
        self.is_trained = False
        self.embedding_lock = threading.Lock()
    # ---------- add docs ----------
    def add_documents(self, documents: List[str]) -> None:
        all_chunks, meta = [], []
        with ThreadPoolExecutor(max_workers=4) as exe:
            chunk_results = list(exe.map(chunk_text, documents))
        for doc_idx, chunks in enumerate(chunk_results):
            doc_id = hashlib.md5(documents[doc_idx].encode()).hexdigest()
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta.append({"doc_id": doc_id, "chunk_index": idx, "content": chunk})
        print(f"Encoding {len(all_chunks)} chunks via Jina …")
        embeddings = []
        for i in range(0, len(all_chunks), 16):
            batch = all_chunks[i:i+16]
            embs = jina_embed(batch)
            embeddings.append(embs)
        embeddings = np.vstack(embeddings).astype(np.float32)
        if QUANTIZE_EMBEDDINGS:
            embeddings = (embeddings * 127).astype(np.int8).astype(np.float32) / 127
        self.embeddings = embeddings
        if not self.is_trained and len(embeddings) > self.nlist * 10:
            print("Training IVF index …")
            self.index.train(embeddings)
            self.is_trained = True
        if self.is_trained:
            self.index.add(embeddings)
        else:
            fallback = faiss.IndexFlatIP(self.dimension)
            fallback.add(embeddings)
            self.index = fallback
        self.chunks = meta
        print(f"Indexed {len(all_chunks)} chunks.")
    # ---------- search ----------
    @lru_cache(maxsize=1000)
    def _get_query_embedding(self, query: str) -> np.ndarray:
        cached = self.query_cache.get(query)
        if cached is not None:
            return cached
        emb = jina_embed([query])[0]
        self.query_cache.put(query, emb)
        return emb
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qemb = self._get_query_embedding(query).reshape(1, -1)
        keywords = set(re.findall(r'\b\w+\b', query.lower()))
        scores, idxs = self.index.search(qemb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            content = chunk["content"]
            content_words = set(re.findall(r'\b\w+\b', content.lower()))
            kw_score = len(keywords & content_words) / max(len(keywords), 1)
            hybrid = 0.7 * float(score) + 0.3 * kw_score
            results.append({**chunk,
                            "vector_similarity": float(score),
                            "keyword_score": kw_score,
                            "hybrid_score": hybrid})
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results

# ------------------------------------------------------------------
# 7.  Async helpers
# ------------------------------------------------------------------
async def async_search(vector_store: OptimizedVectorStore, query: str) -> List[Dict[str, Any]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, vector_store.search, query)
async def handle_concurrent_requests(vector_store: OptimizedVectorStore, queries: List[str]) -> List[List[Dict[str, Any]]]:
    tasks = [async_search(vector_store, q) for q in queries]
    return await asyncio.gather(*tasks)

# ------------------------------------------------------------------
# 8.  Z.AI GLM answer generation (unchanged except for prompt)
# ------------------------------------------------------------------
client = OpenAI(api_key=Z_AI_API_KEY, base_url=Z_AI_BASE_URL)
async def generate_answer(query: str, vector_store: OptimizedVectorStore, chat_history: List[Dict[str, str]] = None) -> str:
    results = vector_store.search(query, k=5)
    context = "\n\n".join([r["content"] for r in results]) if results else "No relevant information found."
    hist = ""
    if chat_history:
        hist = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in chat_history])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nChat History:\n{hist}\n\nQuestion: {query}"}
    ]
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(model=Z_AI_MODEL, messages=messages, max_tokens=500, temperature=0.2)
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {type(e).__name__}: {str(e)}"

# ------------------------------------------------------------------
# 9.  Utility runner (works with/without running loop)
# ------------------------------------------------------------------
def run_async_task(coro):
    try:
        loop = asyncio.get_running_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except RuntimeError:
        return asyncio.run(coro)

# ------------------------------------------------------------------
# 10.  Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    docs = [
        "Organic skincare for sensitive skin with aloe vera and chamomile…",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille…",
        "Cuidado de la piel orgánico para piel sensible con aloe vera…",
        "针对敏感肌专门设计的天然有机护肤产品…",
        "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています…"
    ]
    store = OptimizedVectorStore(EMBEDDING_DIMENSION)
    store.add_documents(docs)
    query = "aloe vera skincare"
    print(f"\nQuery: {query}")
    for rank, hit in enumerate(store.search(query, 5), 1):
        print(f"{rank}. {hit['content'][:80]}…  (score={hit['hybrid_score']:.4f})")
    print("\nAnswer from GLM:")
    print(run_async_task(generate_answer(query, store)))
