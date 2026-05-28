import os
import re
import math
import yaml
import logging
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Retriever")

class BM25Retriever:
    """
    Inverted-index accelerated BM25 retriever for high-speed lexical search.
    """
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_lens = []
        self.idf = {}
        self.inverted_index = {}
        self._build_index(corpus)
        
    def _build_index(self, corpus):
        logger.info(f"Building BM25 index on corpus of size {self.corpus_size}...")
        total_len = 0
        doc_counts = {}
        
        for doc_id, doc in enumerate(corpus):
            if not isinstance(doc, str):
                doc = str(doc)
            # Find lowercase tokens
            tokens = re.findall(r'\w+', doc.lower())
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            
            tf_counts = {}
            for token in tokens:
                tf_counts[token] = tf_counts.get(token, 0) + 1
                
            for token, tf in tf_counts.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((doc_id, tf))
                doc_counts[token] = doc_counts.get(token, 0) + 1
                
        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculate IDF for terms
        for token, count in doc_counts.items():
            self.idf[token] = math.log((self.corpus_size - count + 0.5) / (count + 0.5) + 1.0)
            
    def get_top_k_ranks(self, query, top_k):
        query_tokens = re.findall(r'\w+', query.lower())
        scores = {}
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            postings = self.inverted_index.get(token, [])
            for doc_id, tf in postings:
                doc_len = self.doc_lens[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = idf * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0.0) + score
                
        if not scores:
            return list(range(min(top_k, self.corpus_size))), [0.0] * min(top_k, self.corpus_size)
            
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k]
        
        doc_ids = [d[0] for d in top_docs]
        doc_scores = [d[1] for d in top_docs]
        
        # Fallback if too few documents matched the lexical terms
        if len(doc_ids) < top_k:
            all_ids = set(range(self.corpus_size))
            remaining = list(all_ids - set(doc_ids))[:top_k - len(doc_ids)]
            doc_ids.extend(remaining)
            doc_scores.extend([0.0] * len(remaining))
            
        return doc_ids, doc_scores


class DenseRetriever:
    """
    GPU-accelerated SentenceTransformers dense retriever with caching support.
    """
    def __init__(self, corpus, model_name, cache_folder, device="cuda", cache_path=None, batch_size=256):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Robustly resolve local model paths if they exist
        resolved_model_path = model_name
        if os.path.exists(model_name):
            resolved_model_path = os.path.abspath(model_name)
        else:
            # Check relative to cache_folder
            potential_path = os.path.join(cache_folder, model_name) if cache_folder else ""
            if potential_path and os.path.exists(potential_path):
                resolved_model_path = os.path.abspath(potential_path)
            else:
                # Check relative to project root
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                potential_project_path = os.path.join(project_root, model_name)
                if os.path.exists(potential_project_path):
                    resolved_model_path = os.path.abspath(potential_project_path)
                    
        # Enforce that the model path MUST exist locally as a directory to prevent automatic downloading
        if not os.path.isdir(resolved_model_path):
            raise FileNotFoundError(
                f"Local model path '{resolved_model_path}' does not exist or is not a directory. "
                f"Automatic downloading is disabled. Only pre-downloaded weights are allowed."
            )
            
        logger.info(f"Loading SentenceTransformer model from '{resolved_model_path}' on {self.device}...")
        self.model = SentenceTransformer(resolved_model_path, cache_folder=cache_folder, device=self.device, local_files_only=True)
        self.corpus_size = len(corpus)
        self.batch_size = batch_size
        self.embeddings = None
        self._load_or_compute_embeddings(corpus, cache_path)
        
    def _load_or_compute_embeddings(self, corpus, cache_path):
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading precomputed corpus embeddings from {cache_path}...")
            try:
                emb_arr = np.load(cache_path)
                if len(emb_arr) == self.corpus_size:
                    logger.info("Successfully loaded matching embeddings from cache.")
                    self.embeddings = torch.tensor(emb_arr, device=self.device)
                    # Normalize immediately to prepare for dot product cosine similarity
                    self.embeddings = self.embeddings / torch.norm(self.embeddings, dim=1, keepdim=True)
                    return
                else:
                    logger.warning("Cached embeddings size mismatch. Recomputing...")
            except Exception as e:
                logger.error(f"Failed to load cached embeddings: {e}")
        
        logger.info(f"Computing embeddings for corpus of size {self.corpus_size}...")
        embeddings_list = []
        for i in tqdm(range(0, self.corpus_size, self.batch_size), desc="Encoding corpus"):
            batch = corpus[i : i + self.batch_size]
            batch = [str(x) for x in batch]
            batch_emb = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            embeddings_list.append(batch_emb)
        
        self.embeddings = torch.cat(embeddings_list, dim=0)
        
        if cache_path:
            logger.info(f"Saving computed embeddings to cache: {cache_path}...")
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, self.embeddings.cpu().numpy())
            except Exception as e:
                logger.error(f"Failed to save embeddings to cache: {e}")
                
        # Normalize corpus embeddings
        self.embeddings = self.embeddings / torch.norm(self.embeddings, dim=1, keepdim=True)

    def get_dense_scores(self, query):
        query_emb = self.model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        query_emb = query_emb / torch.norm(query_emb)
        scores = torch.mv(self.embeddings, query_emb)
        return scores.cpu().numpy()


class HybridRetriever:
    """
    Combines dense and sparse retrievals using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, corpus, dense_retriever, bm25_retriever, rrf_k=60, top_m=1000):
        self.corpus = corpus
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.top_m = top_m
        
    def retrieve(self, query, top_k):
        # 1. Get sparse top-M candidates
        sparse_doc_ids, _ = self.bm25_retriever.get_top_k_ranks(query, self.top_m)
        sparse_ranks = {doc_id: rank for rank, doc_id in enumerate(sparse_doc_ids)}
        
        # 2. Get dense top-M candidates
        dense_scores = self.dense_retriever.get_dense_scores(query)
        dense_top_indices = np.argpartition(dense_scores, -self.top_m)[-self.top_m:]
        dense_top_scores = dense_scores[dense_top_indices]
        sorted_dense_indices = dense_top_indices[np.argsort(-dense_top_scores)]
        dense_ranks = {doc_id: rank for rank, doc_id in enumerate(sorted_dense_indices)}
        
        # 3. Union of candidate document IDs
        candidates = set(sparse_ranks.keys()) | set(dense_ranks.keys())
        
        # 4. Compute RRF scores
        rrf_scores = []
        for doc_id in candidates:
            s_rank = sparse_ranks.get(doc_id, self.top_m)
            d_rank = dense_ranks.get(doc_id, self.top_m)
            
            score = 1.0 / (self.rrf_k + s_rank) + 1.0 / (self.rrf_k + d_rank)
            rrf_scores.append((doc_id, score))
            
        # 5. Sort and take top K
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_results = rrf_scores[:top_k]
        
        return [res[0] for res in top_k_results]


def main():
    # Determine the project root directory (parent of the retrieval script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load configuration
    config_path = os.path.join(project_root, "configs", "retrieval.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    question_types = config.get("question_types", [])
    datasets = config.get("datasets", [])
    retrieval_settings = config.get("retrieval_settings", {})
    
    method = retrieval_settings.get("method", "hybrid")
    top_k = retrieval_settings.get("top_k", 5)
    dense_model = retrieval_settings.get("dense_model", "BAAI/bge-small-en-v1.5")
    models_dir = retrieval_settings.get("models_dir", "models")
    rrf_k = retrieval_settings.get("rrf_k", 60)
    batch_size = retrieval_settings.get("batch_size", 256)
    
    max_corpus_docs = config.get("max_corpus_docs", 0)
    max_questions = config.get("max_questions", 0)
    
    # Setup local models path
    abs_models_dir = os.path.abspath(os.path.join(project_root, models_dir))
    os.makedirs(abs_models_dir, exist_ok=True)
    
    for dataset in datasets:
        logger.info(f"==================================================")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"==================================================")
        
        corpus_path = os.path.join(project_root, "data", "processed", f"{dataset}.parquet")
        if not os.path.exists(corpus_path):
            logger.error(f"Corpus processed parquet not found at: {corpus_path}. Please run prepare_datasets.py first.")
            continue
            
        df_corpus = pd.read_parquet(corpus_path)
        
        # Apply debugging limit on corpus size if specified
        if max_corpus_docs and max_corpus_docs > 0:
            logger.info(f"Applying debug limit: loading first {max_corpus_docs} corpus documents.")
            df_corpus = df_corpus.head(max_corpus_docs)
            
        corpus_ids = df_corpus["id"].tolist()
        text_column = "chat" if "chat" in df_corpus.columns else "content"
        corpus_texts = df_corpus[text_column].tolist()
        
        # Setup retrievers
        bm25_retriever = None
        dense_retriever = None
        hybrid_retriever = None
        
        if method in ("sparse", "hybrid"):
            bm25_retriever = BM25Retriever(corpus_texts)
            
        if method in ("dense", "hybrid"):
            # Normalize model name for cache filename
            model_name_safe = dense_model.replace("/", "_").replace("\\", "_").replace(":", "_")
            cache_filename = f"{dataset}_{model_name_safe}_embeddings.npy"
            if max_corpus_docs and max_corpus_docs > 0:
                cache_filename = f"{dataset}_{model_name_safe}_embeddings_limit_{max_corpus_docs}.npy"
                
            cache_path = os.path.join(project_root, "data", "processed", "embeddings", cache_filename)
            
            dense_retriever = DenseRetriever(
                corpus=corpus_texts,
                model_name=dense_model,
                cache_folder=abs_models_dir,
                device="cuda",
                cache_path=cache_path,
                batch_size=batch_size
            )
            
        if method == "hybrid":
            hybrid_retriever = HybridRetriever(
                corpus=corpus_texts,
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
                rrf_k=rrf_k,
                top_m=1000
            )
            
        for q_type in question_types:
            logger.info(f"Running retrieval for query type: {q_type}")
            q_path = os.path.join(project_root, "prompts", q_type, f"{dataset}.parquet")
            if not os.path.exists(q_path):
                logger.warning(f"Question parquet file not found at: {q_path}")
                continue
                
            df_questions = pd.read_parquet(q_path)
            
            # Apply debug limit on questions if specified
            if max_questions and max_questions > 0:
                logger.info(f"Applying debug limit: processing first {max_questions} questions.")
                df_questions = df_questions.head(max_questions)
                
            q_ids = df_questions["id"].tolist()
            q_texts = df_questions["question"].tolist()
            
            retrieved_results = []
            
            for idx, q_text in enumerate(tqdm(q_texts, desc=f"Retrieving {q_type}/{dataset}")):
                if method == "sparse":
                    top_indices, _ = bm25_retriever.get_top_k_ranks(q_text, top_k)
                elif method == "dense":
                    dense_scores = dense_retriever.get_dense_scores(q_text)
                    top_indices = np.argpartition(dense_scores, -top_k)[-top_k:]
                    top_scores = dense_scores[top_indices]
                    top_indices = top_indices[np.argsort(-top_scores)].tolist()
                else: # hybrid
                    top_indices = hybrid_retriever.retrieve(q_text, top_k)
                    
                # Map back to original document IDs and texts
                retrieved_doc_ids = [corpus_ids[i] for i in top_indices]
                retrieved_contexts = [corpus_texts[i] for i in top_indices]
                
                retrieved_results.append({
                    "question_id": q_ids[idx],
                    "question": q_text,
                    "retrieved_ids": retrieved_doc_ids,
                    "retrieved_contexts": retrieved_contexts
                })
                
            # Write outputs to data/retrieved/
            output_dir = os.path.join(project_root, "data", "retrieved", q_type)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{dataset}.parquet")
            
            df_output = pd.DataFrame(retrieved_results)
            df_output.to_parquet(output_file, index=False)
            logger.info(f"Successfully saved retrieved results ({len(retrieved_results)} queries) -> {output_file}")


if __name__ == "__main__":
    main()
