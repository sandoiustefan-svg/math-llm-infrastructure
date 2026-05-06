"""
  Embed the RAG corpus problems and build a FAISS index.
                                                                                                                                                                                                                                                             
  Reads  : data/rag_corpus/raw_corpus.json
  Writes : data/rag_corpus/embeddings.npy                                                                                                                                                                                                                    
           data/rag_corpus/faiss.index
           data/rag_corpus/metadata.json                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
  Usage
  -----                                                                                                                                                                                                                                                      
      # full run  
      python -m src.rag.embed_corpus --corpus-dir data/rag_corpus
                                                                                                                                                                                                                                                             
      # different embedding model
      python -m src.rag.embed_corpus --corpus-dir data/rag_corpus \                                                                                                                                                                                          
          --embedding-model all-mpnet-base-v2                                                                                                                                                                                                                
  """
from __future__ import annotations                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                            
import argparse
import json                                                                                                                                                                                                                                                
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
                                                                                                                                                                                                                                                            

def load_corpus(corpus_dir: Path) -> list[dict]:                                                                                                                                                                                                           
    path = corpus_dir / "raw_corpus.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)       
    
def embed_problems(problems: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    print(f"Embedding {len(problems)} problems with '{model_name}' ...")                                                                                                                                                                                   
    model = SentenceTransformer(model_name)                                                                                                                                                                                                                
    embeddings = model.encode(
        problems,                                                                                                                                                                                                                                          
        batch_size=batch_size,                                                                                                                                                                                                                             
        normalize_embeddings=True,   # L2-normalise → inner product == cosine similarity
        show_progress_bar=True,                                                                                                                                                                                                                            
        convert_to_numpy=True,
    )                                                                                                                                                                                                                                                      
    return embeddings.astype(np.float32)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
      dim = embeddings.shape[1]                                                                                                                                                                                                                              
      index = faiss.IndexFlatIP(dim)
      index.add(embeddings)
      print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")                                                                                                                                                                                         
      return index
                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                             
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Embed corpus problems and build FAISS index.")                                                                                                                                                               
    ap.add_argument("--corpus-dir", default="data/rag_corpus")                                                                                                                                                                                             
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)                                                                                                                                                                                                  
    return ap.parse_args()
                                                                                                                                                                                                                                                            
                
def main() -> None:
    args = _parse_args()
    corpus_dir = Path(args.corpus_dir)
                                                                                                                                                                                                                                                            
    examples = load_corpus(corpus_dir)
    problems = [ex["problem"] for ex in examples]                                                                                                                                                                                                          
                
    embeddings = embed_problems(problems, args.embedding_model, args.batch_size)                                                                                                                                                                           
    np.save(corpus_dir / "embeddings.npy", embeddings)
    print(f"Embeddings → {corpus_dir / 'embeddings.npy'}  shape={embeddings.shape}")                                                                                                                                                                       
                                                                                                                                                                                                                                                            
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(corpus_dir / "faiss.index"))                                                                                                                                                                                              
    print(f"Index      → {corpus_dir / 'faiss.index'}")
                                                                                                                                                                                                                                                            
    metadata = {
        "embedding_model": args.embedding_model,                                                                                                                                                                                                           
        "num_examples":    len(examples),
        "embedding_dim":   int(embeddings.shape[1]),
    }
    with (corpus_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)                                                                                                                                                                                                                   
    print(f"Metadata   → {corpus_dir / 'metadata.json'}")
                                                                                                                                                                                                                                                            
                
if __name__ == "__main__":
    main()