"""
Load the FAISS index and retrieve top-K similar corpus examples for a query problem.

Usage
-----
    from src.rag.retriever import Retriever

    retriever = Retriever("data/rag_corpus")
    results = retriever.retrieve("Janet has 3 apples and buys 5 more. How many?", top_k=3)
    # results: list of dicts with {problem, solution, answer, source, text}
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, corpus_dir: str | Path, embedding_model: str | None = None) -> None:
        corpus_dir = Path(corpus_dir)

        with (corpus_dir / "metadata.json").open() as f:
            metadata = json.load(f)

        model_name = embedding_model or metadata["embedding_model"]

        with (corpus_dir / "raw_corpus.json").open() as f:
            self.corpus = json.load(f)

        self.embeddings = np.load(corpus_dir / "embeddings.npy")
        self.index      = faiss.read_index(str(corpus_dir / "faiss.index"))
        self.model      = SentenceTransformer(model_name)

        print(f"Retriever loaded: {len(self.corpus)} examples, model='{model_name}'")

    def retrieve(self, problem: str, top_k: int = 3) -> list[dict]:
        query = self.model.encode(
            [problem],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query, top_k)
        return [
            {**self.corpus[i], "score": float(scores[0][rank])}
            for rank, i in enumerate(indices[0])
            if i < len(self.corpus)
        ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Debug: retrieve top-K examples for a problem.")
    ap.add_argument("--corpus-dir", default="data/rag_corpus")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--problem",
        default="Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes "
                "muffins with 4. She sells the remainder at $2 per egg. How much does she make daily?",
    )
    args = ap.parse_args()

    retriever = Retriever(args.corpus_dir)
    results   = retriever.retrieve(args.problem, top_k=args.top_k)

    print(f"\nQuery:\n  {args.problem}\n")
    print(f"Top-{args.top_k} retrieved examples:")
    print("=" * 60)
    for rank, ex in enumerate(results, 1):
        print(f"\n[{rank}]  score={ex['score']:.4f}  source={ex['source']}")
        print(f"  Problem : {ex['problem']}")
        print(f"  Solution: {ex['solution']}")
        print(f"  Answer  : {ex['answer']}")
