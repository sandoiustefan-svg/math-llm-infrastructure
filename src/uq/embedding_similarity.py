from __future__ import annotations

import numpy as np

_RANK_LOW  = 0.3
_RANK_HIGH = 0.7


def _rank(mean_sim: float) -> str:
    if mean_sim < _RANK_LOW:
        return "low"
    if mean_sim > _RANK_HIGH:
        return "high"
    return "medium"


def _encode(model, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=len(texts) > 200,
    ).astype(np.float32)


def enrich(results: list[dict], problems: list[dict], model) -> list[dict]:
    """
    Add embedding-similarity fields to each result dict in-place.

    Args:
        results:  output of MCDropoutEvaluator.evaluate() — must be aligned with problems.
        problems: test-set dicts with "expected_answer" key.
        model:    loaded SentenceTransformer instance.

    Adds per problem:
        raw_similarities    : list[float]  cosine sim per MC Dropout pass
        mean_raw_similarity : float        mean across passes
        std_raw_similarity  : float        std across passes
        similarity_rank     : str          "low" / "medium" / "high"
    """
    # Use full reference solution for embedding if available, else fall back to
    # expected_answer. This gives a fair semantic comparison: model CoT vs
    # dataset CoT, rather than model CoT vs a bare answer string.
    expected = [
        p.get("reference_solution") or p.get("expected_answer", "")
        for p in problems
    ]

    unique_answers = list(dict.fromkeys(expected))
    answer_to_idx  = {a: i for i, a in enumerate(unique_answers)}

    all_raws: list[str] = []
    offsets: list[tuple[int, int]] = []
    for r in results:
        start = len(all_raws)
        all_raws.extend(r.get("raws", []))
        offsets.append((start, len(all_raws)))

    answer_embs = _encode(model, unique_answers)
    raw_embs = (
        _encode(model, all_raws)
        if all_raws
        else np.zeros((0, answer_embs.shape[1]), dtype=np.float32)
    )

    for i, r in enumerate(results):
        ans_emb    = answer_embs[answer_to_idx[expected[i]]]
        start, end = offsets[i]

        sims = (raw_embs[start:end] @ ans_emb).tolist() if end > start else []

        mean_sim = float(np.mean(sims)) if sims else float("nan")
        std_sim  = float(np.std(sims))  if sims else float("nan")

        r["raw_similarities"]    = [round(s, 6) for s in sims]
        r["mean_raw_similarity"] = round(mean_sim, 6)
        r["std_raw_similarity"]  = round(std_sim,  6)
        r["similarity_rank"]     = _rank(mean_sim) if sims else "unknown"

    return results
