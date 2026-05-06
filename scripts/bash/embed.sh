#!/bin/bash
# Build the RAG retrieval corpus: fetch datasets → format → embed → FAISS index.
#
# Usage:
#   bash scripts/bash/embed.sh           # full run
#   bash scripts/bash/embed.sh --debug   # 5 examples per dataset, fast smoke-test

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
OUT_DIR="data/rag_corpus"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
BATCH_SIZE=64
DEBUG_FLAG=""
# ──────────────────────────────────────────────────────────────────────────────

for arg in "$@"; do
  case $arg in
    --debug)
      DEBUG_FLAG="--debug"
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " RAG CORPUS CONFIG"
echo "  out_dir         : $OUT_DIR"
echo "  embedding_model : $EMBEDDING_MODEL"
echo "  batch_size      : $BATCH_SIZE"
echo "  debug           : ${DEBUG_FLAG:-off}"
echo "============================================================"

# ── Step 1: Fetch + format datasets ───────────────────────────────────────────
if [ -f "$OUT_DIR/raw_corpus.json" ] && [ -z "$DEBUG_FLAG" ]; then
  echo "raw_corpus.json already exists — skipping fetch."
else
  echo ""
  echo "Step 1: Fetching SVAMP, ASDiv, MAWPS ..."
  python -m src.rag.build_corpus \
    --out-dir "$OUT_DIR" \
    $DEBUG_FLAG
  echo "Done → $OUT_DIR/raw_corpus.json"
fi

# ── Step 2: Embed + build FAISS index ─────────────────────────────────────────
if [ -f "$OUT_DIR/faiss.index" ] && [ -z "$DEBUG_FLAG" ]; then
  echo "faiss.index already exists — skipping embedding."
else
  echo ""
  echo "Step 2: Embedding problems and building FAISS index ..."
  python -m src.rag.embed_corpus \
    --corpus-dir      "$OUT_DIR" \
    --embedding-model "$EMBEDDING_MODEL" \
    --batch-size      "$BATCH_SIZE"
  echo "Done → $OUT_DIR/faiss.index"
fi

echo ""
echo "============================================================"
echo " RAG corpus complete."
echo "  $(python -c "import json; c=json.load(open('$OUT_DIR/raw_corpus.json')); print(f\"{len(c)} examples total\")")"
echo "  $(python -c "import json; m=json.load(open('$OUT_DIR/metadata.json')); print(f\"embedding_model: {m['embedding_model']}  dim: {m['embedding_dim']}\")")"
echo "  output: $OUT_DIR"
echo "============================================================"
