"""
UQ Demo — interactive math reasoning with uncertainty quantification.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import plotly.graph_objects as go

from src.uq.mc_dropout import MCDropoutConfig, MCDropoutEvaluator
from src.uq.ensemble import EnsembleConfig, EnsembleEvaluator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Math LLM — UQ Demo",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Chat container max width */
.block-container { max-width: 900px; padding-top: 2rem; }

/* UQ metrics card */
.uq-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.6rem;
    font-size: 0.85rem;
}
.uq-label { color: #a6adc8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.uq-value { color: #cdd6f4; font-size: 1.1rem; font-weight: 600; }
.high   { color: #a6e3a1; }
.medium { color: #f9e2af; }
.low    { color: #f38ba8; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached — only reloads when config changes)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_mc_dropout(model_path: str, tokenizer: str, num_passes: int, device: str) -> MCDropoutEvaluator:
    cfg = MCDropoutConfig(
        model_path=model_path,
        tokenizer_name=tokenizer,
        num_passes=num_passes,
        device=device,
    )
    return MCDropoutEvaluator(cfg)


@st.cache_resource(show_spinner="Loading ensemble models…")
def load_ensemble(model_paths_str: str, tokenizer: str, device: str) -> EnsembleEvaluator:
    paths = [p.strip() for p in model_paths_str.split(",") if p.strip()]
    cfg = EnsembleConfig(
        model_paths=paths,
        tokenizer_name=tokenizer,
        device=device,
    )
    return EnsembleEvaluator(cfg)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧮 Math LLM — UQ Demo")
    st.divider()

    model_choice = st.radio(
        "Model",
        ["Scratch 1B — MC Dropout", "Fine-tuned Llama-3.2-1B — Ensemble"],
        help="Scratch model uses MC Dropout (stochastic dropout at inference). "
             "Fine-tuned model uses Deep Ensembles (multiple seeds).",
    )

    st.divider()

    is_mc = model_choice.startswith("Scratch")

    if is_mc:
        model_path = st.text_input(
            "Model checkpoint path",
            value="outputs/scratch_1b/checkpoints/best",
        )
        num_passes = st.slider("MC Dropout passes", min_value=5, max_value=50, value=20, step=5)
    else:
        model_paths = st.text_area(
            "Ensemble checkpoint paths (one per line)",
            value="\n".join([
                "outputs/finetune_seed42/checkpoints/best",
                "outputs/finetune_seed43/checkpoints/best",
                "outputs/finetune_seed44/checkpoints/best",
            ]),
            height=120,
        )

    tokenizer = st.text_input("Tokenizer", value="mistralai/Mistral-7B-v0.1")
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 512, step=64)

    load_btn = st.button("Load Model", type="primary", use_container_width=True)

    st.divider()

    # Session state for evaluator
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = None
        st.session_state.evaluator_type = None

    if load_btn:
        with st.spinner("Loading…"):
            try:
                if is_mc:
                    st.session_state.evaluator = load_mc_dropout(
                        model_path, tokenizer, num_passes, device
                    )
                    st.session_state.evaluator_type = "mc_dropout"
                    st.session_state.evaluator.cfg.max_new_tokens = max_new_tokens
                else:
                    paths_str = ",".join(
                        line.strip() for line in model_paths.splitlines() if line.strip()
                    )
                    st.session_state.evaluator = load_ensemble(paths_str, tokenizer, device)
                    st.session_state.evaluator_type = "ensemble"
                    st.session_state.evaluator.cfg.max_new_tokens = max_new_tokens
                st.success("Model loaded!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if st.session_state.evaluator is not None:
        ev_type = st.session_state.evaluator_type
        label = "MC Dropout" if ev_type == "mc_dropout" else "Ensemble"
        st.success(f"✅ {label} ready")
    else:
        st.warning("No model loaded")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Helper: confidence gauge (Plotly)
# ---------------------------------------------------------------------------

def confidence_gauge(value: float) -> go.Figure:
    """Semicircular gauge, value in [0, 1]."""
    pct = value * 100
    if pct >= 80:
        color = "#a6e3a1"
    elif pct >= 50:
        color = "#f9e2af"
    else:
        color = "#f38ba8"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 22, "color": "#cdd6f4"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#585b70", "tickfont": {"color": "#a6adc8"}},
            "bar": {"color": color},
            "bgcolor": "#1e1e2e",
            "bordercolor": "#313244",
            "steps": [
                {"range": [0, 50],  "color": "#313244"},
                {"range": [50, 80], "color": "#2a2a3e"},
                {"range": [80, 100],"color": "#2a2a3e"},
            ],
        },
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=20, b=0),
        paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: render UQ metrics block
# ---------------------------------------------------------------------------

def render_uq(result: dict, method: str) -> None:
    confidence = result["confidence"]
    entropy    = result["entropy"]
    perplexity = result["token_perplexity"]
    answers    = result["answers"]
    n          = len(answers)
    majority   = result["majority_answer"]
    votes      = answers.count(majority)

    conf_class = "high" if confidence >= 0.8 else ("medium" if confidence >= 0.5 else "low")

    with st.expander("📊 Uncertainty Quantification", expanded=True):
        col_gauge, col_metrics = st.columns([1, 2])

        with col_gauge:
            st.plotly_chart(confidence_gauge(confidence), use_container_width=True, config={"displayModeBar": False})

        with col_metrics:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="uq-label">Answer confidence</div><div class="uq-value {conf_class}">{confidence*100:.1f}%</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="uq-label">Entropy</div><div class="uq-value">{entropy:.3f} bits</div>', unsafe_allow_html=True)
            with c3:
                method_label = f"MC Dropout ({n} passes)" if method == "mc_dropout" else f"Ensemble ({n} members)"
                st.markdown(f'<div class="uq-label">{method_label}</div><div class="uq-value">{votes}/{n} agree</div>', unsafe_allow_html=True)

        # Token-level confidence comparison table
        st.markdown("**Token-level confidence measures**")
        measures = [
            ("Full sequence", "full_sequence_mean_confidence", "full_sequence_perplexity"),
            ("Answer span only", "answer_span_mean_confidence", "answer_span_perplexity"),
            ("Numeric tokens only", "numeric_mean_confidence", "numeric_perplexity"),
        ]
        cols = st.columns(3)
        for col, (label, conf_key, ppl_key) in zip(cols, measures):
            conf_val = result.get(conf_key, float("nan"))
            ppl_val  = result.get(ppl_key, float("nan"))
            conf_str = f"{conf_val*100:.1f}%" if conf_val == conf_val else "n/a"
            ppl_str  = f"{ppl_val:.2f}"       if ppl_val == ppl_val  else "n/a"
            with col:
                st.markdown(
                    f'<div class="uq-label">{label}</div>'
                    f'<div class="uq-value">conf: {conf_str}</div>'
                    f'<div style="color:#a6adc8;font-size:0.8rem">ppl: {ppl_str}</div>',
                    unsafe_allow_html=True,
                )

        # Answer distribution bar
        st.markdown("**Answer distribution across passes**")
        counts: dict[str, int] = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        for ans, cnt in sorted_counts[:5]:  # top 5 distinct answers
            pct = cnt / n
            bar_color = "#a6e3a1" if ans == majority else "#f38ba8"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
                f'<div style="background:{bar_color};height:14px;width:{int(pct*200)}px;border-radius:3px"></div>'
                f'<span style="color:#a6adc8;font-size:0.8rem">{cnt}/{n} — <code>{ans[:60]}</code></span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

st.markdown("### Math Reasoning + Uncertainty Quantification")
st.caption("Ask a math problem. The model will solve it and show how confident it is.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "uq" in msg:
            render_uq(msg["uq"], msg["method"])

# Input
prompt = st.chat_input("Ask a math problem…", disabled=(st.session_state.evaluator is None))

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run inference
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            evaluator = st.session_state.evaluator
            method    = st.session_state.evaluator_type
            try:
                results = evaluator.evaluate([{"problem": prompt}])
                result  = results[0]
                answer  = result["majority_answer"]

                st.markdown(answer)
                render_uq(result, method)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "uq": result,
                    "method": method,
                })
            except Exception as e:
                st.error(f"Inference error: {e}")

if st.session_state.evaluator is None:
    st.info("👈 Load a model from the sidebar to start.")
