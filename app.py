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

import torch
import streamlit as st
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer

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
.block-container { max-width: 900px; padding-top: 2rem; }
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
# Model loading
# ---------------------------------------------------------------------------

def _resolve(path: str) -> str:
    p = Path(path)
    return str((PROJECT_ROOT / p).resolve()) if not p.is_absolute() else path


@st.cache_resource(show_spinner="Loading chat model…")
def load_chat_model(model_path: str, tokenizer_name: str):
    """Single model cached for fast chat inference."""
    model_path = _resolve(model_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to("cuda").eval()
    return model, tok


@st.cache_resource(show_spinner="Loading MC Dropout evaluator…")
def load_mc_dropout(model_path: str, tokenizer: str, num_passes: int) -> MCDropoutEvaluator:
    cfg = MCDropoutConfig(
        model_path=_resolve(model_path),
        tokenizer_name=tokenizer,
        num_passes=num_passes,
        device="cuda",
    )
    return MCDropoutEvaluator(cfg)


@st.cache_resource(show_spinner="Loading ensemble evaluator…")
def load_ensemble(model_paths_str: str, tokenizer: str) -> EnsembleEvaluator:
    paths = [_resolve(p.strip()) for p in model_paths_str.split("\n") if p.strip()]
    cfg = EnsembleConfig(model_paths=paths, tokenizer_name=tokenizer, device="cuda")
    return EnsembleEvaluator(cfg)


def quick_generate(model, tokenizer, problem: str, max_new_tokens: int) -> str:
    """Single greedy forward pass for chat (no UQ overhead)."""
    prompt = f"### Problem:\n{problem.strip()}\n\n### Solution:\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.3,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧮 Math LLM — UQ Demo")
    st.divider()

    model_choice = st.radio(
        "Model",
        ["Scratch 1B — MC Dropout", "Fine-tuned Llama-3.2-1B — Ensemble"],
    )
    is_mc = model_choice.startswith("Scratch")

    st.divider()

    def _find_checkpoints() -> list[str]:
        outputs_dir = PROJECT_ROOT / "outputs"
        checkpoints = []
        if outputs_dir.exists():
            for run_dir in sorted(outputs_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                for name in ("best", "final"):
                    p = run_dir / "checkpoints" / name
                    if p.exists():
                        checkpoints.append(str(p.resolve()))
        return checkpoints

    def _label(p: str) -> str:
        parts = Path(p).parts
        try:
            idx = parts.index("checkpoints")
            return "/".join(parts[idx - 1:])
        except ValueError:
            return Path(p).name

    available = _find_checkpoints()

    if is_mc:
        if available:
            model_path = st.selectbox("Model checkpoint", available, format_func=_label)
        else:
            model_path = st.text_input("Model checkpoint path", value="outputs/scratch_1b/checkpoints/best")
        num_passes = st.slider("MC Dropout passes", 5, 50, 20, step=5)
        chat_model_path = model_path
    else:
        if available:
            selected = st.multiselect(
                "Ensemble checkpoints",
                options=available,
                default=available[:3],
                format_func=_label,
                help="Select one checkpoint per seed.",
            )
            model_paths = "\n".join(selected)
        else:
            model_paths = st.text_area(
                "Ensemble checkpoint paths (one per line)",
                value="\n".join([
                    "outputs/finetune_seed42/checkpoints/best",
                    "outputs/finetune_seed43/checkpoints/best",
                    "outputs/finetune_seed44/checkpoints/best",
                ]),
                height=100,
            )
        # Use the first selected path for chat
        chat_model_path = model_paths.strip().splitlines()[0] if model_paths.strip() else ""

    tokenizer = st.text_input("Tokenizer", value="mistralai/Mistral-7B-v0.1")
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 512, step=64)

    load_btn = st.button("Load Model", type="primary", use_container_width=True)

    st.divider()

    # Session state init
    for key, val in [
        ("chat_model", None),
        ("evaluator", None),
        ("evaluator_type", None),
        ("messages", []),
        ("uq_request", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    if load_btn:
        with st.spinner("Loading…"):
            try:
                st.session_state.chat_model = load_chat_model(chat_model_path, tokenizer)
                st.session_state.chat_max_tokens = max_new_tokens

                if is_mc:
                    st.session_state.evaluator = load_mc_dropout(model_path, tokenizer, num_passes)
                    st.session_state.evaluator_type = "mc_dropout"
                else:
                    st.session_state.evaluator = load_ensemble(model_paths, tokenizer)
                    st.session_state.evaluator_type = "ensemble"

                st.session_state.evaluator.cfg.max_new_tokens = max_new_tokens
                st.success("Model loaded!")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    if st.session_state.chat_model is not None:
        label = "MC Dropout" if st.session_state.evaluator_type == "mc_dropout" else "Ensemble"
        st.success(f"✅ {label} ready")
    else:
        st.warning("No model loaded")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: confidence gauge
# ---------------------------------------------------------------------------

def confidence_gauge(value: float) -> go.Figure:
    pct = value * 100
    color = "#a6e3a1" if pct >= 80 else ("#f9e2af" if pct >= 50 else "#f38ba8")
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
                {"range": [80, 100], "color": "#2a2a3e"},
            ],
        },
    ))
    fig.update_layout(height=160, margin=dict(l=10, r=10, t=20, b=0),
                      paper_bgcolor="#1e1e2e", font_color="#cdd6f4")
    return fig


# ---------------------------------------------------------------------------
# Helper: render UQ metrics block
# ---------------------------------------------------------------------------

def render_uq(result: dict, method: str) -> None:
    confidence = result.get("confidence", float("nan"))
    entropy    = result.get("entropy", float("nan"))
    answers    = result.get("answers", [])
    n          = len(answers)
    majority   = result.get("majority_answer", "")
    votes      = answers.count(majority)

    conf_class = "high" if confidence >= 0.8 else ("medium" if confidence >= 0.5 else "low")

    with st.expander("📊 Uncertainty Quantification", expanded=True):
        col_gauge, col_metrics = st.columns([1, 2])

        with col_gauge:
            st.plotly_chart(confidence_gauge(confidence),
                            use_container_width=True, config={"displayModeBar": False})

        with col_metrics:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="uq-label">Answer confidence</div>'
                    f'<div class="uq-value {conf_class}">{confidence*100:.1f}%</div>',
                    unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div class="uq-label">Entropy</div>'
                    f'<div class="uq-value">{entropy:.3f} bits</div>',
                    unsafe_allow_html=True)
            with c3:
                method_label = f"MC Dropout ({n} passes)" if method == "mc_dropout" else f"Ensemble ({n} members)"
                st.markdown(
                    f'<div class="uq-label">{method_label}</div>'
                    f'<div class="uq-value">{votes}/{n} agree</div>',
                    unsafe_allow_html=True)

        st.markdown("**Token-level confidence measures**")
        measures = [
            ("Full sequence",      "full_sequence_mean_confidence", "full_sequence_perplexity"),
            ("Answer span only",   "answer_span_mean_confidence",   "answer_span_perplexity"),
            ("Numeric tokens",     "numeric_mean_confidence",       "numeric_perplexity"),
        ]
        cols = st.columns(3)
        for col, (label, ck, pk) in zip(cols, measures):
            cv = result.get(ck, float("nan"))
            pv = result.get(pk, float("nan"))
            cs = f"{cv*100:.1f}%" if cv == cv else "n/a"
            ps = f"{pv:.2f}"     if pv == pv  else "n/a"
            with col:
                st.markdown(
                    f'<div class="uq-label">{label}</div>'
                    f'<div class="uq-value">conf: {cs}</div>'
                    f'<div style="color:#a6adc8;font-size:0.8rem">ppl: {ps}</div>',
                    unsafe_allow_html=True)

        if answers:
            st.markdown("**Answer distribution**")
            counts: dict[str, int] = {}
            for a in answers:
                counts[a] = counts.get(a, 0) + 1
            for ans, cnt in sorted(counts.items(), key=lambda x: -x[1])[:5]:
                pct = cnt / n
                bar_color = "#a6e3a1" if ans == majority else "#f38ba8"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
                    f'<div style="background:{bar_color};height:14px;width:{int(pct*200)}px;border-radius:3px"></div>'
                    f'<span style="color:#a6adc8;font-size:0.8rem">{cnt}/{n} — <code>{ans[:60]}</code></span>'
                    f'</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

st.markdown("### Math Reasoning + Uncertainty Quantification")
st.caption("Ask a math problem. The model will solve it and you can calculate uncertainty on any reply.")

# Handle pending UQ calculation (triggered by button click on previous rerun)
if st.session_state.uq_request is not None:
    idx = st.session_state.uq_request
    msg = st.session_state.messages[idx]
    with st.spinner("Calculating uncertainty — this may take a minute…"):
        try:
            evaluator = st.session_state.evaluator
            results = evaluator.evaluate([{"problem": msg["prompt"]}])
            st.session_state.messages[idx]["uq"] = results[0]
        except Exception as e:
            st.session_state.messages[idx]["uq_error"] = str(e)
    st.session_state.uq_request = None
    st.rerun()

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("uq"):
                render_uq(msg["uq"], st.session_state.evaluator_type)
            elif msg.get("uq_error"):
                st.error(f"UQ error: {msg['uq_error']}")
            elif st.session_state.evaluator is not None:
                if st.button("📊 Calculate UQ", key=f"uq_btn_{i}"):
                    st.session_state.uq_request = i
                    st.rerun()

# Chat input
user_input = st.chat_input(
    "Ask a math problem…",
    disabled=(st.session_state.chat_model is None),
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                model, tok = st.session_state.chat_model
                max_tok = getattr(st.session_state, "chat_max_tokens", 512)
                raw = quick_generate(model, tok, user_input, max_tok)
                st.markdown(raw)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": raw,
                    "prompt": user_input,
                    "uq": None,
                })
            except Exception as e:
                st.error(f"Inference error: {e}")

if st.session_state.chat_model is None:
    st.info("👈 Load a model from the sidebar to start.")
