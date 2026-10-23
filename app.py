import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure src directory is in sys.path
SRC_DIR = str(Path(__file__).resolve().parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
import requests



st.set_page_config(page_title="Multi-Mode AI Agent", layout="wide")

app_mode = st.sidebar.radio(
    "Choose Mode:",
    ["💻 Code Assistant", "🎙️ Voice Data Analyst"],
)

if app_mode == "🎙️ Voice Data Analyst":
    st.title("🎙️ Voice-Based Data Analysis Engine")
    st.caption("Upload a dataset, record a question, and review the analysis results.")
    api_url = os.getenv("VOICE_API_URL", "http://127.0.0.1:8000").rstrip("/")

    selected_lang_label = st.selectbox(
        "Select spoken language:",
        ["Auto Detect", "Arabic (العربية)", "English"],
        key="voice_language",
    )
    chosen_language = {
        "Auto Detect": "auto",
        "Arabic (العربية)": "ar",
        "English": "en",
    }[selected_lang_label]

    dataset_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls", "json"],
        key="voice_dataset",
    )
    if dataset_file is not None and st.button("Load dataset", type="primary"):
        try:
            response = requests.post(
                f"{api_url}/upload-dataset",
                files={"dataset_file": (dataset_file.name, dataset_file.getvalue(), dataset_file.type)},
                timeout=120,
            )
            response.raise_for_status()
            st.session_state.voice_dataset_info = response.json()
            st.session_state.voice_analysis = None
            st.success(f"Loaded {dataset_file.name}")
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Could not load dataset: {detail}")

    dataset_info = st.session_state.get("voice_dataset_info")
    if dataset_info:
        st.success(
            f"Dataset ready: {dataset_info['row_count']} rows · "
            f"table `{dataset_info['table_name']}`"
        )
        st.caption(f"Columns: {', '.join(dataset_info['columns'])}")
        with st.expander("Dataset preview"):
            st.dataframe(dataset_info["sample_records"], use_container_width=True)

        audio = st.audio_input("Record your data question", key="voice_question")
        if audio is not None and st.button("Analyze recording", type="primary"):
            with st.spinner("Transcribing your question and analyzing the dataset..."):
                try:
                    response = requests.post(
                        f"{api_url}/analyze-voice",
                        files={"audio_file": ("question.wav", audio.getvalue(), audio.type or "audio/wav")},
                        data={"language": chosen_language},
                        timeout=300,
                    )
                    response.raise_for_status()
                    st.session_state.voice_analysis = response.json()
                except requests.RequestException as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Analysis request failed: {detail}")

    analysis = st.session_state.get("voice_analysis")
    if analysis:
        st.subheader("Recognized question")
        st.write(analysis["transcription"])
        st.caption(
            f"Detected language: {analysis['language_used']} · "
            f"Confidence: {analysis['language_probability']}"
        )
        st.subheader("Generated SQL")
        st.code(analysis["sql"], language="sql")
        st.subheader("Results")
        st.dataframe(analysis["rows"], use_container_width=True)
        if analysis["truncated"]:
            st.caption("Showing the first 500 result rows.")
    st.stop()

from src import llm
from src import rag
from src.classifier import classify, route
from src.executer import run_code
from src.memory import Memory
from src.prompts import (
    EXPLAIN_SYSTEM,
    GENERATE_SYSTEM,
    NOT_RELEVANT_MSG,
    SOLUTION_DETECTION_PROMPT,
)


CODE_BLOCK = re.compile(r"```([^\r\n`]*)\r?\n(.*?)```", re.S)

# ---------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []      # {"role", "content", "meta"}
if "memory" not in st.session_state:
    st.session_state.memory = Memory()
if "pending_learning" not in st.session_state:
    st.session_state.pending_learning = None   # original request awaiting a solution
if "run_results" not in st.session_state:
    st.session_state.run_results = {}

memory: Memory = st.session_state.memory


# ---------------------------------------------------------------- helpers
def python_blocks(text: str) -> list[str]:
    blocks = []
    for language, code in CODE_BLOCK.findall(text):
        normalized_language = language.strip().lower()
        if normalized_language in ("python", "py", ""):
            blocks.append(code.strip("\r\n"))
    return blocks


def render_run_buttons(text: str, key_prefix: str):
    for i, code in enumerate(python_blocks(text)):
        key = f"{key_prefix}-{i}"
        if st.button(f"▶ Run code block {i + 1}", key=f"btn-{key}"):
            with st.spinner("Running in code runner tool..."):
                st.session_state.run_results[key] = run_code(code)
        res = st.session_state.run_results.get(key)
        if res:
            if res["stdout"]:
                st.caption("stdout")
                st.code(res["stdout"], language="text")
            if res["stderr"]:
                st.caption("stderr")
                st.code(res["stderr"], language="text")
            st.caption(f"exit code: {res['returncode']}")


def looks_like_solution(pending_request: str, message: str) -> bool:
    """Is this message a solution/reference for the request that failed retrieval?"""
    if "```" in message:
        return True
    try:
        out = llm.complete(
            SOLUTION_DETECTION_PROMPT,
            [{"role": "user",
              "content": f"Earlier request:\n{pending_request}\n\nMessage:\n{message[:4000]}"}],
            max_tokens=3,
        )
        return out.strip().upper().startswith("YES")
    except Exception:
        return False


def build_messages(user_content: str) -> list[dict]:
    msgs = st.session_state.messages
    # If the user message was already appended to session_state, take previous messages as history
    if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"] == user_content:
        history = [{"role": m["role"], "content": m["content"]} for m in msgs[-9:-1]]
    else:
        history = [{"role": m["role"], "content": m["content"]} for m in msgs[-8:]]
    return history + [{"role": "user", "content": user_content}]


def stream_to_ui(system: str, messages: list[dict]) -> str:
    return st.write_stream(llm.stream(system, messages))


# ---------------------------------------------------------------- pipelines
def explain_pipeline(query: str) -> tuple[str, dict]:
    """Route 1: direct LLM analysis. No RAG, no vector DB, no embeddings."""
    system = EXPLAIN_SYSTEM + "\n\nConversation memory:\n" + memory.render()
    return stream_to_ui(system, build_messages(query)), {}


def generate_pipeline(query: str) -> tuple[str, dict]:
    """Route 2: RAG retrieval -> relevance check -> LLM."""
    with st.status("Searching knowledge base...", expanded=False) as status:
        docs = rag.retrieve(query, k=4)
        status.update(label=f"Retrieved {len(docs)} chunk(s). Checking relevance...")
        ok = rag.is_relevant(query, docs)
        status.update(label="Relevant context found." if ok else "No relevant context.",
                      state="complete")
    meta = {"retrieved": len(docs), "relevant": ok,
            "sources": sorted({d["meta"].get("source", "?") for d in docs})}

    if not ok:
        st.markdown(NOT_RELEVANT_MSG)
        st.session_state.pending_learning = query
        return NOT_RELEVANT_MSG, meta

    context = "\n\n---\n\n".join(
        f"[source: {d['meta'].get('source')}]\n{d['text']}" for d in docs)
    system = (GENERATE_SYSTEM + "\n\nConversation memory:\n" + memory.render()
              + "\n\nRetrieved knowledge:\n" + context)
    return stream_to_ui(system, build_messages(query)), meta


PIPELINES = {"explain_pipeline": explain_pipeline, "generate_pipeline": generate_pipeline}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Model: `{llm.MODEL}`")
    uploaded = st.file_uploader("Attach a code file (optional)",
                                type=["py", "txt", "js", "ts", "java", "c", "cpp", "cs", "csx", "go", "rs", "md"])

    if uploaded is not None:
        if st.button("📥 Index file to Knowledge Base"):
            file_text = uploaded.getvalue().decode("utf-8", errors="replace")
            chunks_added = rag.add_document(file_text, source=uploaded.name)
            st.success(f"Added {chunks_added} chunk(s) from `{uploaded.name}`!")
            st.rerun()

    st.divider()
    st.subheader("📚 Knowledge base")
    st.metric("Chunks in Chroma", rag.count())

    st.divider()
    st.subheader("🧠 Memory")
    with st.expander("Preferences", expanded=True):
        st.json(memory.data["preferences"] or {})
    with st.expander("Recent questions"):
        for t in memory.data["turns"][-8:]:
            st.write("•", t["user"][:100])
    with st.expander("Last generated code"):
        st.code(memory.data["last_code"] or "# none yet", language="python")
    if st.button("Clear memory & chat"):
        memory.clear()
        st.session_state.messages = []
        st.session_state.pending_learning = None
        st.session_state.run_results = {}
        st.rerun()

# ---------------------------------------------------------------- main UI
st.title("💻 Code Assistant Pipeline")
st.caption("Explain code · Generate code with RAG")

for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta") or {}
        if meta.get("route"):
            st.caption(f"intent: {meta['intent']} → {meta['route']}"
                       + (f" · sources: {', '.join(meta['sources'])}" if meta.get("sources") else ""))
        if m["role"] == "assistant":
            render_run_buttons(m["content"], f"hist{idx}")

prompt = st.chat_input("Ask me to explain or write code...")

if prompt:
    full_prompt = prompt
    if uploaded is not None:
        file_text = uploaded.getvalue().decode("utf-8", errors="replace")
        full_prompt += f"\n\nAttached file `{uploaded.name}`:\n```\n{file_text[:20000]}\n```"

    st.session_state.messages.append({"role": "user", "content": full_prompt})
    with st.chat_message("user"):
        st.markdown(full_prompt)

    with st.chat_message("assistant"):
        pending = st.session_state.pending_learning
        learned = False

        # Section 7: human feedback learning
        if pending and looks_like_solution(pending, full_prompt):
            n = rag.add_document(
                f"Request: {pending}\n\nSolution:\n{full_prompt}",
                source="user-feedback",
                extra_meta={"origin": "human_feedback", "original_request": pending[:300]},
            )
            st.session_state.pending_learning = None
            reply = (f"Thank you! I stored your solution in my knowledge base "
                     f"({n} chunk{'s' if n != 1 else ''}) and will use it in future answers.")
            st.markdown(reply)
            meta, learned = {"route": "learning", "intent": "FEEDBACK"}, True
        else:
            st.session_state.pending_learning = None
            memory.update_preferences(prompt)
            intent = classify(full_prompt)                  # Section 2
            pipeline = route(intent)                        # Section 3
            st.caption(f"intent: {intent} → {pipeline}")
            reply, meta = PIPELINES[pipeline](full_prompt)
            meta.update({"intent": intent, "route": pipeline})

        blocks = python_blocks(reply)
        if not learned:
            # Only store in "last_code" if produced by Route 2 (Code Generation)
            generated_code = blocks[-1] if (blocks and meta.get("route") == "generate_pipeline") else ""
            memory.add_turn(prompt, reply, code=generated_code)
        st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
        render_run_buttons(reply, f"new{len(st.session_state.messages)}")





