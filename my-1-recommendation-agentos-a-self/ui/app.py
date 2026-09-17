"""AgentOS Chat UI — Autonomous Self-Evolving AI Assistant with Live Thought Streaming."""
import json
import uuid
import time
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentOS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* Chat bubbles */
.user-bubble {
    background: #1e3a5f;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0 8px 80px;
    color: #e8f4fd;
    font-size: 0.95rem;
    line-height: 1.5;
}
.assistant-container {
    background: #121226;
    border: 1px solid #262648;
    border-radius: 16px 16px 16px 4px;
    padding: 16px 20px;
    margin: 8px 80px 8px 0;
    color: #e0e0e0;
}
.tool-badge {
    display: inline-block;
    background: #0d3349;
    border: 1px solid #1565c0;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #64b5f6;
    margin: 4px 3px 2px 0;
}
.new-tool-badge {
    display: inline-block;
    background: #1a2e1a;
    border: 1px solid #2e7d32;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #81c784;
    margin: 4px 3px 2px 0;
}
.timeout-warning {
    background: #3e2723;
    border: 1px solid #bf360c;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.82rem;
    color: #ff8a65;
    margin-top: 8px;
}
.welcome-box {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1a3e 100%);
    border: 1px solid #2d2d54;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 40px auto;
    max-width: 650px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────

def _get(path: str, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default

def _delete(path: str):
    r = requests.delete(f"{API_BASE}{path}", timeout=5)
    r.raise_for_status()
    return r.json()

def _stream_chat(prompt: str, session_id: str):
    """Generator consuming Server-Sent Events from /chat/stream."""
    url = f"{API_BASE}/chat/stream"
    with requests.post(
        url,
        json={"prompt": prompt, "session_id": session_id},
        stream=True,
        timeout=3600,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                raw = line[6:].strip()
                try:
                    yield json.loads(raw)
                except Exception:
                    pass

def check_health():
    data = _get("/health", {"api": False, "ollama": False})
    return data.get("api", False), data.get("ollama", False)

def get_tools():
    return _get("/tools", [])

TOOL_ICONS = {
    "file_search": "📁",
    "read_file": "📄",
    "file_reader": "📂",
    "web_search": "🔍",
    "url_scrape": "🌐",
    "url_read": "🌐",
    "run_python": "🐍",
    "memory_query": "🧠",
    "generate_tool": "✨",
}

def tool_icon(name: str) -> str:
    for key, icon in TOOL_ICONS.items():
        if key in name.lower():
            return icon
    return "🔧"

# ── Session state ─────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thinking" not in st.session_state:
    st.session_state.thinking = False
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    api_ok, llm_ok = check_health()
    col1, col2 = st.columns(2)
    with col1:
        dot = "🟢" if api_ok else "🔴"
        st.markdown(f"{dot} **API**")
    with col2:
        dot = "🟢" if llm_ok else "🔴"
        st.markdown(f"{dot} **LLM**")

    st.markdown("---")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    if st.session_state.total_tokens:
        st.caption(f"Tokens used: {st.session_state.total_tokens:,}")

    st.markdown("---")

    st.subheader("🛠 Available Tools")
    tools = get_tools()
    static_tools = [t for t in tools if t["name"] != "generate_tool" and "✨" not in t.get("description", "")]
    generated_tools = [t for t in tools if t not in static_tools]

    if static_tools:
        with st.expander(f"Built-in ({len(static_tools)})", expanded=False):
            for t in static_tools:
                icon = tool_icon(t["name"])
                st.markdown(f"**{icon} {t['name']}**")
                st.caption(t["description"][:90] + "..." if len(t["description"]) > 90 else t["description"])

    if generated_tools:
        with st.expander(f"✨ Generated ({len(generated_tools)})", expanded=True):
            for t in generated_tools:
                col_t, col_del = st.columns([4, 1])
                with col_t:
                    st.markdown(f"**✨ {t['name']}**")
                    st.caption(t["description"][:90] + "..." if len(t["description"]) > 90 else t["description"])
                with col_del:
                    if st.button("🗑", key=f"del_{t['name']}", help=f"Delete {t['name']}"):
                        try:
                            _delete(f"/tools/generated/{t['name']}")
                            st.success(f"Deleted {t['name']}")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
    elif not tools:
        st.caption("No tools loaded yet.")

    st.markdown("---")

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
**Autonomous Execution:**
- **No Micro-Approvals:** The agent executes steps autonomously until your request is completed.
- **Live Thoughts:** See the internal reasoning and tool calls in real time.
- **Deep Exploration:** Reads local files (PDFs, DOCX), searches the web, or generates new tools.
        """)

# ── Main chat area ────────────────────────────────────────────────────
st.markdown("<h2 style='text-align:center;margin-bottom:0'>🤖 AgentOS</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888;margin-top:0'>Autonomous Self-Evolving AI Assistant</p>", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
<div class='welcome-box'>
    <h3 style='color:#7ecfff;margin-top:0'>Welcome to AgentOS</h3>
    <p style='color:#ccc'>Ask me to do tasks on your computer, search the web, compare documents, or connect to services.</p>
    <p style='color:#81c784;font-size:0.92rem'>Try: <em>"Find my resumes in Downloads and compare them in a table"</em></p>
</div>
""", unsafe_allow_html=True)

# Message history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown("<div class='assistant-container'>", unsafe_allow_html=True)
                
                # Show reasoning accordion if steps exist
                steps = msg.get("thought_steps", [])
                if steps:
                    with st.expander(f"💭 Reasoning & Tool Steps ({len(steps)})", expanded=False):
                        for step in steps:
                            stype = step.get("type")
                            if stype == "thought":
                                st.markdown(f"💭 **Thought:** {step.get('content')}")
                            elif stype == "tool_call":
                                st.markdown(f"🔧 **Action:** `{step.get('tool')}` `({step.get('args', '')})`")
                            elif stype == "tool_result":
                                st.markdown(f"📋 **Observation ({step.get('tool')}):** {step.get('summary', '')}")
                            elif stype == "new_tool":
                                st.markdown(f"✨ **Created Tool:** `{step.get('tool')}`")

                # Main message content (rendered as Markdown for tables and code)
                st.markdown(msg["content"])

                # Tool badges
                badges = ""
                for t in msg.get("tools_used", []):
                    badges += f"<span class='tool-badge'>{tool_icon(t)} {t}</span>"
                for t in msg.get("new_tools", []):
                    badges += f"<span class='new-tool-badge'>✨ new: {t}</span>"
                if badges:
                    st.markdown(badges, unsafe_allow_html=True)

                if msg.get("timed_out"):
                    st.markdown("<div class='timeout-warning'>⚠️ Wrapped up early due to time limit.</div>", unsafe_allow_html=True)

                if msg.get("duration_ms"):
                    st.caption(f"⏱ {msg['duration_ms'] / 1000:.1f}s")

                st.markdown("</div>", unsafe_allow_html=True)

# ── Live Execution Stream ─────────────────────────────────────────────
if st.session_state.thinking:
    last_user = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "user"), None
    )
    if last_user:
        thought_steps = []
        final_event = {}

        status_box = st.status("🤖 AgentOS is working on your task...", expanded=True)
        with status_box:
            try:
                for event in _stream_chat(last_user["content"], st.session_state.session_id):
                    etype = event.get("type")
                    if etype == "thought":
                        thought_text = event.get("content", "")
                        st.markdown(f"💭 **Thought:** {thought_text}")
                        thought_steps.append({"type": "thought", "content": thought_text})
                    elif etype == "tool_call":
                        tool = event.get("tool", "")
                        args = event.get("args", "")
                        st.markdown(f"🔧 **Action:** `{tool}` `({args})`")
                        thought_steps.append({"type": "tool_call", "tool": tool, "args": args})
                    elif etype == "tool_result":
                        tool = event.get("tool", "")
                        summary = event.get("summary", "")
                        st.markdown(f"📋 **Observation ({tool}):** {summary}")
                        thought_steps.append({"type": "tool_result", "tool": tool, "summary": summary})
                    elif etype == "new_tool":
                        ntool = event.get("tool", "")
                        st.markdown(f"✨ **Created New Tool:** `{ntool}`")
                        thought_steps.append({"type": "new_tool", "tool": ntool})
                    elif etype == "final":
                        final_event = event

                status_box.update(label="✅ Task Completed!", state="complete", expanded=False)

                if final_event:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_event.get("content", ""),
                        "tools_used": final_event.get("tools_used", []),
                        "new_tools": final_event.get("new_tools", []),
                        "thought_steps": thought_steps,
                        "timed_out": final_event.get("timed_out", False),
                        "duration_ms": final_event.get("duration_ms", 0),
                    })
                    st.session_state.total_tokens += final_event.get("prompt_tokens", 0) + final_event.get("output_tokens", 0)
            except Exception as e:
                status_box.update(label="⚠️ Error during execution", state="error", expanded=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}",
                    "tools_used": [],
                    "new_tools": [],
                    "thought_steps": thought_steps,
                    "timed_out": False,
                    "duration_ms": 0,
                })
            finally:
                st.session_state.thinking = False
                st.rerun()

# ── Input ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([8, 1])
    with col_input:
        user_input = st.text_input(
            label="Message",
            placeholder="Ask a task… I will execute autonomously and show all thoughts and steps.",
            label_visibility="collapsed",
            disabled=st.session_state.thinking,
        )
    with col_send:
        submitted = st.form_submit_button("▶", use_container_width=True, disabled=st.session_state.thinking)

if submitted and user_input.strip():
    if not api_ok:
        st.error("❌ Backend API is not running. Start it with: `uvicorn app.main:app --port 8000`")
    elif not llm_ok:
        st.error("❌ LLM (Ollama) is not running.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip(),
        })
        st.session_state.thinking = True
        st.rerun()
