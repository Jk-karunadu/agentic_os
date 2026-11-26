"""AgentOS UI — Autonomous Self-Evolving AI Operating System."""
import json
import uuid
import time
import requests
import streamlit as st
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentOS - Self-Evolving AI",
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
.memory-badge {
    display: inline-block;
    background: #2e1a47;
    border: 1px solid #7b1fa2;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #ba68c8;
    margin: 4px 3px 2px 0;
}
.welcome-box {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1a3e 100%);
    border: 1px solid #2d2d54;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 20px auto;
    max-width: 680px;
}
.kpi-card {
    background: #151828;
    border: 1px solid #282d4a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.kpi-val {
    font-size: 1.8rem;
    font-weight: bold;
    color: #64b5f6;
}
.kpi-lbl {
    font-size: 0.85rem;
    color: #90a4ae;
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

def _post(path: str, payload: dict, default=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def _delete(path: str):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"error": "Delete failed"}

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

def render_mermaid(graph_code: str, height: int = 380):
    escaped_code = graph_code.replace("`", "")
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            body {{ background-color: transparent; margin: 0; display: flex; justify-content: center; }}
            .mermaid {{ font-family: monospace, sans-serif; }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{escaped_code}
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    darkMode: true,
                    background: '#0e1117',
                    primaryColor: '#1f77b4',
                    lineColor: '#64b5f6'
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=True)

TOOL_ICONS = {
    "file_search": "📁",
    "read_file": "📄",
    "file_reader": "📂",
    "web_search": "🔍",
    "url_scrape": "🌐",
    "url_read": "🌐",
    "run_python": "🐍",
    "query_memory": "🧠",
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
if "current_dag_task_id" not in st.session_state:
    st.session_state.current_dag_task_id = None

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AgentOS Control")
    api_ok, llm_ok = check_health()
    col1, col2 = st.columns(2)
    with col1:
        dot = "🟢" if api_ok else "🔴"
        st.markdown(f"{dot} **Kernel API**")
    with col2:
        dot = "🟢" if llm_ok else "🔴"
        st.markdown(f"{dot} **Ollama LLM**")

    st.markdown("---")

    if st.button("➕ New Chat Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    if st.session_state.total_tokens:
        st.caption(f"Tokens used: {st.session_state.total_tokens:,}")

    st.markdown("---")
    st.subheader("🛠 Active Tools")
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
        with st.expander(f"✨ Synthesized ({len(generated_tools)})", expanded=True):
            for t in generated_tools:
                col_t, col_del = st.columns([4, 1])
                with col_t:
                    st.markdown(f"**✨ {t['name']}**")
                    st.caption(t["description"][:90] + "..." if len(t["description"]) > 90 else t["description"])
                with col_del:
                    if st.button("🗑", key=f"del_{t['name']}", help=f"Delete {t['name']}"):
                        _delete(f"/tools/generated/{t['name']}")
                        st.rerun()
    elif not tools:
        st.caption("No tools registered.")

# ── Main Tabs ─────────────────────────────────────────────────────────
tab_assistant, tab_dag, tab_evolution = st.tabs([
    "💬 Autonomous Assistant",
    "🕸️ DAG Orchestrator & Execution Graph",
    "🧠 Reflection & Self-Evolution",
])

# =====================================================================
# TAB 1: Autonomous Assistant
# =====================================================================
with tab_assistant:
    if not st.session_state.messages:
        st.markdown("""
        <div class='welcome-box'>
            <h3 style='color:#7ecfff;margin-top:0'>AgentOS Autonomous Runtime</h3>
            <p style='color:#ccc'>Self-evolving assistant powered by local LLM, procedural memory, and live tool synthesis.</p>
            <p style='color:#81c784;font-size:0.92rem'>Try: <em>"Find my resumes in Downloads and compare them in a table"</em> or <em>"Research AI safety and cite sources"</em></p>
        </div>
        """, unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-bubble'>👤 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                with st.container():
                    st.markdown("<div class='assistant-container'>", unsafe_allow_html=True)
                    steps = msg.get("thought_steps", [])
                    if steps:
                        with st.expander(f"💭 Reasoning & Execution Steps ({len(steps)})", expanded=False):
                            for step in steps:
                                stype = step.get("type")
                                if stype == "thought":
                                    st.markdown(f"💭 **Thought:** {step.get('content')}")
                                elif stype == "tool_call":
                                    st.markdown(f"🔧 **Action:** `{step.get('tool')}` `({step.get('args', '')})`")
                                elif stype == "tool_result":
                                    st.markdown(f"📋 **Observation ({step.get('tool')}):** {step.get('summary', '')}")
                                elif stype == "new_tool":
                                    st.markdown(f"✨ **Synthesized Tool:** `{step.get('tool')}`")

                    st.markdown(msg["content"])

                    badges = ""
                    for t in msg.get("tools_used", []):
                        badges += f"<span class='tool-badge'>{tool_icon(t)} {t}</span>"
                    for t in msg.get("new_tools", []):
                        badges += f"<span class='new-tool-badge'>✨ new: {t}</span>"
                    if badges:
                        st.markdown(badges, unsafe_allow_html=True)

                    if msg.get("duration_ms"):
                        st.caption(f"⏱ {msg['duration_ms'] / 1000:.1f}s")
                    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.thinking:
        last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        if last_user:
            thought_steps = []
            final_event = {}
            status_box = st.status("🤖 AgentOS is executing autonomously...", expanded=True)
            with status_box:
                try:
                    for event in _stream_chat(last_user["content"], st.session_state.session_id):
                        etype = event.get("type")
                        if etype == "thought":
                            t = event.get("content", "")
                            st.markdown(f"💭 **Thought:** {t}")
                            thought_steps.append({"type": "thought", "content": t})
                        elif etype == "tool_call":
                            tl, args = event.get("tool", ""), event.get("args", "")
                            st.markdown(f"🔧 **Action:** `{tl}` `({args})`")
                            thought_steps.append({"type": "tool_call", "tool": tl, "args": args})
                        elif etype == "tool_result":
                            tl, summ = event.get("tool", ""), event.get("summary", "")
                            st.markdown(f"📋 **Observation ({tl}):** {summ}")
                            thought_steps.append({"type": "tool_result", "tool": tl, "summary": summ})
                        elif etype == "new_tool":
                            ntl = event.get("tool", "")
                            st.markdown(f"✨ **Synthesized Tool:** `{ntl}`")
                            thought_steps.append({"type": "new_tool", "tool": ntl})
                        elif etype == "final":
                            final_event = event

                    status_box.update(label="✅ Completed", state="complete", expanded=False)
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
                    status_box.update(label=f"⚠️ Error: {e}", state="error", expanded=True)
                finally:
                    st.session_state.thinking = False
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        c_in, c_btn = st.columns([9, 1])
        with c_in:
            u_in = st.text_input("Message", placeholder="Enter your request...", label_visibility="collapsed", disabled=st.session_state.thinking)
        with c_btn:
            sub = st.form_submit_button("▶", use_container_width=True, disabled=st.session_state.thinking)

    if sub and u_in.strip():
        if api_ok and llm_ok:
            st.session_state.messages.append({"role": "user", "content": u_in.strip()})
            st.session_state.thinking = True
            st.rerun()
        else:
            st.error("Backend services are not responding.")

# =====================================================================
# TAB 2: DAG Orchestrator & Execution Graph
# =====================================================================
with tab_dag:
    st.markdown("### 🕸️ Directed Acyclic Graph (DAG) Orchestrator")
    st.caption("Plan decomposition, deterministic verification, and dynamic graph rewiring upon verification failure.")

    col_task_input, col_task_preview = st.columns([5, 5])
    with col_task_input:
        st.subheader("Submit Task")
        demo_choice = st.selectbox(
            "Load Sample Research Task",
            [
                "Custom",
                "AI Alignment & Safety (from ArXiv)",
                "Transformer Architecture (Wikipedia)",
            ]
        )
        if demo_choice == "AI Alignment & Safety (from ArXiv)":
            default_obj = "Summarise key approaches in AI safety (RLHF, Constitutional AI, reward hacking). Cite sources."
            default_e1_title = "Concrete Problems in AI Safety"
            default_e1_url = "https://arxiv.org/abs/1606.06565"
            default_e1_text = "We present five concrete problems in AI safety: avoiding negative side effects, avoiding reward hacking, scalable oversight, safe exploration, and robustness to distributional shift."
            default_e2_title = "Constitutional AI: Harmlessness from AI Feedback"
            default_e2_url = "https://arxiv.org/abs/2212.08073"
            default_e2_text = "The method is called Constitutional AI (CAI), because it uses a set of principles to make judgments about which outputs are desirable. Resulting model is both more helpful and less harmful."
        elif demo_choice == "Transformer Architecture (Wikipedia)":
            default_obj = "Explain the transformer deep learning architecture and its self-attention mechanism. Cite source."
            default_e1_title = "Transformer Architecture"
            default_e1_url = "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"
            default_e1_text = "A transformer is a deep learning architecture that relies on the self-attention mechanism."
            default_e2_title = ""
            default_e2_url = ""
            default_e2_text = ""
        else:
            default_obj, default_e1_title, default_e1_url, default_e1_text = "", "", "", ""
            default_e2_title, default_e2_url, default_e2_text = "", "", ""

        obj_input = st.text_area("Objective", value=default_obj, height=80)
        st.markdown("**Evidence Source 1**")
        e1_t = st.text_input("Title 1", value=default_e1_title)
        e1_u = st.text_input("URL 1", value=default_e1_url)
        e1_x = st.text_area("Excerpt 1", value=default_e1_text, height=60)

        with st.expander("Evidence Source 2 (Optional)"):
            e2_t = st.text_input("Title 2", value=default_e2_title)
            e2_u = st.text_input("URL 2", value=default_e2_url)
            e2_x = st.text_area("Excerpt 2", value=default_e2_text, height=60)

        if st.button("🚀 Launch Orchestrated DAG Run", use_container_width=True):
            if not obj_input.strip() or not e1_u.strip():
                st.error("Please enter an objective and at least one evidence source URL.")
            else:
                evidence = [{"title": e1_t, "url": e1_u, "excerpt": e1_x}]
                if e2_u.strip():
                    evidence.append({"title": e2_t, "url": e2_u, "excerpt": e2_x})

                # 1. Create task
                t_res = _post("/tasks", {"objective": obj_input, "evidence": evidence})
                if "id" in t_res:
                    task_id = t_res["id"]
                    st.session_state.current_dag_task_id = task_id
                    # 2. Trigger run-orchestrated
                    _post(f"/tasks/{task_id}/run-orchestrated", {})
                    st.success(f"Task `{task_id[:8]}...` started!")
                    st.rerun()
                else:
                    st.error(f"Task creation failed: {t_res}")

    with col_task_preview:
        st.subheader("Live DAG Execution Graph")
        current_id = st.session_state.current_dag_task_id
        if not current_id:
            st.info("Launch a task or select an existing task to view its live execution graph.")
        else:
            st.caption(f"Monitoring Task: `{current_id}`")
            status_data = _get(f"/tasks/{current_id}/run-orchestrated/status")

            if not status_data:
                st.warning("⏳ Waiting for backend response... Checking again.")
                time.sleep(2)
                st.rerun()
            elif status_data.get("status") == "error":
                st.error(f"❌ Execution error: {status_data.get('error')}")
                if status_data.get("final_answer"):
                    st.markdown("**Error Details:**")
                    st.markdown(status_data.get("final_answer"))
            else:
                # Extract plan nodes
                plan_dict = status_data.get("plan")
                nodes = plan_dict.get("nodes", []) if plan_dict else []

                # Build Mermaid diagram
                lines = ["graph TD"]
                lines.append("  classDef completed fill:#1b5e20,stroke:#81c784,stroke-width:2px,color:#fff;")
                lines.append("  classDef running fill:#0d47a1,stroke:#64b5f6,stroke-width:2px,color:#fff;")
                lines.append("  classDef failed fill:#b71c1c,stroke:#e57373,stroke-width:2px,color:#fff;")
                lines.append("  classDef ready fill:#e65100,stroke:#ffb74d,stroke-width:2px,color:#fff;")
                lines.append("  classDef pending fill:#263238,stroke:#78909c,stroke-width:1px,color:#b0bec5;")

                for n in nodes:
                    nid = n["id"]
                    ntype = n.get("type", "step")
                    nstatus = n.get("status", "pending")
                    lines.append(f'  {nid}["{nid} ({ntype})"]:::{nstatus}')
                    for dep in n.get("depends_on", []):
                        lines.append(f"  {dep} --> {nid}")

                if len(nodes) > 0:
                    render_mermaid("\n".join(lines), height=320)
                else:
                    st.caption("Initializing plan...")

                # If completed
                if "verification_passed" in status_data:
                    passed = status_data.get("verification_passed")
                    badge = "✅ Passed Verification" if passed else "❌ Failed Verification"
                    st.markdown(f"#### Status: {badge}")
                    st.caption(f"Tokens: {status_data.get('total_tokens')} | Duration: {status_data.get('total_duration_ms')}ms")

                    # Show applied lessons & credit assignment
                    applied = status_data.get("applied_lessons", [])
                    rewarded = status_data.get("lessons_rewarded", [])
                    penalized = status_data.get("lessons_penalized", [])
                    if applied:
                        st.markdown("**Procedural Lessons Applied:**")
                        for l in applied:
                            st.markdown(f"- 🧠 `{l}`")
                        if rewarded:
                            st.markdown(f"🏆 *Rewarded {len(rewarded)} lesson(s) for success!*")
                        if penalized:
                            st.markdown(f"⚠️ *Penalized {len(penalized)} lesson(s) for failure.*")

                    # Verification issues / feedback
                    v_out = status_data.get("verification_output")
                    if v_out and not passed:
                        with st.expander("🔍 Verification Feedback & Errors", expanded=True):
                            st.warning(v_out)
                    elif v_out and passed:
                        st.caption(f"Verification: {v_out}")

                    # Final synthesized answer
                    final_ans = status_data.get("final_answer")
                    if final_ans:
                        st.markdown("**Synthesized Output:**")
                        st.markdown(final_ans)

                    # Node-by-node outputs
                    node_res = status_data.get("node_results", {})
                    if node_res:
                        with st.expander("📋 Node Execution Details", expanded=False):
                            for nid, nr in node_res.items():
                                st.markdown(f"**Node `{nid}`**")
                                if isinstance(nr, dict):
                                    if nr.get("error"):
                                        st.error(f"Error: {nr['error']}")
                                    if nr.get("output"):
                                        st.text_area(f"Output ({nid})", nr["output"], height=100, disabled=True, key=f"dag_out_{nid}")
                else:
                    time.sleep(2)
                    st.rerun()

# =====================================================================
# TAB 3: Reflection & Self-Evolution Database
# =====================================================================
with tab_evolution:
    st.markdown("### 🧠 Self-Evolution & Reflection Engine")
    st.caption("Inspect procedural lessons, Bayesian success scores, failure records, and synthesized tool health.")

    stats = _get("/reflection/stats", {})
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>{stats.get('total_lessons', 0)}</div><div class='kpi-lbl'>Procedural Lessons</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>{stats.get('avg_effectiveness', 0.0):.2f}</div><div class='kpi-lbl'>Avg Effectiveness Score</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>{stats.get('total_failures', 0)}</div><div class='kpi-lbl'>Historical Failures Caught</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='kpi-card'><div class='kpi-val'>{stats.get('generated_tools_count', 0)}</div><div class='kpi-lbl'>Synthesized Tools</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_prune, col_refresh = st.columns([8, 2])
        with col_refresh:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        with col_prune:
            if st.button("🧹 Prune Ineffective Lessons (< 0.20 score)", use_container_width=True):
                res = _post("/reflection/prune", {"threshold": 0.20})
                st.success(f"Pruned {res.get('pruned_lessons', 0)} ineffective lessons.")
                st.rerun()

        st.markdown("---")
        st.subheader("📚 Active Procedural Lessons (Knowledge Base)")
        lessons = stats.get("lessons", [])
        if lessons:
            for l in lessons:
                eff = l.get("effectiveness_score", 0.5)
                color = "#81c784" if eff >= 0.6 else ("#ffb74d" if eff >= 0.3 else "#e57373")
                st.markdown(f"""
                <div style='background:#121226;border-left:4px solid {color};padding:10px 16px;border-radius:6px;margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;'>
                        <strong style='color:#64b5f6'>Trigger: {l.get('trigger')}</strong>
                        <span style='color:{color};font-weight:bold'>Score: {eff:.2f} (✅ {l.get('success_count', 0)} / ❌ {l.get('failure_count', 0)})</span>
                    </div>
                    <div style='color:#ccc;margin-top:4px;'>{l.get('lesson')}</div>
                    <div style='color:#81c784;font-size:0.9rem;margin-top:4px;'><strong>Action:</strong> {l.get('recommended_action')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No procedural lessons recorded yet. Run tasks in the DAG Orchestrator to observe the AI fail, reflect, and learn.")

        st.markdown("---")
        st.subheader("🚨 Failure Log & Verification Catches")
        failures = _get("/failures", [])
        if failures:
            for f in reversed(failures[-10:]):
                status_txt = "✅ Repaired" if f.get("repair_succeeded") else ("⚠️ Repair Failed" if f.get("repair_attempted") else "Logged")
                st.markdown(f"- **[{f.get('error_type')}]** `{f.get('failing_node') or 'root'}` — {f.get('error_detail')[:140]}... *({status_txt})*")
        else:
            st.caption("No failures recorded.")
    else:
        st.info("Unable to retrieve reflection stats. Ensure the backend API is running.")
