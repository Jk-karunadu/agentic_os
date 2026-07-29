import json
from pathlib import Path

import streamlit as st
import client

# ── Demo Datasets ────────────────────────────────────────────────────
# Pre-scraped evidence so users can test end-to-end without 403 errors.

DEMO_DATASETS = {
    "🤖 AI Safety Research": {
        "objective": "Summarise the current state of AI alignment research, focusing on key approaches like RLHF, constitutional AI, and interpretability. Cite the supplied sources.",
        "evidence": [
            {
                "Title": "Concrete Problems in AI Safety",
                "URL": "https://arxiv.org/abs/1606.06565",
                "Excerpt": (
                    "As machine learning systems become more broadly deployed, there is increasing interest in "
                    "making sure these systems operate safely and reliably. We present five concrete problems in "
                    "AI safety that are ready for research today: avoiding negative side effects, avoiding reward "
                    "hacking, scalable oversight, safe exploration, and robustness to distributional shift. We "
                    "describe each problem in detail along with example environments and discuss approaches that "
                    "have been proposed or could be developed. Reward hacking occurs when an agent finds an "
                    "unintended way to maximise its reward signal without fulfilling the designer's true "
                    "objective. Scalable oversight addresses the challenge of supervising AI systems that operate "
                    "in environments that are too complex for humans to fully evaluate."
                ),
            },
            {
                "Title": "Constitutional AI: Harmlessness from AI Feedback",
                "URL": "https://arxiv.org/abs/2212.08073",
                "Excerpt": (
                    "We experiment with methods for training a harmless AI assistant through self-improvement, "
                    "without any human labels identifying harmful outputs. The method is called Constitutional AI "
                    "(CAI), because it uses a set of principles (a 'constitution') to make judgments about which "
                    "outputs are desirable. The process involves first generating responses using a helpful-only "
                    "AI assistant, then asking the model to critique and revise its own response according to a "
                    "set of constitutional principles such as 'Choose the response that is least likely to be "
                    "harmful or toxic.' We find that the resulting model is both more helpful and less harmful "
                    "than models trained with RLHF alone. CAI reduces the need for human feedback labels for "
                    "harmlessness by a factor of roughly ten while achieving comparable or superior results."
                ),
            },
        ],
    },
    "🌍 Climate Change Policy": {
        "objective": "Analyse the effectiveness of carbon pricing mechanisms versus direct regulation for reducing industrial greenhouse gas emissions. Reference the supplied evidence.",
        "evidence": [
            {
                "Title": "Carbon Pricing Dashboard - World Bank",
                "URL": "https://carbonpricingdashboard.worldbank.org/",
                "Excerpt": (
                    "Carbon pricing instruments now cover approximately 23 percent of global greenhouse gas "
                    "emissions, up from 5 percent a decade ago. There are currently 73 carbon pricing initiatives "
                    "in operation or scheduled for implementation worldwide, consisting of 37 emissions trading "
                    "systems (ETS) and 36 carbon taxes. The EU Emissions Trading System remains the largest "
                    "carbon market, covering about 40 percent of EU emissions. Prices vary widely, from less than "
                    "US$1 per tonne in some developing economies to over US$140 per tonne in Switzerland. "
                    "Research suggests that a carbon price of US$50-100 per tonne by 2030 is needed to meet "
                    "Paris Agreement goals, yet fewer than 5 percent of covered emissions are priced at levels "
                    "consistent with this target."
                ),
            },
            {
                "Title": "Comparing Carbon Pricing and Direct Regulation - Nature Climate Change",
                "URL": "https://www.nature.com/articles/s41558-021-01128-0",
                "Excerpt": (
                    "A meta-analysis of 37 empirical studies covering carbon pricing schemes in 21 jurisdictions "
                    "found that carbon pricing reduced emissions by 5-21 percent on average compared to "
                    "business-as-usual scenarios. However, the effectiveness depends heavily on price levels, "
                    "scope of coverage, and complementary policies. Direct regulation, such as emission "
                    "performance standards and technology mandates, was found to produce faster initial emissions "
                    "reductions in targeted sectors but at higher economic cost per tonne abated. The study "
                    "concludes that a hybrid approach combining carbon pricing with targeted regulations for "
                    "hard-to-abate sectors delivers the most cost-effective pathway to deep decarbonisation. "
                    "Revenue recycling from carbon pricing can offset regressive impacts on lower-income "
                    "households when designed with equity considerations."
                ),
            },
        ],
    },
    "⚛️ Quantum Computing Advances": {
        "objective": "Write a technical briefing on the latest advances in quantum error correction and fault-tolerant quantum computing. Use only the supplied sources.",
        "evidence": [
            {
                "Title": "Suppressing quantum errors by scaling a surface code logical qubit - Google",
                "URL": "https://www.nature.com/articles/s41586-022-05434-1",
                "Excerpt": (
                    "Google Quantum AI demonstrated that increasing the size of a quantum error-correcting code "
                    "can reduce the logical error rate, a key milestone for fault-tolerant quantum computing. "
                    "Using their Sycamore processor, the team implemented surface codes with distances 3 and 5, "
                    "showing that the larger code suppressed errors by a factor of approximately 4× compared to "
                    "the smaller code. This constitutes a 'below threshold' demonstration, where adding more "
                    "physical qubits makes the logical qubit more reliable rather than less. The surface code "
                    "distance-5 logical qubit achieved a logical error rate per round of approximately 2.9×10⁻³, "
                    "requiring 49 physical data qubits and 48 measurement qubits. The team estimates that a "
                    "distance-17 surface code could achieve error rates below 10⁻⁶ per cycle."
                ),
            },
            {
                "Title": "Quantum error correction beyond break-even - Yale/AWS",
                "URL": "https://www.nature.com/articles/s41586-023-06927-3",
                "Excerpt": (
                    "Researchers at Yale and AWS demonstrated a quantum error-corrected memory that surpasses "
                    "break-even — the point where error correction extends the lifetime of quantum information "
                    "beyond what any individual component can achieve. Using a bosonic approach with a "
                    "Gottesman-Kitaev-Preskill (GKP) code encoded in a superconducting cavity, they achieved "
                    "a logical qubit lifetime of 2.3 milliseconds, compared to 0.2 milliseconds for the best "
                    "uncorrected physical qubit in the same system. This represents an 11.5× improvement in "
                    "coherence time. The result is significant because it shows that the overhead of error "
                    "correction does not negate its benefits, addressing a long-standing concern. The GKP "
                    "approach is complementary to surface codes and may enable more hardware-efficient "
                    "fault-tolerant architectures."
                ),
            },
        ],
    },
}


# ── Load custom CSS (path-safe) ──────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "style.css"
    try:
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


st.set_page_config(page_title="AgentOS", page_icon="🤖", layout="wide")
load_css()

# ── Header ───────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; color: #00FF9D;'>AgentOS Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8;'>A Self-Improving Multi-Agent Operating System</p>", unsafe_allow_html=True)

# ── Health indicator ─────────────────────────────────────────────────
health = client.check_health()
col_h1, col_h2, col_h3 = st.columns([6, 1, 1])
with col_h2:
    if health.get("api"):
        st.markdown("🟢 **API**")
    else:
        st.markdown("🔴 **API**")
with col_h3:
    if health.get("ollama"):
        st.markdown("🟢 **LLM**")
    else:
        st.markdown("🔴 **LLM**")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🚀 Mission Control", "🧠 Memory Core", "🛠️ Reflection Database"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: MISSION CONTROL
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Deploy Agents")

    # ── Demo Dataset Loader ──────────────────────────────────────────
    with st.container(border=True):
        st.markdown("##### 📦 Quick-Start Demo Datasets")
        st.caption("Pre-loaded evidence so you can test the full pipeline without external URLs (no 403 errors).")

        demo_col1, demo_col2 = st.columns([3, 1])
        with demo_col1:
            selected_demo = st.selectbox(
                "Choose a dataset",
                options=list(DEMO_DATASETS.keys()),
                label_visibility="collapsed",
            )
        with demo_col2:
            load_demo = st.button("⚡ Load Dataset", use_container_width=True)

        if load_demo and selected_demo:
            demo = DEMO_DATASETS[selected_demo]
            st.session_state["current_objective"] = demo["objective"]
            st.session_state["current_evidence"] = demo["evidence"]

    # ── Task Input ───────────────────────────────────────────────────
    with st.container(border=True):
        if "current_objective" not in st.session_state:
            st.session_state["current_objective"] = ""
        if "current_evidence" not in st.session_state:
            st.session_state["current_evidence"] = [{"Title": "", "URL": "", "Excerpt": ""}]

        objective = st.text_area(
            "Objective",
            key="current_objective",
            placeholder="Enter the task objective here...",
            height=100,
        )

        st.markdown("##### Evidence Sources (Optional)")
        st.caption("Provide initial context or source links for the agents to start with.")

        evidence_df = st.data_editor(
            st.session_state["current_evidence"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("Title", width="medium"),
                "URL": st.column_config.TextColumn("URL", width="medium"),
                "Excerpt": st.column_config.TextColumn("Excerpt (Content)", width="large"),
            },
        )
        # Save any user edits back to session state so they persist when buttons are clicked
        st.session_state["current_evidence"] = evidence_df

        if st.button("🚀 Run Orchestrated Task", type="primary", use_container_width=True):
            if not objective or len(objective.strip()) < 10:
                st.error("Please enter an objective (at least 10 characters).")
            else:
                try:
                    # Convert dataframe rows to evidence list, filtering empty rows
                    evidence = []
                    for row in evidence_df:
                        title = str(row.get("Title") or "").strip()
                        url = str(row.get("URL") or "").strip()
                        excerpt = str(row.get("Excerpt") or "").strip()
                        if title and excerpt:
                            if not url:
                                url = f"https://local-source-{len(evidence)+1}.com"
                            evidence.append({
                                "title": title,
                                "url": url,
                                "excerpt": excerpt,
                            })

                    with st.spinner("Connecting to AgentOS API..."):
                        task_data = client.create_task(objective, evidence)
                        task_id = task_data["id"]

                    st.success(f"Task created successfully. ID: `{task_id}`")

                    with st.status("Executing Multi-Agent DAG...", expanded=True) as status:
                        st.write("Initializing PlanScheduler and dispatching workers...")
                        
                        live_plan_ui = st.empty()
                        
                        def update_ui(plan_data):
                            nodes = plan_data.get("nodes", [])
                            if not nodes:
                                return
                            md = "#### Live Plan Execution\n"
                            for n in nodes:
                                icon = "⏳"
                                if n["status"] == "running": icon = "🔄"
                                elif n["status"] == "completed": icon = "✅"
                                elif n["status"] == "failed": icon = "❌"
                                md += f"* {icon} **{n['type']}** ({n['id']}): {n['objective']}\n"
                                
                                events = n.get("events", [])
                                if events:
                                    for e in events:
                                        role = e.get("role")
                                        content = e.get("content", "")
                                        
                                        if role == "system":
                                            md += f"  * ⚙️ *{content.strip()}*\n"
                                        elif role == "assistant":
                                            # Indent and blockquote the LLM's full thought process
                                            formatted_content = "\n".join(f"    > {line}" for line in content.split("\n") if line.strip())
                                            md += f"  * 🧠 **LLM Thought Process & Output:**\n{formatted_content}\n"
                                        elif role == "tool_call":
                                            md += f"  * 🛠️ *{content.strip()}*\n"
                                        elif role == "tool_result":
                                            # Only truncate tool results if they are massively long to prevent UI lag, but keep it very generous
                                            if len(content) > 1500:
                                                content = content[:1500] + "... (truncated)"
                                            formatted_content = "\n".join(f"    > {line}" for line in content.split("\n") if line.strip())
                                            md += f"  * 📄 **Tool Result:**\n{formatted_content}\n"
                                        
                            live_plan_ui.markdown(md, unsafe_allow_html=True)

                        try:
                            result = client.run_orchestrated(task_id, status_callback=update_ui)

                            status.update(label="✅ Execution Complete!", state="complete", expanded=False)

                            st.markdown("### Final Answer")
                            if result.get("verification_passed"):
                                st.success("✅ **Verification Passed!** This answer meets all deterministic constraints.")
                            else:
                                st.warning("⚠️ **Verification did not pass.** Procedural memory lesson stored for self-improvement.")

                            st.info(result.get("final_answer") or "No answer produced — the LLM may not have generated a draft node output.")

                            st.markdown("### Execution Trace")
                            col1, col2 = st.columns(2)
                            duration_ms = result.get("total_duration_ms", 0)
                            col1.metric("Latency", f"{duration_ms / 1000:.1f}s")
                            col2.metric("Tokens Used", f"{result.get('total_tokens', 0):,}")

                            node_results = result.get("node_results", {})
                            if node_results:
                                with st.expander("View Node Results", expanded=False):
                                    for node_id, node_res in node_results.items():
                                        st.markdown(f"**`{node_id}`** — {node_res.get('duration_ms', 0)}ms, {node_res.get('tokens_used', 0)} tokens")
                                        if node_res.get("error"):
                                            st.error(f"Error: {node_res['error']}")
                                        else:
                                            st.text(node_res.get("output", "")[:500])
                                        st.markdown("---")

                        except Exception as e:
                            status.update(label="❌ Execution Failed", state="error", expanded=True)
                            error_msg = str(e)
                            if "Read timed out" in error_msg or "ReadTimeout" in error_msg:
                                st.error("**TIMEOUT_DEBUG_CODE:** The local LLM is taking too long to generate the responses. Please check the backend terminal for progress. The task is likely still running in the background!")
                            elif "503" in error_msg:
                                st.error("**Ollama is not running.** Please start Ollama and pull the required model (`ollama pull qwen3:8b`).")
                            elif "Connection" in error_msg:
                                st.error("**Cannot connect to the API server.** Please start the backend: `uvicorn app.main:app --reload`")
                            else:
                                st.error(f"Error during orchestration: {e}")

                except Exception as e:
                    st.error(f"Error creating task: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: MEMORY CORE
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Query Semantic Memory")

    col_q, col_t = st.columns([3, 1])
    with col_q:
        search_query = st.text_input("Search Query", placeholder="What are you looking for?")
    with col_t:
        memory_type = st.selectbox("Filter by Type", ["All", "episodic", "procedural", "semantic"])

    if st.button("🔍 Search Memory"):
        if not search_query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching Vector Database..."):
                filter_val = None if memory_type == "All" else memory_type
                try:
                    results = client.search_memory(search_query, filter_val)
                    if results:
                        for idx, item in enumerate(results):
                            with st.container(border=True):
                                mcol1, mcol2, mcol3 = st.columns([1, 1, 4])
                                mcol1.markdown(f"**Type:** `{item.get('type')}`")
                                mcol2.markdown(f"**Importance:** `{item.get('importance', 0):.2f}`")
                                mcol3.markdown(f"**Success:** `{item.get('success_score', 0):.2f}`")
                                st.write(item.get("text"))
                    else:
                        st.info("No relevant memories found. Run some tasks first to populate the memory store!")
                except Exception as e:
                    error_msg = str(e)
                    if "Connection" in error_msg:
                        st.error("Cannot connect to the API. Is the backend running?")
                    else:
                        st.error(f"Failed to query memory: {e}")

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: REFLECTION DATABASE
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Failure Logs & Procedural Lessons")
    st.write("When agents fail a verification step, they reflect on their mistakes and store a lesson. This tab tracks those events.")

    if st.button("🔄 Refresh Database"):
        try:
            failures = client.get_failures()
            if failures:
                for failure in failures:
                    with st.expander(f"Task: {failure.get('task_id', 'N/A')} | Type: {failure.get('error_type', 'unknown')}"):
                        st.markdown("**Error Detail:**")
                        st.code(failure.get("error_detail", "No details available."))

                        # Parse lesson from lesson_json if available
                        lesson_json = failure.get("lesson_json")
                        if lesson_json:
                            try:
                                lesson = json.loads(lesson_json)
                                st.markdown("**Lesson Learned (Procedural Memory):**")
                                st.success(f"**{lesson.get('lesson', 'N/A')}**\n\n_Action:_ {lesson.get('recommended_action', 'N/A')}")
                            except (json.JSONDecodeError, TypeError):
                                st.info("Lesson data could not be parsed.")
                        else:
                            st.info("No lesson data stored for this failure.")

                        # Repair status
                        if failure.get("repair_attempted"):
                            if failure.get("repair_succeeded"):
                                st.success("✅ Repair Attempted — Succeeded")
                            else:
                                st.error("❌ Repair Attempted — Failed")
                        else:
                            st.info("🔲 No repair attempted yet.")

                        st.caption(f"Model: {failure.get('model', 'N/A')}")
            else:
                st.info("No failures logged yet! The agents are doing perfectly. 🎉")
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg:
                st.error("Cannot connect to the API. Is the backend running?")
            else:
                st.error(f"Failed to fetch failures: {e}")
