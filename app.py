"""
XER Schedule Assistant — Streamlit App (Fixed)
================================================
Fixes vs original:
  - Mode map includes "anthropic" key (was missing → ValueError on set_mode)
  - set_mode wrapped in try/except (never crashes silently)
  - HybridLLMClient availability check before calling
  - get_ai_response uses new anti-hallucination system prompt
  - Near-critical threshold uses NEAR_CRITICAL_DAYS from xer_analyzer
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def get_api_keys() -> dict:
    keys = {}
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        val = os.getenv(name, "")
        if not val:
            try:
                val = st.secrets.get(name, "")
            except Exception:
                pass
        if val:
            keys[name] = val
    return keys


def has_cloud_key() -> bool:
    k = get_api_keys()
    return bool(k.get("ANTHROPIC_API_KEY") or k.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

from xer_analyzer import XERAnalyzer, IntentClassifier, NEAR_CRITICAL_DAYS
from xer_context_builder import classify_intent
from hybrid_llm import HybridLLMClient

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xer_complete_extractor import CompleteXERExtractor


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="XER Schedule Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .project-bar {
        background: var(--background-color, #f8f9fa);
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 14px;
    }
    .update-file-item {
        background: #e8f4ea;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 13px;
    }
    .baseline-item {
        background: #e3f2fd;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "baseline_loaded":       False,
        "analyzer":              XERAnalyzer(),
        "messages":              [],
        "project_name":          None,
        "baseline_info":         None,
        "update_files_info":     [],
        "baseline_file_id":      None,
        "processed_update_ids":  set(),
        "llm_mode":              "auto",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# XER file loading
# ---------------------------------------------------------------------------

def load_xer_file(uploaded_file, file_type: str = "baseline") -> dict:
    try:
        raw_bytes = uploaded_file.read()
        suffix    = f"_{file_type}_{uploaded_file.name}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
            tmp.write(raw_bytes)
            temp_path = tmp.name

        try:
            extractor = CompleteXERExtractor(temp_path, file_type)
            extractor.extract_all()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        project_info = extractor.get_project_info()
        data = {
            "project":    project_info,
            "tasks":      extractor.get_all_tasks(),
            "wbs":        extractor.get_wbs_structure(),
            "tables":     extractor.tables,
            "statistics": extractor.extraction_stats,
            "relationships": dict(extractor.table_relationships),
        }
        data_date = str(project_info.get("data_date", ""))[:10]
        return {
            "success":      True,
            "data":         data,
            "project_name": project_info.get("project_name") or uploaded_file.name,
            "data_date":    data_date,
            "file_name":    uploaded_file.name,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# AI response — fixed pipeline
# ---------------------------------------------------------------------------

# Streamlit UI label → HybridLLMClient mode key
_MODE_MAP = {
    "Anthropic (Claude)": "anthropic",   # ← was missing; caused ValueError crash
    "OpenAI (GPT-4o)":    "cloud",
    "Local (Ollama)":     "local",
}

_COMPLEX_KEYWORDS = [
    "trend", "month over month", "each update", "over time", "each month",
    "calculate", "compute", "average across", "sum of all",
    "rank by", "sort by", "top 10", "top 20", "breakdown by",
    "group by", "distribution", "histogram",
]

_ALWAYS_CODE_INTENTS = {"comparison"}


def _needs_code_gen(query: str, intents: list, has_updates: bool) -> bool:
    q = query.lower()
    if any(kw in q for kw in _COMPLEX_KEYWORDS):
        return True
    if has_updates and any(i in intents for i in _ALWAYS_CODE_INTENTS):
        return True
    return False


def _strip_fences(code: str) -> str:
    for fence in ("```python", "```"):
        if fence in code:
            code = code.split(fence)[1].split("```")[0]
    return code.strip()


def _safe_set_mode(client: HybridLLMClient, raw_mode: str) -> None:
    """Map Streamlit label → internal key, then set on client gracefully."""
    resolved = _MODE_MAP.get(raw_mode, raw_mode)
    try:
        client.set_mode(resolved)
    except ValueError:
        import warnings
        warnings.warn(
            f"[app] Invalid LLM mode '{resolved}' (from '{raw_mode}'). "
            "Using 'auto' fallback."
        )
        client.set_mode("auto")


def get_ai_response(user_query: str) -> str:
    """Generate a focused, data-grounded AI response."""
    analyzer: XERAnalyzer = st.session_state.analyzer
    basic_stats = analyzer.get_basic_stats()

    client = HybridLLMClient()
    _safe_set_mode(client, st.session_state.get("llm_mode", "auto"))

    if not client.has_provider():
        return (
            "⚠️ **No LLM provider configured.**\n\n"
            "Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your `.env` file, "
            "or start Ollama locally and reload the app."
        )

    intents        = classify_intent(user_query)
    primary_intent = intents[0] if intents else "general"
    has_updates    = bool(st.session_state.get("update_files_info"))

    # Anti-hallucination system prompt
    system_prompt  = analyzer.get_system_prompt(primary_intent)

    code_result  = None
    code_success = False
    code_error   = None

    # ── Code generation (only for complex aggregation queries) ────────────────
    if _needs_code_gen(user_query, intents, has_updates):
        code_gen_prompt = analyzer.get_code_generation_prompt(user_query, basic_stats)

        for attempt in range(2):
            try:
                prompt_with_error = code_gen_prompt
                if attempt == 1 and code_error:
                    prompt_with_error = (
                        code_gen_prompt
                        + f"\n\nPREVIOUS ATTEMPT FAILED: {code_error}\n"
                        "Fix exactly that error. Return corrected Python only."
                    )
                raw: str = client.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Output ONLY valid Python. "
                                "No markdown fences. No explanations. "
                                "Final line must be: result = <json-serialisable value>"
                            ),
                        },
                        {"role": "user", "content": prompt_with_error},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                )
                code        = _strip_fences(raw)
                exec_result = analyzer.execute_code(code)
                if exec_result["success"]:
                    code_result  = exec_result.get("result")
                    code_success = True
                    code_error   = None
                    break
                else:
                    code_error = exec_result.get("error", "unknown error")
            except Exception as exc:
                code_error = str(exc)

    # ── Build response prompt with real data context ──────────────────────────
    response_prompt = analyzer.get_response_prompt(
        user_query=user_query,
        basic_stats=basic_stats,
        code_result=code_result,
        code_success=code_success,
        code_error=code_error if not code_success else None,
    )

    # ── Generate final answer ─────────────────────────────────────────────────
    try:
        return client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": response_prompt},
            ],
            temperature=0.1,
            max_tokens=2500,
        )
    except Exception as exc:
        try:
            raw_ctx = analyzer._build_targeted_context(user_query)
        except Exception:
            raw_ctx = f"Basic stats: {basic_stats}"
        return (
            f"**LLM error:** {exc}\n\n"
            "Here is the raw schedule data relevant to your question:\n\n"
            f"{raw_ctx}"
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_cost(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:,.1f}K"
    return f"{v:,.0f}"


# ---------------------------------------------------------------------------
# Upload screen
# ---------------------------------------------------------------------------

def show_upload_modal():
    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;min-height:60vh;">
        <div style="text-align:center;max-width:500px;padding:40px;">
            <h1>📊 XER Schedule Assistant</h1>
            <p style="color:#666;margin:20px 0;">
                Upload your Primavera P6 baseline XER file to begin analysis
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Upload Baseline File")
        uploaded_file = st.file_uploader(
            "Select XER file", type=["xer"], key="baseline_upload",
            label_visibility="collapsed",
        )
        if uploaded_file:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.baseline_file_id != file_id:
                with st.spinner("Loading baseline…"):
                    result = load_xer_file(uploaded_file, "baseline")
                    if result["success"]:
                        analyzer: XERAnalyzer = st.session_state.analyzer
                        analyzer.load_baseline(
                            result["data"], result["project_name"], result["data_date"],
                        )
                        st.session_state.project_name  = result["project_name"]
                        st.session_state.baseline_info = {
                            "name":      result["project_name"],
                            "data_date": result["data_date"],
                            "file_name": result["file_name"],
                        }
                        st.session_state.baseline_loaded = True
                        st.session_state.baseline_file_id = file_id
                        st.success("Baseline loaded!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")


# ---------------------------------------------------------------------------
# Insights dashboard
# ---------------------------------------------------------------------------

def show_insights_dashboard(analyzer: XERAnalyzer, basic_stats: dict):
    inner_tabs = st.tabs(["⏱ Delays", "🔴 Critical Path", "💰 Resources & Cost"])

    # ── TAB 1: DELAYS ──────────────────────────────────────────────────────────
    with inner_tabs[0]:
        try:
            di = analyzer.get_delay_insights()
        except Exception as exc:
            st.error(f"Could not compute delay insights: {exc}")
            di = {}

        k1, k2, k3, k4 = st.columns(4)
        slip     = di.get("project_end_slippage_days")
        slip_str = f"{slip:+d} days" if slip is not None else "N/A (no baseline)"
        with k1:
            st.metric("Project End Slippage", slip_str,
                      delta="behind" if (slip or 0) > 0 else "on track",
                      delta_color="inverse")
        with k2:
            st.metric("Slipped Activities",   di.get("slipped_activities_count", 0))
        with k3:
            st.metric("Newly Negative Float", di.get("gained_negative_float_count", 0))
        with k4:
            ns_over = len(di.get("delayed_not_started", []))
            ip_over = len(di.get("delayed_in_progress", []))
            st.metric("Overdue Activities", ns_over + ip_over,
                      help=f"{ns_over} not started · {ip_over} in progress")

        st.markdown("---")
        trend = di.get("float_trend", [])
        if trend:
            import pandas as pd
            import plotly.graph_objects as go
            tdf = pd.DataFrame(trend)
            fig = go.Figure()
            if "critical_count" in tdf.columns:
                fig.add_trace(go.Scatter(
                    x=tdf["data_date"], y=tdf["critical_count"],
                    mode="lines+markers", name="Critical",
                    line=dict(color="#E24B4A", width=2), marker=dict(size=7),
                ))
            if "negative_float_count" in tdf.columns:
                fig.add_trace(go.Scatter(
                    x=tdf["data_date"], y=tdf["negative_float_count"],
                    mode="lines+markers", name="Negative Float",
                    line=dict(color="#EF9F27", width=2, dash="dot"), marker=dict(size=7),
                ))
            if "avg_float_days" in tdf.columns:
                fig.add_trace(go.Scatter(
                    x=tdf["data_date"], y=tdf["avg_float_days"],
                    mode="lines+markers", name="Avg Float (days)",
                    line=dict(color="#1D9E75", width=2), marker=dict(size=7),
                    yaxis="y2",
                ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(title="Activity Count"),
                yaxis2=dict(title="Avg Float (days)", overlaying="y", side="right", showgrid=False),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load multiple update files to see float trend over time.")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        import pandas as pd

        with col_a:
            slipped = di.get("most_slipped_activities", [])
            st.markdown(f"#### Top slipped activities ({len(slipped)})")
            if slipped:
                sdf = pd.DataFrame(slipped)
                dc  = [c for c in ("task_code","task_name","bl_end_date","target_end_date","slippage_days") if c in sdf.columns]
                st.dataframe(sdf[dc].rename(columns={
                    "task_code":"Code","task_name":"Activity",
                    "bl_end_date":"BL Finish","target_end_date":"Current Finish","slippage_days":"Slip (d)",
                }), use_container_width=True, hide_index=True)
            else:
                st.info("No slippage detected vs baseline.")

        with col_b:
            st.markdown("#### Overdue activities")
            over_all = []
            for r in di.get("delayed_not_started", []):
                r = dict(r); r["overdue_type"] = "Not Started"; over_all.append(r)
            for r in di.get("delayed_in_progress", []):
                r = dict(r); r["overdue_type"] = "In Progress"; over_all.append(r)
            if over_all:
                odf = pd.DataFrame(over_all)
                dc  = [c for c in ("task_code","task_name","overdue_type","days_overdue") if c in odf.columns]
                st.dataframe(odf[dc].sort_values("days_overdue", ascending=False).rename(columns={
                    "task_code":"Code","task_name":"Activity",
                    "overdue_type":"Type","days_overdue":"Days Overdue",
                }), use_container_width=True, hide_index=True)
            else:
                st.info("No overdue activities detected.")

    # ── TAB 2: CRITICAL PATH ───────────────────────────────────────────────────
    with inner_tabs[1]:
        try:
            cp = analyzer.get_critical_path_insights()
        except Exception as exc:
            st.error(f"Could not compute critical path insights: {exc}")
            cp = {}

        summ      = cp.get("summary", {})
        nc_thresh = cp.get("near_critical_threshold_days", NEAR_CRITICAL_DAYS)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Critical Activities",
                      f"{summ.get('critical_count',0)} ({summ.get('critical_pct',0)}%)")
        with k2:
            st.metric(f"Near-Critical (≤{nc_thresh}d)", summ.get("near_critical_count", 0))
        with k3:
            st.metric("Total Critical Duration",
                      f"{summ.get('total_critical_duration_days',0):.0f} days")
        with k4:
            changes   = cp.get("critical_path_changes", [])
            newly     = sum(1 for c in changes if c["change"] == "became_critical")
            left_cp   = sum(1 for c in changes if c["change"] == "left_critical")
            st.metric("Path Changes vs BL", f"+{newly} / -{left_cp}")

        warn_cols = st.columns(2)
        with warn_cols[0]:
            if cp.get("constrained_critical"):
                st.warning(f"⚠️ **{len(cp['constrained_critical'])} hard-constrained activities on the critical path**")
        with warn_cols[1]:
            if cp.get("critical_with_lags"):
                st.warning(f"⚠️ **{len(cp['critical_with_lags'])} critical path ties have lags**")

        st.markdown("---")
        col_a, col_b = st.columns([3, 2])
        import plotly.express as px

        with col_a:
            st.markdown("#### Critical activities")
            crit_list = cp.get("critical_activities", [])
            if crit_list:
                cdf = pd.DataFrame(crit_list)
                dc  = [c for c in ("task_code","task_name","status_code","_float_days",
                                   "_dur_days","target_start_date","target_end_date","cstr_type") if c in cdf.columns]
                st.dataframe(cdf[dc].rename(columns={
                    "task_code":"Code","task_name":"Activity","status_code":"Status",
                    "_float_days":"Float(d)","_dur_days":"Dur(d)",
                    "target_start_date":"Start","target_end_date":"Finish","cstr_type":"Constraint",
                }), use_container_width=True, hide_index=True, height=380)
            else:
                st.info("No critical activities found.")

        with col_b:
            st.markdown("#### Critical by WBS")
            by_wbs = cp.get("critical_by_wbs", [])
            if by_wbs:
                wdf   = pd.DataFrame(by_wbs[:15]).copy()
                wdf["label"] = wdf.get("wbs_name", wdf.get("wbs_id","?")).fillna(wdf.get("wbs_id","?"))
                wdf   = wdf.sort_values("critical_count", ascending=True)
                fig   = px.bar(wdf, x="critical_count", y="label", orientation="h",
                               text="critical_count", color="critical_count",
                               color_continuous_scale=[[0,"#F7C1C1"],[1,"#A32D2D"]])
                fig.update_traces(textposition="outside")
                fig.update_layout(height=360, margin=dict(l=0,r=20,t=10,b=0),
                                  showlegend=False, coloraxis_showscale=False,
                                  yaxis=dict(tickfont=dict(size=11)))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No WBS breakdown available.")

            nc_list = cp.get("near_critical_activities", [])
            if nc_list:
                st.markdown("#### Near-critical activities")
                ncdf    = pd.DataFrame(nc_list)
                nc_cols = [c for c in ("task_code","task_name","_float_days","_dur_days") if c in ncdf.columns]
                st.dataframe(ncdf[nc_cols].rename(columns={
                    "task_code":"Code","task_name":"Activity",
                    "_float_days":"Float(d)","_dur_days":"Dur(d)",
                }), use_container_width=True, hide_index=True, height=200)

        if changes:
            st.markdown("---")
            st.markdown("#### Critical path changes vs baseline")
            chdf = pd.DataFrame(changes)
            chdf["change"] = chdf["change"].replace({
                "became_critical": "🔴 Became critical",
                "left_critical":   "🟢 Left critical path",
            })
            dc = [c for c in ("task_code","task_name","change","current_float_days") if c in chdf.columns]
            st.dataframe(chdf[dc].rename(columns={
                "task_code":"Code","task_name":"Activity",
                "change":"Change","current_float_days":"Current Float (d)",
            }), use_container_width=True, hide_index=True)

    # ── TAB 3: RESOURCES & COST ───────────────────────────────────────────────
    with inner_tabs[2]:
        try:
            ri = analyzer.get_resource_insights()
        except Exception as exc:
            st.error(f"Could not compute resource insights: {exc}")
            ri = {}

        cs = ri.get("cost_summary", {})
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: st.metric("Budget (BAC)", _fmt_cost(cs.get("total_budgeted_cost", 0)))
        with k2: st.metric("Actual Cost",  _fmt_cost(cs.get("total_actual_cost",   0)))
        with k3: st.metric("Remaining",    _fmt_cost(cs.get("total_remaining_cost",0)))
        with k4: st.metric("EAC",          _fmt_cost(cs.get("estimate_at_completion",0)))
        with k5:
            cpi = cs.get("cost_performance_index")
            st.metric("CPI", f"{cpi:.3f}" if cpi else "N/A",
                      delta="over budget" if (cpi or 1) < 1 else "on/under budget",
                      delta_color="inverse" if (cpi or 1) < 1 else "normal")

        cv = cs.get("cost_variance", 0)
        if cv and cv != 0:
            msg = f"**Cost variance: {_fmt_cost(cv)}** — {'over' if cv > 0 else 'under'} budget"
            st.error(msg) if cv > 0 else st.success(msg)

        if ri.get("unresourced_count", 0):
            st.warning(f"⚠️ **{ri['unresourced_count']} work activities have no resource assignments**")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            top_res = ri.get("top_resources_by_cost", [])
            st.markdown("#### Top resources by budget")
            if top_res:
                rdf      = pd.DataFrame(top_res)
                name_col = next((c for c in ("rsrc_name","rsrc_short_name","rsrc_id") if c in rdf.columns), None)
                if name_col:
                    show = [name_col] + [c for c in ("total_budget","total_actual","total_remain") if c in rdf.columns]
                    st.dataframe(rdf[show].rename(columns={
                        name_col:"Resource","total_budget":"Budget",
                        "total_actual":"Actual","total_remain":"Remaining",
                    }), use_container_width=True, hide_index=True, height=300)
            else:
                st.info("No resource data available.")

        with col_b:
            unres_list = ri.get("unresourced_activities", [])
            if unres_list:
                st.markdown(f"#### Unresourced activities ({len(unres_list)})")
                udf    = pd.DataFrame(unres_list)
                u_cols = [c for c in ("task_code","task_name","status_code","_dur_days") if c in udf.columns]
                st.dataframe(udf[u_cols].rename(columns={
                    "task_code":"Code","task_name":"Activity",
                    "status_code":"Status","_dur_days":"Dur(d)",
                }), use_container_width=True, hide_index=True, height=300)
            else:
                st.success("No unresourced work activities.")


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

def show_chat_interface():
    analyzer:    XERAnalyzer = st.session_state.analyzer
    basic_stats: dict        = analyzer.get_basic_stats()

    with st.sidebar:
        # ── AI engine ──────────────────────────────────────────────────────────
        st.markdown("### AI Engine")
        providers = []
        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append("Anthropic (Claude)")
        if os.getenv("OPENAI_API_KEY"):
            providers.append("OpenAI (GPT-4o)")
        providers.append("Local (Ollama)")

        llm_choice = st.radio("Select provider:", providers, index=0)

        # Map display label → internal mode and store in session
        new_mode = _MODE_MAP.get(llm_choice, "auto")
        if new_mode != st.session_state.llm_mode:
            st.session_state.llm_mode = new_mode

        if "Anthropic" in llm_choice:
            st.success("☁️ Claude (Anthropic)")
        elif "OpenAI" in llm_choice:
            st.success("☁️ GPT-4o (OpenAI)")
        else:
            st.info("🤖 Local Ollama")

        st.markdown("---")

        # ── Project files ──────────────────────────────────────────────────────
        st.markdown("### Project Files")
        if st.session_state.baseline_info:
            bi = st.session_state.baseline_info
            st.markdown(f"""
            <div class="baseline-item">
                <strong>Baseline</strong><br>
                <small>{bi['name']} | {bi['data_date'] or 'N/A'}</small>
            </div>
            """, unsafe_allow_html=True)

        for i, uf in enumerate(st.session_state.update_files_info):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"""
                <div class="update-file-item">
                    <strong>{uf['name']}</strong><br>
                    <small>{uf['data_date'] or 'N/A'}</small>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                if st.button("✕", key=f"rm_{i}"):
                    st.session_state.update_files_info.pop(i)
                    st.session_state.processed_update_ids.discard(uf.get("file_id"))
                    analyzer.remove_update(i)
                    st.rerun()

        if not st.session_state.update_files_info:
            st.markdown("*No updates loaded*")

        st.markdown("---")
        st.markdown("**Add Update File:**")
        update_file = st.file_uploader(
            "Upload update", type=["xer"], key="update_upload", label_visibility="collapsed",
        )
        if update_file:
            file_id = f"{update_file.name}_{update_file.size}"
            if file_id not in st.session_state.processed_update_ids:
                with st.spinner("Loading update…"):
                    result = load_xer_file(update_file, "update")
                    if result["success"]:
                        analyzer.add_update(result["data"], result["project_name"], result["data_date"])
                        st.session_state.update_files_info.append({
                            "name":      result["project_name"],
                            "data_date": result["data_date"],
                            "file_name": result["file_name"],
                            "file_id":   file_id,
                        })
                        st.session_state.processed_update_ids.add(file_id)
                        st.success(f"Loaded: {result['project_name']}")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")

        st.markdown("---")

        # ── Schedule health quick-view ──────────────────────────────────────────
        st.markdown("### Schedule Health")
        st.markdown(f"- Activities: **{basic_stats.get('total_activities', 0):,}**")
        st.markdown(f"- Critical: **{basic_stats.get('critical_count', 0)}** ({basic_stats.get('critical_pct', 0)}%)")
        st.markdown(f"- Near-Critical: **{basic_stats.get('near_critical_count', 0)}**")
        st.markdown(f"- Neg Float: **{basic_stats.get('negative_float_count', 0)}**")
        st.markdown(f"- Open-Ended: **{basic_stats.get('open_ended_count', 0)}**")
        st.markdown(f"- Long Dur (>20d): **{basic_stats.get('long_duration_count', 0)}**")
        st.markdown(f"- Overdue (NS): **{basic_stats.get('overdue_not_started', 0)}**")
        st.markdown(f"- Overdue (IP): **{basic_stats.get('overdue_in_progress', 0)}**")

        with st.expander("⏱ Delay Insights", expanded=False):
            try:
                di   = analyzer.get_delay_insights()
                slip = di.get("project_end_slippage_days")
                if slip is not None:
                    icon = "🔴" if slip > 0 else ("🟢" if slip < 0 else "🟡")
                    st.markdown(f"{icon} **Slippage: {slip:+d} days**")
                    st.caption(f"BL: `{di.get('baseline_project_end','N/A')}` → Now: `{di.get('current_project_end','N/A')}`")
                else:
                    st.caption("Load baseline + update to see slippage.")
                st.markdown(f"- Slipped: **{di.get('slipped_activities_count',0)}**")
                st.markdown(f"- Newly neg float: **{di.get('gained_negative_float_count',0)}**")
            except Exception as e:
                st.caption(f"Could not compute: {e}")

        with st.expander("🔴 Critical Path", expanded=False):
            try:
                cp   = analyzer.get_critical_path_insights()
                summ = cp.get("summary", {})
                st.markdown(f"- Critical: **{summ.get('critical_count',0)}** ({summ.get('critical_pct',0)}%)")
                st.markdown(f"- Near-critical: **{summ.get('near_critical_count',0)}**")
                st.markdown(f"- Total crit duration: **{summ.get('total_critical_duration_days',0):.0f}d**")
            except Exception as e:
                st.caption(f"Could not compute: {e}")

        with st.expander("💰 Resources & Cost", expanded=False):
            try:
                ri = analyzer.get_resource_insights()
                cs = ri.get("cost_summary", {})
                if cs:
                    b   = cs.get("total_budgeted_cost", 0)
                    a   = cs.get("total_actual_cost",   0)
                    cpi = cs.get("cost_performance_index")
                    icon= "🟢" if (cpi or 0) >= 1 else "🔴"
                    st.markdown(f"- Budget: **{b:,.0f}** | Actual: **{a:,.0f}**")
                    st.markdown(f"- {icon} CPI: **{cpi:.3f}**" if cpi else "- CPI: N/A")
                else:
                    st.caption("No cost data.")
                if ri.get("unresourced_count"):
                    st.markdown(f"- ⚠️ Unresourced: **{ri['unresourced_count']}**")
            except Exception as e:
                st.caption(f"Could not compute: {e}")

        st.markdown("---")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Main content tabs ─────────────────────────────────────────────────────
    main_tab, insights_tab = st.tabs(["💬 Chat", "📊 Insights Dashboard"])

    with main_tab:
        updates_txt = (
            f" | Updates: {len(st.session_state.update_files_info)}"
            if st.session_state.update_files_info else ""
        )
        st.markdown(f"""
        <div class="project-bar">
            <strong>Project:</strong> {st.session_state.project_name} &nbsp;|&nbsp;
            <strong>Activities:</strong> {basic_stats.get('total_activities',0):,} &nbsp;|&nbsp;
            <strong>Period:</strong> {basic_stats.get('project_start','N/A')}
            → {basic_stats.get('project_finish','N/A')}{updates_txt}
        </div>
        """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.chat_message("assistant"):
                with st.spinner("Analysing schedule…"):
                    response = get_ai_response(st.session_state.messages[-1]["content"])
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        if not st.session_state.messages:
            st.markdown(f"""
### Welcome to XER Schedule Assistant

Ask about any aspect of the schedule — the assistant reads actual activity data,
not just aggregate counts, so answers include specific task codes and dates.

**Example questions:**
- *Show me all critical activities with their task codes and finish dates*
- *Which activities have duration over 30 days?*
- *Are there open-ended activities or lags on the critical path?*
- *Give me a DCMA schedule quality assessment*
- *What are the top 10 longest activities?*

**Current schedule health:**
- **{basic_stats.get('critical_count',0)}** critical ({basic_stats.get('critical_pct',0)}%) ·
  **{basic_stats.get('near_critical_count',0)}** near-critical ·
  **{basic_stats.get('negative_float_count',0)}** negative float
- **{basic_stats.get('long_duration_count',0)}** activities > 20 days ·
  **{basic_stats.get('open_ended_count',0)}** open-ended
- **{basic_stats.get('overdue_not_started',0)}** not-started past target start ·
  **{basic_stats.get('overdue_in_progress',0)}** in-progress past target finish
            """)

        if prompt := st.chat_input("Ask about your schedule…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    with insights_tab:
        show_insights_dashboard(analyzer, basic_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not has_cloud_key() and st.session_state.llm_mode not in ("local", "auto"):
        st.warning(
            "**No API key found.** Set ANTHROPIC_API_KEY or OPENAI_API_KEY in `.env`, "
            "or switch to **Local (Ollama)** in the sidebar."
        )

    if not st.session_state.baseline_loaded:
        show_upload_modal()
    else:
        show_chat_interface()


if __name__ == "__main__":
    main()
