import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_openai_key() -> str:
    """Return the OpenAI API key from env or Streamlit secrets."""
    key = os.getenv('OPENAI_API_KEY', '')
    if not key:
        try:
            key = st.secrets.get('OPENAI_API_KEY', '')
        except Exception:
            pass
    return key


# ---------------------------------------------------------------------------
# Lazy imports (only what's available)
# ---------------------------------------------------------------------------
from xer_analyzer import XERAnalyzer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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
        margin-bottom: 20px;
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
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        'baseline_loaded': False,
        'analyzer': XERAnalyzer(),
        'messages': [],
        'project_name': None,
        'baseline_info': None,
        'update_files_info': [],
        'baseline_file_id': None,
        'processed_update_ids': set(),
        'llm_mode': 'cloud',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# XER file loading
# ---------------------------------------------------------------------------

def load_xer_file(uploaded_file, file_type: str = 'baseline') -> dict:
    """Parse an uploaded XER file and return structured data."""
    try:
        raw_bytes = uploaded_file.read()

        # Use a proper temp file so concurrent uploads don't collide
        suffix = f"_{file_type}_{uploaded_file.name}"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, mode='wb'
        ) as tmp:
            tmp.write(raw_bytes)
            temp_path = tmp.name

        try:
            extractor = CompleteXERExtractor(temp_path, file_type)
            extractor.extract_all()
        finally:
            # Always clean up, even if extraction raises
            try:
                os.remove(temp_path)
            except OSError:
                pass

        project_info = extractor.get_project_info()

        data = {
            'project': project_info,
            'tasks': extractor.get_all_tasks(),
            'wbs': extractor.get_wbs_structure(),
            'tables': extractor.tables,
            'statistics': extractor.extraction_stats,
            'relationships': dict(extractor.table_relationships),
        }

        data_date = str(project_info.get('data_date', ''))[:10]

        return {
            'success': True,
            'data': data,
            'project_name': project_info.get('project_name') or uploaded_file.name,
            'data_date': data_date,
            'file_name': uploaded_file.name,
        }

    except Exception as exc:
        return {'success': False, 'error': str(exc)}


# ---------------------------------------------------------------------------
# AI response
# ---------------------------------------------------------------------------

def get_ai_response(user_query: str) -> str:
    """Generate an AI response for the given schedule query."""
    analyzer: XERAnalyzer = st.session_state.analyzer
    basic_stats = analyzer.get_basic_stats()

    from hybrid_llm import HybridLLMClient
    client = HybridLLMClient()
    client.set_mode(st.session_state.llm_mode)

    # ---- Step 1: Generate Python analysis code ----
    code_gen_prompt = analyzer.get_code_generation_prompt(user_query, basic_stats)
    code_success = False
    code_result = None
    code_error = None

    try:
        generated_code: str = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Python code generator for Primavera P6 schedule analysis. "
                        "Generate ONLY valid Python code that sets result = ... at the end. "
                        "No markdown fences, no explanations, no comments — just executable code."
                    ),
                },
                {"role": "user", "content": code_gen_prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
            return_model=False,
        )

        # Strip markdown fences if the model disobeyed instructions
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0]
        generated_code = generated_code.strip()

        print("=" * 60)
        print("GENERATED CODE:")
        print(generated_code)
        print("=" * 60)

        exec_result = analyzer.execute_code(generated_code)
        code_success = exec_result['success']
        code_result = exec_result.get('result') if code_success else None
        code_error = exec_result.get('error') if not code_success else None

        if code_success:
            print("EXECUTION SUCCESS:", code_result)
        else:
            print("EXECUTION FAILED:", code_error)

    except Exception as exc:
        code_error = str(exc)
        print("CODE GENERATION ERROR:", code_error)

    # ---- Step 2: Generate human-readable response ----
    response_prompt = analyzer.get_response_prompt(
        user_query, basic_stats, code_result, code_success, code_error
    )

    try:
        final_response: str = client.chat(
            messages=[
                {"role": "system", "content": analyzer.get_system_prompt()},
                {"role": "user", "content": response_prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
            return_model=False,
        )
        return final_response

    except Exception as exc:
        return (
            f"**Error generating response:** {exc}\n\n"
            f"**Project:** {basic_stats.get('data_source', 'N/A')} | "
            f"**Data Date:** {basic_stats.get('data_date', 'N/A')}\n"
            f"- Total Activities: {basic_stats.get('total_activities', 'N/A')}\n"
            f"- Critical: {basic_stats.get('critical_count', 'N/A')} "
            f"({basic_stats.get('critical_pct', 'N/A')}%)\n"
            f"- Negative Float: {basic_stats.get('negative_float_count', 'N/A')}\n"
            f"- Open-Ended: {basic_stats.get('open_ended_count', 'N/A')}"
        )


# ---------------------------------------------------------------------------
# Upload screen
# ---------------------------------------------------------------------------

def show_upload_modal():
    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;min-height:60vh;">
        <div style="text-align:center;max-width:500px;padding:40px;">
            <h1>XER Schedule Assistant</h1>
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
            "Select XER file",
            type=['xer'],
            key='baseline_upload',
            label_visibility='collapsed',
        )

        if uploaded_file:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.baseline_file_id != file_id:
                with st.spinner("Loading baseline..."):
                    result = load_xer_file(uploaded_file, 'baseline')
                    if result['success']:
                        analyzer: XERAnalyzer = st.session_state.analyzer
                        analyzer.load_baseline(
                            result['data'],
                            result['project_name'],
                            result['data_date'],
                        )
                        st.session_state.project_name = result['project_name']
                        st.session_state.baseline_info = {
                            'name': result['project_name'],
                            'data_date': result['data_date'],
                            'file_name': result['file_name'],
                        }
                        st.session_state.baseline_loaded = True
                        st.session_state.baseline_file_id = file_id
                        st.success("Baseline loaded!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

def _fmt_cost(v) -> str:
    """Format a cost number compactly."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:,.1f}K"
    return f"{v:,.0f}"


def _color_sign(v, positive_bad: bool = True) -> str:
    """Return a colored markdown string for a signed number."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    prefix = "🔴" if (v > 0) == positive_bad else ("🟢" if v != 0 else "🟡")
    return f"{prefix} {v:+,.0f}"


def show_insights_dashboard(analyzer: XERAnalyzer, basic_stats: dict):
    """
    Full-page insights dashboard with three sections:
    Delays | Critical Path | Resources & Cost
    """
    inner_tabs = st.tabs(["⏱ Delays", "🔴 Critical Path", "💰 Resources & Cost"])

    # =========================================================
    # TAB 1 — DELAYS
    # =========================================================
    with inner_tabs[0]:
        try:
            di = analyzer.get_delay_insights()
        except Exception as exc:
            st.error(f"Could not compute delay insights: {exc}")
            di = {}

        # --- KPI row ---
        k1, k2, k3, k4 = st.columns(4)
        slip = di.get('project_end_slippage_days')
        slip_str = f"{slip:+d} days" if slip is not None else "N/A (no baseline)"
        slip_delta = None if slip is None else (
            "behind schedule" if slip > 0 else ("ahead of schedule" if slip < 0 else "on schedule")
        )
        with k1:
            st.metric(
                "Project End Slippage",
                slip_str,
                delta=slip_delta,
                delta_color="inverse",
            )
        with k2:
            st.metric("Slipped Activities", di.get('slipped_activities_count', 0))
        with k3:
            st.metric("Newly Negative Float", di.get('gained_negative_float_count', 0))
        with k4:
            ns_over = len(di.get('delayed_not_started', []))
            ip_over = len(di.get('delayed_in_progress', []))
            st.metric("Overdue Activities", ns_over + ip_over,
                      help=f"{ns_over} not started · {ip_over} in progress")

        st.markdown("---")

        # --- Float trend chart ---
        trend = di.get('float_trend', [])
        if trend:
            st.markdown("#### Float trend across updates")
            import pandas as pd
            import plotly.graph_objects as go
            tdf = pd.DataFrame(trend)
            fig = go.Figure()
            if 'critical_count' in tdf.columns:
                fig.add_trace(go.Scatter(
                    x=tdf['data_date'], y=tdf['critical_count'],
                    mode='lines+markers', name='Critical',
                    line=dict(color='#E24B4A', width=2),
                    marker=dict(size=7),
                    hovertemplate='%{x}<br>Critical: %{y}<extra></extra>',
                ))
            if 'negative_float_count' in tdf.columns:
                fig.add_trace(go.Scatter(
                    x=tdf['data_date'], y=tdf['negative_float_count'],
                    mode='lines+markers', name='Negative Float',
                    line=dict(color='#EF9F27', width=2, dash='dot'),
                    marker=dict(size=7),
                    hovertemplate='%{x}<br>Negative Float: %{y}<extra></extra>',
                ))
            if 'avg_float_days' in tdf.columns:
                fig.add_trace(go.Scatter(
                    x=tdf['data_date'], y=tdf['avg_float_days'],
                    mode='lines+markers', name='Avg Float (days)',
                    line=dict(color='#1D9E75', width=2),
                    marker=dict(size=7),
                    yaxis='y2',
                    hovertemplate='%{x}<br>Avg Float: %{y:.1f}d<extra></extra>',
                ))
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation='h', y=1.1),
                xaxis=dict(title='Data Date'),
                yaxis=dict(title='Activity Count'),
                yaxis2=dict(title='Avg Float (days)', overlaying='y', side='right', showgrid=False),
                hovermode='x unified',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load multiple update files to see the float trend over time.")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        # --- Most slipped activities ---
        with col_a:
            slipped = di.get('most_slipped_activities', [])
            st.markdown(f"#### Top slipped activities ({len(slipped)})")
            if slipped:
                import pandas as pd
                sdf = pd.DataFrame(slipped)
                display_cols = [c for c in (
                    'task_code', 'task_name', 'bl_end_date', 'target_end_date', 'slippage_days'
                ) if c in sdf.columns]
                st.dataframe(
                    sdf[display_cols].rename(columns={
                        'task_code': 'Code', 'task_name': 'Activity',
                        'bl_end_date': 'Baseline Finish', 'target_end_date': 'Current Finish',
                        'slippage_days': 'Slippage (d)',
                    }),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No slippage detected vs baseline.")

        # --- Overdue activities ---
        with col_b:
            st.markdown("#### Overdue activities")
            import pandas as pd
            ns_list = di.get('delayed_not_started', [])
            ip_list = di.get('delayed_in_progress', [])
            over_all = []
            for r in ns_list:
                r = dict(r); r['overdue_type'] = 'Not Started'; over_all.append(r)
            for r in ip_list:
                r = dict(r); r['overdue_type'] = 'In Progress'; over_all.append(r)

            if over_all:
                odf = pd.DataFrame(over_all)
                display_cols = [c for c in (
                    'task_code', 'task_name', 'overdue_type',
                    'target_start_date', 'target_end_date', 'days_overdue'
                ) if c in odf.columns]
                odf_sorted = odf[display_cols].sort_values('days_overdue', ascending=False) \
                    if 'days_overdue' in odf.columns else odf[display_cols]
                st.dataframe(
                    odf_sorted.rename(columns={
                        'task_code': 'Code', 'task_name': 'Activity',
                        'overdue_type': 'Type', 'days_overdue': 'Days Overdue',
                        'target_start_date': 'Target Start', 'target_end_date': 'Target Finish',
                    }),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No overdue activities detected.")

    # =========================================================
    # TAB 2 — CRITICAL PATH
    # =========================================================
    with inner_tabs[1]:
        try:
            cp = analyzer.get_critical_path_insights()
        except Exception as exc:
            st.error(f"Could not compute critical path insights: {exc}")
            cp = {}

        summ = cp.get('summary', {})
        nc_thresh = cp.get('near_critical_threshold_days', 8)

        # --- KPI row ---
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Critical Activities",
                      f"{summ.get('critical_count', 0)} ({summ.get('critical_pct', 0)}%)")
        with k2:
            st.metric(f"Near-Critical (≤{nc_thresh}d)", summ.get('near_critical_count', 0))
        with k3:
            st.metric("Total Critical Duration",
                      f"{summ.get('total_critical_duration_days', 0):.0f} days")
        with k4:
            changes = cp.get('critical_path_changes', [])
            newly = sum(1 for c in changes if c['change'] == 'became_critical')
            left_cp = sum(1 for c in changes if c['change'] == 'left_critical')
            st.metric("Path Changes vs Baseline", f"+{newly} / -{left_cp}",
                      help=f"{newly} newly critical · {left_cp} recovered from critical path")

        # --- Warnings row ---
        import pandas as pd
        warn_cols = st.columns(2)
        with warn_cols[0]:
            const_crit = cp.get('constrained_critical', [])
            if const_crit:
                st.warning(
                    f"⚠️ **{len(const_crit)} hard-constrained activities on the critical path** "
                    f"— constraints may be artificially holding float at zero."
                )
        with warn_cols[1]:
            lag_crit = cp.get('critical_with_lags', [])
            if lag_crit:
                st.warning(
                    f"⚠️ **{len(lag_crit)} critical path logic ties have lags** "
                    f"— lags reduce schedule predictability."
                )

        st.markdown("---")
        col_a, col_b = st.columns([3, 2])

        # --- Full critical activity table ---
        with col_a:
            st.markdown("#### Critical activities")
            crit_list = cp.get('critical_activities', [])
            if crit_list:
                cdf = pd.DataFrame(crit_list)
                display_cols = [c for c in (
                    'task_code', 'task_name', 'status_code',
                    '_float_days', '_dur_days', '_complete_pct',
                    'target_start_date', 'target_end_date', 'cstr_type',
                ) if c in cdf.columns]
                rename = {
                    'task_code': 'Code', 'task_name': 'Activity',
                    'status_code': 'Status', '_float_days': 'Float (d)',
                    '_dur_days': 'Duration (d)', '_complete_pct': '% Complete',
                    'target_start_date': 'Start', 'target_end_date': 'Finish',
                    'cstr_type': 'Constraint',
                }
                st.dataframe(
                    cdf[display_cols].rename(columns=rename),
                    use_container_width=True, hide_index=True, height=380,
                )
            else:
                st.info("No critical activities found.")

        # --- Critical by WBS bar chart ---
        with col_b:
            st.markdown("#### Critical count by WBS")
            by_wbs = cp.get('critical_by_wbs', [])
            if by_wbs:
                import plotly.express as px
                wdf = pd.DataFrame(by_wbs[:15]).copy()
                wdf['label'] = wdf.get('wbs_name', wdf.get('wbs_id', '?')).fillna(wdf.get('wbs_id', '?'))
                wdf = wdf.sort_values('critical_count', ascending=True)
                fig = px.bar(
                    wdf, x='critical_count', y='label',
                    orientation='h',
                    text='critical_count',
                    color='critical_count',
                    color_continuous_scale=[[0, '#F7C1C1'], [1, '#A32D2D']],
                    labels={'critical_count': 'Critical Activities', 'label': ''},
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=360, margin=dict(l=0, r=20, t=10, b=0),
                    showlegend=False, coloraxis_showscale=False,
                    yaxis=dict(tickfont=dict(size=11)),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No WBS breakdown available.")

            # Near-critical table
            st.markdown("#### Near-critical activities")
            nc_list = cp.get('near_critical_activities', [])
            if nc_list:
                ncdf = pd.DataFrame(nc_list)
                nc_cols = [c for c in ('task_code', 'task_name', '_float_days', '_dur_days') if c in ncdf.columns]
                st.dataframe(
                    ncdf[nc_cols].rename(columns={
                        'task_code': 'Code', 'task_name': 'Activity',
                        '_float_days': 'Float (d)', '_dur_days': 'Duration (d)',
                    }),
                    use_container_width=True, hide_index=True, height=200,
                )
            else:
                st.info("No near-critical activities.")

        # --- Path changes ---
        if changes:
            st.markdown("---")
            st.markdown("#### Critical path changes vs baseline")
            chdf = pd.DataFrame(changes)
            chdf['change'] = chdf['change'].replace({
                'became_critical': '🔴 Became critical',
                'left_critical': '🟢 Left critical path',
            })
            display_cols = [c for c in (
                'task_code', 'task_name', 'change', 'current_float_days'
            ) if c in chdf.columns]
            st.dataframe(
                chdf[display_cols].rename(columns={
                    'task_code': 'Code', 'task_name': 'Activity',
                    'change': 'Change', 'current_float_days': 'Current Float (d)',
                }),
                use_container_width=True, hide_index=True,
            )

    # =========================================================
    # TAB 3 — RESOURCES & COST
    # =========================================================
    with inner_tabs[2]:
        try:
            ri = analyzer.get_resource_insights()
        except Exception as exc:
            st.error(f"Could not compute resource insights: {exc}")
            ri = {}

        import pandas as pd
        cs = ri.get('cost_summary', {})

        # --- KPI row ---
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Budget (BAC)", _fmt_cost(cs.get('total_budgeted_cost', 0)))
        with k2:
            st.metric("Actual Cost (AC)", _fmt_cost(cs.get('total_actual_cost', 0)))
        with k3:
            st.metric("Remaining", _fmt_cost(cs.get('total_remaining_cost', 0)))
        with k4:
            st.metric("EAC", _fmt_cost(cs.get('estimate_at_completion', 0)))
        with k5:
            cpi = cs.get('cost_performance_index')
            cpi_str = f"{cpi:.3f}" if cpi is not None else "N/A"
            cpi_delta = "on budget" if cpi is None else (
                "over budget" if cpi < 1.0 else "under budget"
            )
            st.metric("CPI", cpi_str, delta=cpi_delta,
                      delta_color="normal" if (cpi or 1) >= 1 else "inverse")

        # --- Cost variance alert ---
        cv = cs.get('cost_variance', 0)
        pct_spent = cs.get('budget_spent_pct')
        if cv and cv != 0:
            msg = (
                f"**Cost variance: {_fmt_cost(cv)}** — "
                f"{'over' if cv > 0 else 'under'} budget by {abs(cv / max(cs.get('total_budgeted_cost', 1), 1)) * 100:.1f}%"
            )
            if pct_spent is not None:
                msg += f" ({pct_spent:.1f}% of budget spent)"
            if cv > 0:
                st.error(msg)
            else:
                st.success(msg)

        unres = ri.get('unresourced_count', 0)
        if unres:
            st.warning(
                f"⚠️ **{unres} work activities have no resource assignments** "
                f"— these cannot be cost-tracked."
            )

        st.markdown("---")
        col_a, col_b = st.columns(2)

        # --- Top resources table ---
        with col_a:
            st.markdown("#### Top resources by budget")
            top_res = ri.get('top_resources_by_cost', [])
            if top_res:
                rdf = pd.DataFrame(top_res)
                name_col = next((c for c in ('rsrc_name', 'rsrc_short_name', 'rsrc_id') if c in rdf.columns), None)
                num_cols = [c for c in ('total_budget', 'total_actual', 'cost_variance', 'utilization_pct') if c in rdf.columns]
                if name_col:
                    rdf = rdf.rename(columns={
                        name_col: 'Resource',
                        'total_budget': 'Budget',
                        'total_actual': 'Actual',
                        'cost_variance': 'Variance',
                        'utilization_pct': 'Util %',
                    })
                    show = ['Resource'] + [c for c in ('Budget', 'Actual', 'Variance', 'Util %') if c in rdf.columns]
                    st.dataframe(rdf[show], use_container_width=True, hide_index=True, height=320)
            else:
                st.info("No resource assignment data available.")

            # Resource type breakdown
            by_type = ri.get('resource_type_breakdown', [])
            if by_type:
                st.markdown("#### Budget vs actual by resource type")
                import plotly.graph_objects as go
                tdf = pd.DataFrame(by_type)
                type_labels = tdf.get('rsrc_type', tdf.index).tolist()
                fig = go.Figure()
                if 'budget' in tdf.columns:
                    fig.add_trace(go.Bar(
                        name='Budget', x=type_labels, y=tdf['budget'],
                        marker_color='#378ADD',
                        hovertemplate='%{x}<br>Budget: %{y:,.0f}<extra></extra>',
                    ))
                if 'actual' in tdf.columns:
                    fig.add_trace(go.Bar(
                        name='Actual', x=type_labels, y=tdf['actual'],
                        marker_color='#E24B4A',
                        hovertemplate='%{x}<br>Actual: %{y:,.0f}<extra></extra>',
                    ))
                fig.update_layout(
                    barmode='group', height=260,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation='h', y=1.1),
                    yaxis=dict(title='Cost'),
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Over-budget activities ---
        with col_b:
            st.markdown("#### Over-budget activities")
            over_b = ri.get('overbudget_activities', [])
            if over_b:
                obdf = pd.DataFrame(over_b)
                ob_cols = [c for c in (
                    'task_code', 'task_name', '_budget', '_actual', '_variance'
                ) if c in obdf.columns]
                st.dataframe(
                    obdf[ob_cols].rename(columns={
                        'task_code': 'Code', 'task_name': 'Activity',
                        '_budget': 'Budget', '_actual': 'Actual', '_variance': 'Variance',
                    }),
                    use_container_width=True, hide_index=True, height=220,
                )
            else:
                st.success("No activities are over budget.")

            # Unresourced list
            unres_list = ri.get('unresourced_activities', [])
            if unres_list:
                st.markdown(f"#### Unresourced work activities ({len(unres_list)})")
                udf = pd.DataFrame(unres_list)
                u_cols = [c for c in (
                    'task_code', 'task_name', 'status_code', '_dur_days'
                ) if c in udf.columns]
                st.dataframe(
                    udf[u_cols].rename(columns={
                        'task_code': 'Code', 'task_name': 'Activity',
                        'status_code': 'Status', '_dur_days': 'Duration (d)',
                    }),
                    use_container_width=True, hide_index=True, height=200,
                )

        # --- Cost by WBS ---
        cost_wbs = ri.get('cost_by_wbs', [])
        if cost_wbs:
            st.markdown("---")
            st.markdown("#### Cost by WBS")
            import plotly.graph_objects as go
            cwdf = pd.DataFrame(cost_wbs).head(15)
            label_col = next((c for c in ('wbs_name', 'wbs_id') if c in cwdf.columns), None)
            if label_col:
                labels = cwdf[label_col].fillna('Unknown').tolist()
                fig = go.Figure()
                color_map = {'budget': '#378ADD', 'actual': '#E24B4A', 'remain': '#1D9E75'}
                name_map = {'budget': 'Budget', 'actual': 'Actual', 'remain': 'Remaining'}
                for col in ('budget', 'actual', 'remain'):
                    if col in cwdf.columns:
                        fig.add_trace(go.Bar(
                            name=name_map[col],
                            x=labels,
                            y=cwdf[col],
                            marker_color=color_map[col],
                            hovertemplate=f'%{{x}}<br>{name_map[col]}: %{{y:,.0f}}<extra></extra>',
                        ))
                fig.update_layout(
                    barmode='group', height=340,
                    margin=dict(l=0, r=0, t=10, b=80),
                    legend=dict(orientation='h', y=1.05),
                    xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
                    yaxis=dict(title='Cost'),
                    hovermode='x unified',
                )
                st.plotly_chart(fig, use_container_width=True)


def show_chat_interface():
    analyzer: XERAnalyzer = st.session_state.analyzer
    basic_stats = analyzer.get_basic_stats()

    with st.sidebar:
        # ---- AI settings ----
        st.markdown("### AI Settings")
        llm_choice = st.radio(
            "Select AI Engine:",
            options=["Cloud AI (OpenAI)", "Local AI (Ollama)"],
            index=0 if st.session_state.llm_mode == 'cloud' else 1,
            help="Switch between cloud-based GPT and your local Ollama instance.",
        )
        new_mode = 'cloud' if "Cloud" in llm_choice else 'local'

        if new_mode == 'cloud' and not get_openai_key():
            st.warning("⚠️ No OPENAI_API_KEY found. Switch to Local AI or add your key.")
        elif new_mode == 'local':
            st.info("🤖 Running on local Ollama")
        else:
            st.success("☁️ Running on OpenAI")

        if new_mode != st.session_state.llm_mode:
            st.session_state.llm_mode = new_mode

        st.markdown("---")

        # ---- Loaded files ----
        st.markdown("### Project Files")

        if st.session_state.baseline_info:
            bi = st.session_state.baseline_info
            st.markdown(f"""
            <div class="baseline-item">
                <strong>Baseline</strong><br>
                <small>{bi['name']} | {bi['data_date'] or 'N/A'}</small>
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.update_files_info:
            for i, uf in enumerate(st.session_state.update_files_info):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"""
                    <div class="update-file-item">
                        <strong>{uf['name']}</strong><br>
                        <small>Data Date: {uf['data_date'] or 'N/A'}</small>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    if st.button("✕", key=f"remove_{i}"):
                        st.session_state.update_files_info.pop(i)
                        st.session_state.processed_update_ids.discard(uf['file_id'])
                        analyzer.remove_update(i)
                        st.rerun()
        else:
            st.markdown("*No updates loaded*")

        st.markdown("---")
        st.markdown("**Add Update File:**")
        update_file = st.file_uploader(
            "Upload update",
            type=['xer'],
            key='update_upload',
            label_visibility='collapsed',
        )

        if update_file:
            file_id = f"{update_file.name}_{update_file.size}"
            if file_id not in st.session_state.processed_update_ids:
                with st.spinner("Loading update..."):
                    result = load_xer_file(update_file, 'update')
                    if result['success']:
                        analyzer.add_update(
                            result['data'],
                            result['project_name'],
                            result['data_date'],
                        )
                        st.session_state.update_files_info.append({
                            'name': result['project_name'],
                            'data_date': result['data_date'],
                            'file_name': result['file_name'],
                            'file_id': file_id,
                        })
                        st.session_state.processed_update_ids.add(file_id)
                        st.success(f"Loaded: {result['project_name']} ({result['data_date']})")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")

        st.markdown("---")

        # ---- Schedule Health (quick metrics) ----
        st.markdown("### Schedule Health")
        st.markdown(f"- Activities: **{basic_stats.get('total_activities', 0)}**")
        st.markdown(
            f"- Critical: **{basic_stats.get('critical_count', 0)}** "
            f"({basic_stats.get('critical_pct', 0)}%)"
        )
        st.markdown(f"- Near-Critical: **{basic_stats.get('near_critical_count', 0)}**")
        st.markdown(f"- Neg Float: **{basic_stats.get('negative_float_count', 0)}**")
        st.markdown(f"- Open-Ended: **{basic_stats.get('open_ended_count', 0)}**")
        st.markdown(f"- Long Dur (>20d): **{basic_stats.get('long_duration_count', 0)}**")
        st.markdown(f"- Overdue (NS): **{basic_stats.get('overdue_not_started', 0)}**")
        st.markdown(f"- Overdue (IP): **{basic_stats.get('overdue_in_progress', 0)}**")

        # ---- Delay Insights ----
        with st.expander("⏱ Delay Insights", expanded=False):
            try:
                di = analyzer.get_delay_insights()
                slip = di.get('project_end_slippage_days')
                if slip is not None:
                    color = "🔴" if slip > 0 else ("🟢" if slip < 0 else "🟡")
                    st.markdown(
                        f"{color} **Project end slippage:** {slip:+d} days  \n"
                        f"Baseline: `{di.get('baseline_project_end', 'N/A')}` → "
                        f"Current: `{di.get('current_project_end', 'N/A')}`"
                    )
                else:
                    st.caption("Load a baseline + update to see slippage.")
                st.markdown(f"- Slipped activities: **{di.get('slipped_activities_count', 0)}**")
                st.markdown(f"- Newly negative float: **{di.get('gained_negative_float_count', 0)}**")
                st.markdown(
                    f"- Overdue NS: **{len(di.get('delayed_not_started', []))}**"
                    f" | Overdue IP: **{len(di.get('delayed_in_progress', []))}**"
                )
                trend = di.get('float_trend', [])
                if trend:
                    st.markdown("**Float trend:**")
                    for t in trend:
                        st.caption(
                            f"`{t['data_date']}` crit={t['critical_count']} "
                            f"neg={t['negative_float_count']} avg={t['avg_float_days']}d"
                        )
            except Exception as exc:
                st.caption(f"Could not compute delay insights: {exc}")

        # ---- Critical Path Insights ----
        with st.expander("🔴 Critical Path", expanded=False):
            try:
                cp = analyzer.get_critical_path_insights()
                summ = cp.get('summary', {})
                st.markdown(
                    f"- Critical: **{summ.get('critical_count', 0)}** "
                    f"({summ.get('critical_pct', 0)}%)"
                )
                st.markdown(
                    f"- Near-critical (≤{cp.get('near_critical_threshold_days', 8)}d): "
                    f"**{summ.get('near_critical_count', 0)}**"
                )
                st.markdown(
                    f"- Total critical duration: **{summ.get('total_critical_duration_days', 0)} days**"
                )
                changes = cp.get('critical_path_changes', [])
                if changes:
                    newly = sum(1 for c in changes if c['change'] == 'became_critical')
                    left_cp = sum(1 for c in changes if c['change'] == 'left_critical')
                    st.markdown(f"- Path changes: **{newly} newly**, {left_cp} recovered")
                if cp.get('constrained_critical'):
                    st.markdown(f"- ⚠️ Constrained on CP: **{len(cp['constrained_critical'])}**")
                if cp.get('critical_with_lags'):
                    st.markdown(f"- CP ties with lags: **{len(cp['critical_with_lags'])}**")
                by_wbs = cp.get('critical_by_wbs', [])
                if by_wbs:
                    st.markdown("**Top WBS areas:**")
                    for w in by_wbs[:4]:
                        name = w.get('wbs_name') or w.get('wbs_id', '?')
                        st.caption(f"`{name}` — {w.get('critical_count', 0)} critical")
            except Exception as exc:
                st.caption(f"Could not compute CP insights: {exc}")

        # ---- Resource Insights ----
        with st.expander("💰 Resources & Cost", expanded=False):
            try:
                ri = analyzer.get_resource_insights()
                cs = ri.get('cost_summary', {})
                if cs:
                    budget = cs.get('total_budgeted_cost', 0)
                    actual = cs.get('total_actual_cost', 0)
                    cpi = cs.get('cost_performance_index')
                    eac = cs.get('estimate_at_completion', 0)
                    cv = cs.get('cost_variance', 0)
                    cpi_icon = "🟢" if (cpi or 0) >= 1 else "🔴"
                    cpi_str = f"{cpi:.3f}" if cpi is not None else "N/A"
                    st.markdown(
                        f"- Budget: **{budget:,.0f}** | Actual: **{actual:,.0f}**\n"
                        f"- EAC: **{eac:,.0f}** | CV: **{cv:+,.0f}**\n"
                        f"- {cpi_icon} CPI: **{cpi_str}**"
                    )
                else:
                    st.caption("No cost data available.")
                unres = ri.get('unresourced_count', 0)
                if unres:
                    st.markdown(f"- ⚠️ Unresourced activities: **{unres}**")
                over_b = ri.get('overbudget_activities', [])
                if over_b:
                    st.markdown(f"- Over-budget activities: **{len(over_b)}**")
                by_type = ri.get('resource_type_breakdown', [])
                if by_type:
                    st.markdown("**By resource type:**")
                    for t in by_type[:4]:
                        st.caption(
                            f"`{t.get('rsrc_type', '?')}` — {t.get('budget', 0):,.0f}"
                        )
            except Exception as exc:
                st.caption(f"Could not compute resource insights: {exc}")

        st.markdown("---")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ---- Main content: two tabs ----
    main_tab, insights_tab = st.tabs(["💬 Chat", "📊 Insights Dashboard"])

    with main_tab:
        # ---- Top project bar ----
        updates_text = (
            f" | Updates: {len(st.session_state.update_files_info)}"
            if st.session_state.update_files_info
            else ""
        )
        st.markdown(f"""
        <div class="project-bar">
            <strong>Project:</strong> {st.session_state.project_name} &nbsp;|&nbsp;
            <strong>Activities:</strong> {basic_stats.get('total_activities', 0)} &nbsp;|&nbsp;
            <strong>Period:</strong> {basic_stats.get('project_start', 'N/A')}
            → {basic_stats.get('project_finish', 'N/A')}{updates_text}
        </div>
        """, unsafe_allow_html=True)

        # ---- Chat history ----
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # If the last message is from the user, generate a response
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
        ):
            with st.chat_message("assistant"):
                with st.spinner("Analyzing schedule..."):
                    response = get_ai_response(st.session_state.messages[-1]["content"])
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Welcome / empty state
        if not st.session_state.messages:
            st.markdown(f"""
### Welcome to XER Schedule Assistant

**Schedule Quality Analysis:**
- Long duration activities, open-ended tasks, dangling activities
- Critical path analysis, negative float identification
- Constraint analysis, relationship checks, lag review

**Comparisons:**
- Baseline vs update | Changes between monthly updates | Progress tracking

**Current Schedule Health:**
- **{basic_stats.get('critical_count', 0)}** critical activities ({basic_stats.get('critical_pct', 0)}%)
- **{basic_stats.get('near_critical_count', 0)}** near-critical activities
- **{basic_stats.get('negative_float_count', 0)}** activities with negative float
- **{basic_stats.get('long_duration_count', 0)}** activities > 20 days duration
- **{basic_stats.get('open_ended_count', 0)}** open-ended activities
- **{basic_stats.get('overdue_not_started', 0)}** not-started activities past their target start
- **{basic_stats.get('overdue_in_progress', 0)}** in-progress activities past their target finish

Ask me anything, or switch to the **📊 Insights Dashboard** tab for a visual breakdown.
            """)

        # ---- Chat input ----
        if prompt := st.chat_input("Ask about your schedule..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    with insights_tab:
        show_insights_dashboard(analyzer, basic_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Block startup only if Cloud AI is selected AND no key is configured.
    # Ollama mode does not need an OpenAI key.
    if st.session_state.llm_mode == 'cloud' and not get_openai_key():
        st.warning(
            "**OPENAI_API_KEY not found.** "
            "Set it in `.env` (local) or Streamlit secrets (cloud), "
            "or switch to **Local AI (Ollama)** in the sidebar."
        )
        # Don't st.stop() — allow the user to switch to Ollama without a key

    if not st.session_state.baseline_loaded:
        show_upload_modal()
    else:
        show_chat_interface()


if __name__ == "__main__":
    main()
