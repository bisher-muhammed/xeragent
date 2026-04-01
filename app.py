"""
XER Schedule AI Assistant
LLM-powered chat interface for Primavera P6 schedule analysis.
"""

import os
import sys
import json
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hybrid_llm import HybridLLMClient
from xer_analyzer import XERAnalyzer
from xer_complete_extractor import CompleteXERExtractor


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_FILE_SIZE_MB   = 200
NEAR_CRITICAL_DAYS = 5


# =============================================================================
# PAGE CONFIG
# =============================================================================

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
        background: #f8f9fa; padding: 10px 20px;
        border-radius: 8px; margin-bottom: 20px; font-size: 14px;
    }
    .update-file-item {
        background: #e8f4ea; padding: 8px 12px;
        border-radius: 6px; margin: 5px 0; font-size: 13px;
    }
    .baseline-item {
        background: #e3f2fd; padding: 8px 12px;
        border-radius: 6px; margin: 5px 0; font-size: 13px;
    }
    .insight-card {
        background: #fff8e1; padding: 8px 12px;
        border-radius: 6px; margin: 4px 0; font-size: 12px;
        border-left: 3px solid #f9a825;
    }
    .insight-card.critical {
        background: #fce4ec;
        border-left-color: #c62828;
    }
    .insight-card.ok {
        background: #e8f5e9;
        border-left-color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

def get_openai_key() -> Optional[str]:
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        try:
            key = st.secrets.get('OPENAI_API_KEY')
        except Exception:
            pass
    return key or None


def _get_hybrid_client() -> HybridLLMClient:
    if 'hybrid_client' not in st.session_state:
        st.session_state.hybrid_client = HybridLLMClient(
            local_model='gemma:2b',
            cloud_model='gpt-4o-mini',
            openai_key=get_openai_key(),
        )
    return st.session_state.hybrid_client


# =============================================================================
# SESSION STATE
# =============================================================================

_DEFAULTS = {
    'baseline_loaded':      False,
    'analyzer':             None,
    'messages':             [],
    'project_name':         None,
    'baseline_info':        None,
    'update_files_info':    [],
    'baseline_file_id':     None,
    'processed_update_ids': [],
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.analyzer is None:
    st.session_state.analyzer = XERAnalyzer()


# =============================================================================
# FILE LOADING
# =============================================================================

def load_xer_file(uploaded_file, file_type: str = 'baseline') -> Dict:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return {'success': False,
                'error': f"File is {size_mb:.1f} MB — limit is {MAX_FILE_SIZE_MB} MB."}

    temp_path = f"temp_{file_type}_{uploaded_file.name}"
    try:
        raw_bytes = uploaded_file.read()
        with open(temp_path, 'wb') as f:
            f.write(raw_bytes)

        extractor = CompleteXERExtractor(
            temp_path,
            file_type=file_type,
            near_critical_days=NEAR_CRITICAL_DAYS,
        )
        extractor.extract_all()

        project_info = extractor.get_project_info()
        data_date    = (project_info.get('data_date') or '')[:10]

        data = {
            'project': project_info,
            'tasks':   extractor.get_all_tasks(),
            'wbs':     extractor.structured.get('wbs', {}),
            'summary': extractor.structured.get('summary', {}),
            'tables':  extractor.tables,
        }

        if extractor.parsing_errors:
            print(f"[{uploaded_file.name}] Warnings: {extractor.parsing_errors[:3]}")

        return {
            'success':      True,
            'data':         data,
            'project_name': project_info.get('project_name', uploaded_file.name),
            'data_date':    data_date,
            'file_name':    uploaded_file.name,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# =============================================================================
# AI RESPONSE
# =============================================================================

def _extract_code(raw: str) -> str:
    if '```python' in raw:
        raw = raw.split('```python')[1].split('```')[0]
    elif '```' in raw:
        raw = raw.split('```')[1].split('```')[0]
    return raw.strip()


def _try_direct_answer(query: str, stats: Dict) -> Optional[str]:
    q = query.lower()
    checks = [
        (['how many', 'critical'],
         f"There are **{stats.get('critical_count','N/A')}** critical activities "
         f"({stats.get('critical_pct','N/A')}% of total)."),
        (['negative float'],
         f"**{stats.get('negative_float_count','N/A')}** activities have negative float."),
        (['total', 'activit'],
         f"The schedule contains **{stats.get('total_activities','N/A')}** total activities."),
        (['open', 'ended'],
         f"**{stats.get('open_ended_count','N/A')}** open-ended activities (no successors)."),
        (['dangling'],
         f"**{stats.get('dangling_count','N/A')}** dangling activities (no predecessors)."),
        (['long duration'],
         f"**{stats.get('long_duration_count','N/A')}** activities exceed 30 days duration."),
        (['how many', 'complet'],
         f"**{stats.get('completed','N/A')}** complete, "
         f"**{stats.get('in_progress','N/A')}** in progress, "
         f"**{stats.get('not_started','N/A')}** not started."),
        (['total', 'relationship'],
         f"The schedule has **{stats.get('total_relationships','N/A')}** relationships "
         f"({stats.get('negative_lags','N/A')} with negative lags)."),
    ]
    for keywords, answer in checks:
        if all(k in q for k in keywords):
            return answer
    return None


def get_ai_response(user_query: str) -> str:
    analyzer      = st.session_state.analyzer
    basic_stats   = analyzer.get_basic_stats()
    insights      = analyzer.get_insights()          # pre-computed — no LLM cost
    hybrid_client = _get_hybrid_client()

    quick = _try_direct_answer(user_query, basic_stats)
    if quick:
        return quick

    history: List[Dict] = [
        {'role': m['role'], 'content': m['content']}
        for m in st.session_state.messages[:-1][-6:]
    ]

    code_success, code_result, code_error, generated_code = False, None, None, ''

    try:
        # ── Generate code ─────────────────────────────────
        code_gen_prompt = analyzer.get_code_generation_prompt(
            user_query, basic_stats, conversation_history=history,
        )
        raw_code, code_model = hybrid_client.chat(
            [
                {'role': 'system', 'content': (
                    'You are a Python code generator for Primavera P6 schedule analysis. '
                    'Generate ONLY valid Python code that sets result = ... at the end. '
                    'No explanations, no markdown.'
                )},
                {'role': 'user', 'content': code_gen_prompt},
            ],
            temperature=0.1, max_tokens=2000, return_model=True,
        )
        print(f'Code generated by: {code_model}')
        generated_code = _extract_code(raw_code)
        print('GENERATED CODE:\n', generated_code)

        # ── Execute ────────────────────────────────────────
        exec_result  = analyzer.execute_code(generated_code)
        code_success = exec_result['success']
        code_result  = exec_result.get('result')
        code_error   = exec_result.get('error')

        # ── Retry once on failure ──────────────────────────
        if not code_success:
            print(f'Execution failed: {code_error} — retrying...')
            retry_prompt = analyzer.get_code_generation_prompt(
                user_query, basic_stats,
                previous_code=generated_code,
                previous_error=code_error,
                conversation_history=history,
            )
            raw_retry, retry_model = hybrid_client.chat(
                [
                    {'role': 'system', 'content': (
                        'You are a Python code generator. Fix the error. '
                        'Return ONLY corrected Python code.'
                    )},
                    {'role': 'user', 'content': retry_prompt},
                ],
                temperature=0.1, max_tokens=2000, return_model=True,
            )
            print(f'Retry by: {retry_model}')
            generated_code = _extract_code(raw_retry)
            exec_result    = analyzer.execute_code(generated_code)
            code_success   = exec_result['success']
            code_result    = exec_result.get('result')
            code_error     = exec_result.get('error')

    except Exception as e:
        code_error = str(e)
        print(f'Code generation error: {e}')

    # ── Final response ─────────────────────────────────────
    response_prompt = analyzer.get_response_prompt(
        user_query, basic_stats, code_result, code_success, code_error,
        insights=insights,                            # ← ground-truth insights in LLM context
    )

    try:
        response, response_model = hybrid_client.chat(
            [
                {'role': 'system', 'content': analyzer.get_system_prompt()},
                *history,
                {'role': 'user', 'content': response_prompt},
            ],
            temperature=0.3, max_tokens=2000, return_model=True,
        )
        print(f'Response by: {response_model}')
        return response
    except Exception as e:
        return (
            f"Error generating response.\n\n"
            f"**Project:** {basic_stats.get('data_source','N/A')} | "
            f"**Data Date:** {basic_stats.get('data_date','N/A')}\n\n"
            f"**Activities:** {basic_stats.get('total_activities','N/A')} | "
            f"**Critical:** {basic_stats.get('critical_count','N/A')} ({basic_stats.get('critical_pct','N/A')}%)\n\n"
            f"Error: {e}"
        )


# =============================================================================
# INSIGHTS PANELS  (used in sidebar and welcome screen)
# =============================================================================

def _render_delays_panel(insights: Dict, max_rows: int = 8):
    delays = insights.get('delays', [])
    if not delays:
        st.markdown('<div class="insight-card ok">✓ No delays detected</div>', unsafe_allow_html=True)
        return
    st.markdown(f"**{len(delays)} delayed tasks** — worst first:")
    for d in delays[:max_rows]:
        slip   = d.get('finish_slip_days') or d.get('start_slip_days') or 0
        flag   = '🔴' if d.get('is_critical') else '🟡'
        label  = f"{flag} **{d['task_code']}** {d['task_name'][:35]} — {slip:.0f}d slip"
        st.markdown(f"<div class='insight-card'>{label}</div>", unsafe_allow_html=True)
    if len(delays) > max_rows:
        st.caption(f"… and {len(delays) - max_rows} more. Ask the assistant for the full list.")


def _render_critical_path_panel(insights: Dict, max_rows: int = 8):
    cp = insights.get('critical_path', [])
    if not cp:
        st.markdown("*No critical tasks found.*")
        return
    remaining_total = sum(t.get('remaining_days', 0) for t in cp)
    st.markdown(f"**{len(cp)} critical tasks** | {remaining_total:.0f}d remaining work")
    for t in cp[:max_rows]:
        flag  = '🔴' if t.get('has_negative_float') else'🟢 '
        label = f"{flag} **{t['task_code']}** {t['task_name'][:35]} — {t['remaining_days']}d rem"
        st.markdown(f"<div class='insight-card critical'>{label}</div>", unsafe_allow_html=True)
    if len(cp) > max_rows:
        st.caption(f"… and {len(cp) - max_rows} more.")


def _render_resource_panel(insights: Dict, max_rows: int = 8):
    resources = insights.get('resources', [])
    if not resources:
        st.markdown("*No resource data in schedule.*")
        return
    total_planned = sum(r.get('planned_cost', 0) for r in resources)
    total_actual  = sum(r.get('actual_cost', 0) for r in resources)
    st.markdown(f"**{len(resources)} resources** | Planned: {total_planned:,.0f} | Actual: {total_actual:,.0f}")
    for r in resources[:max_rows]:
        spent_pct = (r['actual_cost'] / r['planned_cost'] * 100) if r['planned_cost'] else 0
        label = (
            f"**{r['resource_name']}** — {r['task_count']} tasks | "
            f"Planned: {r['planned_cost']:,.0f} | Spent: {spent_pct:.0f}%"
        )
        st.markdown(f"<div class='insight-card'>{label}</div>", unsafe_allow_html=True)


# =============================================================================
# UPLOAD MODAL
# =============================================================================

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
        st.markdown('### Upload Baseline File')
        uploaded_file = st.file_uploader(
            'Select XER file', type=['xer'],
            key='baseline_upload', label_visibility='collapsed',
        )
        if uploaded_file:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.baseline_file_id != file_id:
                with st.spinner('Parsing schedule...'):
                    result = load_xer_file(uploaded_file, 'baseline')
                    if result['success']:
                        az = st.session_state.analyzer
                        az.load_baseline(result['data'], result['project_name'], result['data_date'])
                        st.session_state.project_name     = result['project_name']
                        st.session_state.baseline_info    = {
                            'name':      result['project_name'],
                            'data_date': result['data_date'],
                            'file_name': result['file_name'],
                        }
                        st.session_state.baseline_loaded  = True
                        st.session_state.baseline_file_id = file_id
                        st.success('Baseline loaded!')
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")


# =============================================================================
# CHAT INTERFACE
# =============================================================================

def show_chat_interface():
    analyzer    = st.session_state.analyzer
    basic_stats = analyzer.get_basic_stats()
    insights    = analyzer.get_insights()

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown('### Project Files')

        baseline = st.session_state.baseline_info
        if baseline:
            st.markdown(f"""
            <div class="baseline-item">
                <strong>{baseline['name']}</strong><br>
                <small>Data Date: {baseline['data_date'] or 'N/A'}</small>
            </div>""", unsafe_allow_html=True)

        st.markdown('---')
        st.markdown('**Update Files:**')
        if st.session_state.update_files_info:
            for i, uf in enumerate(st.session_state.update_files_info):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"""
                    <div class="update-file-item">
                        <strong>{uf['name']}</strong><br>
                        <small>Data Date: {uf['data_date'] or 'N/A'}</small>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    if st.button('X', key=f'remove_{i}'):
                        st.session_state.update_files_info.pop(i)
                        fid = uf.get('file_id')
                        if fid in st.session_state.processed_update_ids:
                            st.session_state.processed_update_ids.remove(fid)
                        analyzer.remove_update(i)
                        st.rerun()
        else:
            st.markdown('*No updates loaded*')

        st.markdown('---')
        st.markdown('**Add Update:**')
        update_file = st.file_uploader(
            'Upload', type=['xer'], key='update_upload', label_visibility='collapsed',
        )
        if update_file:
            file_id = f"{update_file.name}_{update_file.size}"
            if file_id not in st.session_state.processed_update_ids:
                with st.spinner('Loading...'):
                    result = load_xer_file(update_file, 'update')
                    if result['success']:
                        analyzer.add_update(result['data'], result['project_name'], result['data_date'])
                        st.session_state.update_files_info.append({
                            'name': result['project_name'], 'data_date': result['data_date'],
                            'file_name': result['file_name'], 'file_id': file_id,
                        })
                        st.session_state.processed_update_ids.append(file_id)
                        st.success(f"Loaded: {result['data_date']}")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")

        # ── Schedule Health ────────────────────────────────
        st.markdown('---')
        st.markdown('### Schedule Health')
        col_a, col_b = st.columns(2)
        col_a.metric('Activities', basic_stats.get('total_activities', 0))
        col_a.metric('Critical %', f"{basic_stats.get('critical_pct', 0)}%")
        col_a.metric('Neg Float',  basic_stats.get('negative_float_count', 0))
        col_b.metric('Open-Ended', basic_stats.get('open_ended_count', 0))
        col_b.metric('Dangling',   basic_stats.get('dangling_count', 0))
        col_b.metric('Constrained', basic_stats.get('constrained_activities', 0))

        # ── Insights panels ────────────────────────────────
        st.markdown('---')
        with st.expander('Delays', expanded=False):
            _render_delays_panel(insights, max_rows=6)

        with st.expander(' Critical Path', expanded=False):
            _render_critical_path_panel(insights, max_rows=6)

        with st.expander(' Resources', expanded=False):
            _render_resource_panel(insights, max_rows=6)

        st.markdown('---')
        if st.button('Clear Chat', use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Project bar ───────────────────────────────────────
    updates_text = (
        f" | Updates: {len(st.session_state.update_files_info)}"
        if st.session_state.update_files_info else ''
    )
    st.markdown(f"""
    <div class="project-bar">
        <strong>Project:</strong> {st.session_state.project_name} |
        <strong>Activities:</strong> {basic_stats.get('total_activities', 0)} |
        <strong>Period:</strong> {basic_stats.get('project_start','N/A')} to {basic_stats.get('project_finish','N/A')}{updates_text}
    </div>""", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # ── Process pending user message ──────────────────────
    if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
        with st.chat_message('assistant'):
            with st.spinner('Analyzing...'):
                response = get_ai_response(st.session_state.messages[-1]['content'])
            st.markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    # ── Welcome screen ─────────────────────────────────────
    if not st.session_state.messages:
        delays    = insights.get('delays', [])
        cp        = insights.get('critical_path', [])
        resources = insights.get('resources', [])

        st.markdown("### Schedule Overview")

        tab_health, tab_delays, tab_cp, tab_resources = st.tabs(
            [' Health', ' Delays', ' Critical Path', '\Resources']
        )

        with tab_health:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Total Activities', basic_stats.get('total_activities', 0))
            c2.metric('Critical',
                      f"{basic_stats.get('critical_count', 0)} ({basic_stats.get('critical_pct', 0)}%)")
            c3.metric('Negative Float', basic_stats.get('negative_float_count', 0))
            c4.metric('In Progress', basic_stats.get('in_progress', 0))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric('Open-Ended', basic_stats.get('open_ended_count', 0))
            c6.metric('Dangling', basic_stats.get('dangling_count', 0))
            c7.metric('Long Duration >30d', basic_stats.get('long_duration_count', 0))
            c8.metric('Constrained', basic_stats.get('constrained_activities', 0))

            st.markdown("---")
            st.markdown("**Ask me anything about your schedule:**")
            st.markdown(
                "- Show all critical activities with negative float\n"
                "- Which tasks are delayed and by how much?\n"
                "- Compare baseline vs latest update\n"
                "- List open-ended activities in the civil WBS\n"
                "- What resources are over-budget?"
            )

        with tab_delays:
            if delays:
                st.markdown(f"**{len(delays)} tasks are running late** (vs planned dates):")
                _render_delays_panel(insights, max_rows=15)
            else:
                st.success("No delays detected — all started/completed tasks are on or ahead of plan.")

        with tab_cp:
            if cp:
                cp_remaining = sum(t.get('remaining_days', 0) for t in cp)
                neg_float_count = sum(1 for t in cp if t.get('has_negative_float'))
                st.markdown(
                    f"**{len(cp)} remaining critical tasks** | "
                    f"**{cp_remaining:.0f}d** total remaining work | "
                    f"**{neg_float_count}** with negative float"
                )
                _render_critical_path_panel(insights, max_rows=20)
            else:
                st.info("No remaining critical work tasks found.")

        with tab_resources:
            if resources:
                total_planned  = sum(r.get('planned_cost', 0) for r in resources)
                total_actual   = sum(r.get('actual_cost', 0) for r in resources)
                total_remaining = sum(r.get('remaining_cost', 0) for r in resources)
                c1, c2, c3 = st.columns(3)
                c1.metric('Total Planned Cost', f"{total_planned:,.0f}")
                c2.metric('Total Actual Cost',  f"{total_actual:,.0f}")
                c3.metric('Remaining Cost',      f"{total_remaining:,.0f}")
                st.markdown("---")
                _render_resource_panel(insights, max_rows=20)
            else:
                st.info("No resource assignments found in this schedule.")

    # ── Chat input ────────────────────────────────────────
    if prompt := st.chat_input('Ask about your schedule...'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not get_openai_key():
        st.warning(
            "OPENAI_API_KEY not found. Cloud model unavailable. "
            "Set it in .env or Streamlit secrets. Ollama local model will still work if running."
        )

    if not st.session_state.baseline_loaded:
        show_upload_modal()
    else:
        show_chat_interface()


if __name__ == '__main__':
    main()
