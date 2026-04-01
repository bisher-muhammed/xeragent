import streamlit as st
import json
import os
from dotenv import load_dotenv

load_dotenv()


def get_openai_key():
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        try:
            key = st.secrets.get('OPENAI_API_KEY')
        except Exception:
            pass
    return key


from openai import OpenAI
from xer_analyzer import XERAnalyzer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from xer_complete_extractor import CompleteXERExtractor


st.set_page_config(
    page_title="XER Schedule Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    .project-bar {
        background: #f8f9fa;
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


if 'baseline_loaded' not in st.session_state:
    st.session_state.baseline_loaded = False
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = XERAnalyzer()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'project_name' not in st.session_state:
    st.session_state.project_name = None
if 'baseline_info' not in st.session_state:
    st.session_state.baseline_info = None
if 'update_files_info' not in st.session_state:
    st.session_state.update_files_info = []
if 'baseline_file_id' not in st.session_state:
    st.session_state.baseline_file_id = None
if 'processed_update_ids' not in st.session_state:
    st.session_state.processed_update_ids = set()


def load_xer_file(uploaded_file, file_type: str = 'baseline') -> dict:
    try:
        # Write raw bytes — extractor handles windows-1252 encoding internally
        raw_bytes = uploaded_file.read()
        temp_path = f"temp_{file_type}_{uploaded_file.name}"

        with open(temp_path, 'wb') as f:
            f.write(raw_bytes)

        extractor = CompleteXERExtractor(temp_path, file_type)
        extractor.extract_all()

        project_info = extractor.get_project_info()

        # Include statistics and relationships so the analyzer can surface data quality metrics
        data = {
            'project': project_info,
            'tasks': extractor.get_all_tasks(),
            'wbs': extractor.get_wbs_structure(),
            'tables': extractor.tables,
            'statistics': extractor.extraction_stats,
            'relationships': dict(extractor.table_relationships)
        }

        data_date = project_info.get('data_date', '')
        if data_date:
            data_date = str(data_date)[:10]

        os.remove(temp_path)

        return {
            'success': True,
            'data': data,
            'project_name': project_info.get('project_name', uploaded_file.name),
            'data_date': data_date,
            'file_name': uploaded_file.name
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_ai_response(user_query: str) -> str:
    analyzer = st.session_state.analyzer
    basic_stats = analyzer.get_basic_stats()

    # Initialize hybrid client and set mode from session_state
    from hybrid_llm_client import HybridLLMClient  # your class from previous snippe
    client = HybridLLMClient()
    if 'llm_mode' in st.session_state:
        client.set_mode(st.session_state.llm_mode)

    code_gen_prompt = analyzer.get_code_generation_prompt(user_query, basic_stats)

    code_success = False
    code_result = None
    code_error = None

    try:
        # Generate Python code for analysis
        generated_code = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Python code generator for Primavera P6 schedule analysis. "
                        "Generate ONLY valid Python code that sets result = ... at the end. "
                        "No markdown fences, no explanations, no comments — just executable code."
                    )
                },
                {"role": "user", "content": code_gen_prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            return_model=False
        )

        # Strip markdown fences if any
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

    except Exception as e:
        code_error = str(e)
        print("CODE GENERATION ERROR:", code_error)

    response_prompt = analyzer.get_response_prompt(
        user_query, basic_stats, code_result, code_success, code_error
    )

    try:
        # Generate human-readable response using the same hybrid client
        final_response = client.chat(
            messages=[
                {"role": "system", "content": analyzer.get_system_prompt()},
                {"role": "user", "content": response_prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
            return_model=False
        )
        return final_response

    except Exception as e:
        return (
            f"**Error generating response:** {str(e)}\n\n"
            f"**Project:** {basic_stats.get('data_source', 'N/A')} | "
            f"**Data Date:** {basic_stats.get('data_date', 'N/A')}\n"
            f"- Total Activities: {basic_stats.get('total_activities', 'N/A')}\n"
            f"- Critical: {basic_stats.get('critical_count', 'N/A')} ({basic_stats.get('critical_pct', 'N/A')}%)\n"
            f"- Negative Float: {basic_stats.get('negative_float_count', 'N/A')}\n"
            f"- Open-Ended: {basic_stats.get('open_ended_count', 'N/A')}"
        )


def show_upload_modal():
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; min-height: 60vh;">
        <div style="text-align: center; max-width: 500px; padding: 40px;">
            <h1>XER Schedule Assistant</h1>
            <p style="color: #666; margin: 20px 0;">
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
            label_visibility='collapsed'
        )

        if uploaded_file:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.baseline_file_id != file_id:
                with st.spinner("Loading baseline..."):
                    result = load_xer_file(uploaded_file, 'baseline')
                    if result['success']:
                        analyzer = st.session_state.analyzer
                        analyzer.load_baseline(result['data'], result['project_name'], result['data_date'])
                        st.session_state.project_name = result['project_name']
                        st.session_state.baseline_info = {
                            'name': result['project_name'],
                            'data_date': result['data_date'],
                            'file_name': result['file_name']
                        }
                        st.session_state.baseline_loaded = True
                        st.session_state.baseline_file_id = file_id
                        st.success("Baseline loaded!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result['error']}")


def show_chat_interface():
    analyzer = st.session_state.analyzer
    basic_stats = analyzer.get_basic_stats()

    with st.sidebar:
        # --- ADDED: LLM Mode Selection ---
        st.markdown("### AI Settings")
        llm_mode = st.radio(
            "Select AI Engine:",
            options=["Cloud AI (OpenAI)", "Local AI (Ollama)"],
            index=0 if st.session_state.get('llm_mode') != 'ollama' else 1,
            help="Switch between cloud-based GPT and your local Ollama instance."
        )
        
        # Map the UI selection to the mode expected by your HybridLLMClient
        st.session_state.llm_mode = 'cloud' if "Cloud" in llm_mode else 'ollama'
        
        if st.session_state.llm_mode == 'ollama':
            st.info("🤖 Running on local Ollama")
        else:
            st.success("☁️ Running on OpenAI")
        
        st.markdown("---")
        # --- END OF ADDED SECTION ---

        st.markdown("### Project Files")

        if st.session_state.update_files_info:
            for i, uf in enumerate(st.session_state.update_files_info):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div class="update-file-item">
                        <strong>{uf['name']}</strong><br>
                        <small>Data Date: {uf['data_date'] or 'N/A'}</small>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("X", key=f"remove_{i}"):
                        st.session_state.update_files_info.pop(i)
                        st.session_state.processed_update_ids.discard(uf['file_id'])
                        analyzer.remove_update(i)
                        st.rerun()
        else:
            st.markdown("*No updates loaded*")

        st.markdown("---")
        st.markdown("**Add Update:**")
        update_file = st.file_uploader("Upload", type=['xer'], key='update_upload', label_visibility='collapsed')

        if update_file:
            file_id = f"{update_file.name}_{update_file.size}"
            if file_id not in st.session_state.processed_update_ids:
                with st.spinner("Loading..."):
                    result = load_xer_file(update_file, 'update')
                    if result['success']:
                        analyzer.add_update(result['data'], result['project_name'], result['data_date'])
                        st.session_state.update_files_info.append({
                            'name': result['project_name'],
                            'data_date': result['data_date'],
                            'file_name': result['file_name'],
                            'file_id': file_id
                        })
                        st.session_state.processed_update_ids.add(file_id)
                        st.success(f"Loaded: {result['data_date']}")
                        st.rerun()

        st.markdown("---")
        st.markdown("### Schedule Health")
        st.markdown(f"- Activities: **{basic_stats.get('total_activities', 0)}**")
        st.markdown(f"- Critical: **{basic_stats.get('critical_count', 0)}** ({basic_stats.get('critical_pct', 0)}%)")
        st.markdown(f"- Near-Critical: **{basic_stats.get('near_critical_count', 0)}**")
        st.markdown(f"- Neg Float: **{basic_stats.get('negative_float_count', 0)}**")
        st.markdown(f"- Open-Ended: **{basic_stats.get('open_ended_count', 0)}**")
        st.markdown(f"- Long Dur (>20d): **{basic_stats.get('long_duration_count', 0)}**")
        st.markdown(f"- Overdue (not started): **{basic_stats.get('overdue_not_started', 0)}**")
        st.markdown(f"- Overdue (in progress): **{basic_stats.get('overdue_in_progress', 0)}**")

        st.markdown("---")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    updates_text = f" | Updates: {len(st.session_state.update_files_info)}" if st.session_state.update_files_info else ""
    st.markdown(f"""
    <div class="project-bar">
        <strong>Project:</strong> {st.session_state.project_name} |
        <strong>Activities:</strong> {basic_stats.get('total_activities', 0)} |
        <strong>Period:</strong> {basic_stats.get('project_start', 'N/A')} → {basic_stats.get('project_finish', 'N/A')}{updates_text}
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Analyzing schedule..."):
                response = get_ai_response(st.session_state.messages[-1]["content"])
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

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

Ask me anything about your schedule.
        """)

    if prompt := st.chat_input("Ask about your schedule..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()


def main():
    if not get_openai_key():
        st.error("OPENAI_API_KEY not found. Set it in .env (local) or Streamlit secrets (cloud).")
        st.stop()

    if not st.session_state.baseline_loaded:
        show_upload_modal()
    else:
        show_chat_interface()


if __name__ == "__main__":
    main()
