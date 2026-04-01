import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class XERDataStore:

    def __init__(self):
        self.baseline = None
        self.updates = []
        self.hours_per_day = 8
        self._cached_stats = None

    def load_baseline(self, data: Dict, name: str, data_date: str):
        self.baseline = {
            'name': name,
            'data_date': data_date,
            'data': data,
            'df': self._create_dataframes(data)
        }
        self._cached_stats = None

    def add_update(self, data: Dict, name: str, data_date: str):
        self.updates.append({
            'name': name,
            'data_date': data_date,
            'data': data,
            'df': self._create_dataframes(data)
        })
        self.updates.sort(key=lambda x: x['data_date'])
        self._cached_stats = None

    def remove_update(self, index: int):
        if 0 <= index < len(self.updates):
            self.updates.pop(index)
            self._cached_stats = None

    def _create_dataframes(self, data: Dict) -> Dict[str, pd.DataFrame]:
        dfs = {}

        # Tasks and WBS from convenience accessors (avoids duplication with tables dict)
        if data.get('tasks'):
            dfs['tasks'] = pd.DataFrame(data['tasks'])

        if data.get('wbs'):
            dfs['wbs'] = pd.DataFrame(data['wbs'])

        # All other tables from the raw tables dict, lowercased
        skip = {'TASK', 'PROJWBS'}
        for table_name, records in data.get('tables', {}).items():
            if records and table_name not in skip:
                dfs[table_name.lower()] = pd.DataFrame(records)

        return dfs

    def get_latest(self) -> Optional[Dict]:
        if self.updates:
            return self.updates[-1]
        return self.baseline

    def get_baseline(self) -> Optional[Dict]:
        return self.baseline

    def get_update_by_date(self, date_str: str) -> Optional[Dict]:
        for update in self.updates:
            if date_str in update['data_date'] or date_str in update['name']:
                return update
        return None

    def get_update_by_month(self, month: str, year: str = None) -> Optional[Dict]:
        month_map = {
            'jan': '01', 'january': '01', '01': '01', '1': '01',
            'feb': '02', 'february': '02', '02': '02', '2': '02',
            'mar': '03', 'march': '03', '03': '03', '3': '03',
            'apr': '04', 'april': '04', '04': '04', '4': '04',
            'may': '05', '05': '05', '5': '05',
            'jun': '06', 'june': '06', '06': '06', '6': '06',
            'jul': '07', 'july': '07', '07': '07', '7': '07',
            'aug': '08', 'august': '08', '08': '08', '8': '08',
            'sep': '09', 'september': '09', '09': '09', '9': '09',
            'oct': '10', 'october': '10', '10': '10',
            'nov': '11', 'november': '11', '11': '11',
            'dec': '12', 'december': '12', '12': '12'
        }
        month_num = month_map.get(month.lower(), month)
        for update in self.updates:
            data_date = update['data_date']
            if len(data_date) >= 7:
                if data_date[5:7] == month_num:
                    if year is None or data_date[:4] == year:
                        return update
        return None

    def compute_basic_stats(self) -> Dict:
        if self._cached_stats:
            return self._cached_stats

        source = self.get_latest()
        if not source or 'tasks' not in source.get('df', {}):
            return {'error': 'No data loaded'}

        tasks_df = source['df']['tasks'].copy()
        stats = {
            'total_activities': len(tasks_df),
            'data_source': source['name'],
            'data_date': source['data_date'],
        }

        # Data quality from extractor
        raw = source.get('data', {})
        extractor_stats = raw.get('statistics', {})
        stats['duplicate_pk_count'] = extractor_stats.get('duplicate_pk_count', 0)
        stats['orphan_fk_count'] = extractor_stats.get('orphan_fk_count', 0)
        schema_rels = raw.get('relationships', {})
        stats['inferred_relationship_count'] = sum(len(v) for v in schema_rels.values())

        # Task types
        if 'task_type' in tasks_df.columns:
            type_counts = tasks_df['task_type'].value_counts().to_dict()
            stats['task_types'] = type_counts
            stats['milestones'] = type_counts.get('TT_Mile', 0) + type_counts.get('TT_FinMile', 0)
            stats['loe_activities'] = type_counts.get('TT_LOE', 0)
            stats['regular_tasks'] = type_counts.get('TT_Task', 0)

        # Status
        if 'status_code' in tasks_df.columns:
            status_counts = tasks_df['status_code'].value_counts().to_dict()
            stats.update({
                'status_breakdown': status_counts,
                'completed': status_counts.get('TK_Complete', 0),
                'in_progress': status_counts.get('TK_Active', 0),
                'not_started': status_counts.get('TK_NotStart', 0)
            })

        # Computed numeric columns
        tasks_df['duration_hrs'] = pd.to_numeric(tasks_df.get('target_drtn_hr_cnt', 0), errors='coerce').fillna(0)
        tasks_df['duration_days'] = tasks_df['duration_hrs'] / self.hours_per_day
        tasks_df['float_hrs'] = pd.to_numeric(tasks_df.get('total_float_hr_cnt', 0), errors='coerce').fillna(0)
        tasks_df['complete_pct'] = pd.to_numeric(tasks_df.get('phys_complete_pct', 0), errors='coerce').fillna(0)

        # Work tasks only (exclude LOE and milestones)
        if 'task_type' in tasks_df.columns:
            work_mask = ~tasks_df['task_type'].isin(['TT_LOE', 'TT_Mile', 'TT_FinMile'])
        else:
            work_mask = pd.Series([True] * len(tasks_df), index=tasks_df.index)
        work_tasks = tasks_df[work_mask]

        # Critical path
        critical = work_tasks[work_tasks['float_hrs'] <= 0]
        near_critical_threshold = 14 * self.hours_per_day
        near_critical = work_tasks[
            (work_tasks['float_hrs'] > 0) & (work_tasks['float_hrs'] <= near_critical_threshold)
        ]
        stats.update({
            'critical_count': len(critical),
            'critical_pct': round(len(critical) / max(len(work_tasks), 1) * 100, 1),
            'near_critical_count': len(near_critical)
        })

        # Float issues (computed on work tasks, independent of pred_df)
        stats['negative_float_count'] = int((work_tasks['float_hrs'] < 0).sum())
        stats['high_float_count'] = int((work_tasks['float_hrs'] > (30 * self.hours_per_day)).sum())

        # Long duration
        stats['long_duration_count'] = int((work_tasks['duration_days'] > 20).sum())

        # Constraint analysis
        if 'cstr_type' in tasks_df.columns:
            constrained = tasks_df[tasks_df['cstr_type'].notna() & (tasks_df['cstr_type'] != '')]
            stats['constrained_activities'] = len(constrained)
            if len(constrained) > 0:
                stats['constraint_breakdown'] = constrained['cstr_type'].value_counts().to_dict()

        # Predecessor/successor analysis
        pred_df = source['df'].get('taskpred', pd.DataFrame())
        if not pred_df.empty:
            pred_df = pred_df.copy()
            pred_df['pred_task_id'] = pred_df['pred_task_id'].astype(str)
            pred_df['task_id'] = pred_df['task_id'].astype(str)
            work_task_ids = set(work_tasks['task_id'].astype(str))
            has_successor = set(pred_df['pred_task_id'])
            has_predecessor = set(pred_df['task_id'])

            stats['open_ended_count'] = len(work_task_ids - has_successor)
            stats['dangling_count'] = len(work_task_ids - has_predecessor)
            stats['total_relationships'] = len(pred_df)

            if 'lag_hr_cnt' in pred_df.columns:
                pred_df['lag_hrs'] = pd.to_numeric(pred_df['lag_hr_cnt'], errors='coerce').fillna(0)
                stats['negative_lag_count'] = int((pred_df['lag_hrs'] < 0).sum())
                stats['positive_lag_count'] = int((pred_df['lag_hrs'] > 0).sum())
                stats['relationships_with_lag'] = int((pred_df['lag_hrs'] != 0).sum())
                stats['negative_lags'] = stats['negative_lag_count']
        else:
            stats.update({
                'open_ended_count': 'N/A',
                'dangling_count': 'N/A',
                'total_relationships': 0
            })

        # Resource loading
        rsrc_df = source['df'].get('taskrsrc', pd.DataFrame())
        if not rsrc_df.empty:
            tasks_with_rsrc = rsrc_df['task_id'].nunique()
            stats.update({
                'resource_assignments': len(rsrc_df),
                'tasks_with_resources': tasks_with_rsrc,
                'resource_loaded_pct': round(tasks_with_rsrc / max(stats['total_activities'], 1) * 100, 1)
            })

        # Project date range
        if 'target_start_date' in tasks_df.columns:
            starts = pd.to_datetime(tasks_df['target_start_date'], errors='coerce').dropna()
            if not starts.empty:
                stats['project_start'] = str(starts.min())[:10]

        if 'target_end_date' in tasks_df.columns:
            ends = pd.to_datetime(tasks_df['target_end_date'], errors='coerce').dropna()
            if not ends.empty:
                stats['project_finish'] = str(ends.max())[:10]

        # Delay / overdue analysis relative to data date
        data_date_str = source.get('data_date', '')
        if data_date_str:
            try:
                data_date_dt = pd.to_datetime(data_date_str)

                if 'status_code' in tasks_df.columns and 'target_start_date' in tasks_df.columns:
                    not_started = tasks_df[tasks_df['status_code'] == 'TK_NotStart'].copy()
                    not_started['ts_dt'] = pd.to_datetime(not_started['target_start_date'], errors='coerce')
                    stats['overdue_not_started'] = int((not_started['ts_dt'] < data_date_dt).sum())

                if 'status_code' in tasks_df.columns and 'target_end_date' in tasks_df.columns:
                    in_prog = tasks_df[tasks_df['status_code'] == 'TK_Active'].copy()
                    in_prog['te_dt'] = pd.to_datetime(in_prog['target_end_date'], errors='coerce')
                    stats['overdue_in_progress'] = int((in_prog['te_dt'] < data_date_dt).sum())

            except Exception:
                pass

        # WBS-level progress summary
        wbs_df = source['df'].get('wbs', pd.DataFrame())
        if not wbs_df.empty and 'wbs_id' in tasks_df.columns and 'wbs_name' in wbs_df.columns:
            try:
                wbs_map = wbs_df.set_index('wbs_id')['wbs_name'].to_dict()
                tasks_df['wbs_name'] = tasks_df['wbs_id'].map(wbs_map)
                wbs_progress = (
                    tasks_df[work_mask]
                    .groupby('wbs_name')['complete_pct']
                    .mean()
                    .round(1)
                    .sort_values()
                    .to_dict()
                )
                stats['wbs_progress'] = wbs_progress
            except Exception:
                pass

        # File info
        stats.update({
            'baseline_name': self.baseline['name'] if self.baseline else None,
            'baseline_date': self.baseline['data_date'] if self.baseline else None,
            'update_count': len(self.updates),
            'updates': [{'name': u['name'], 'date': u['data_date']} for u in self.updates]
        })

        self._cached_stats = stats
        return stats


class XERQueryExecutor:

    def __init__(self, data_store: XERDataStore):
        self.data_store = data_store

    def execute(self, code: str) -> Dict[str, Any]:
        try:
            context = {
                'pd': pd,
                'datetime': datetime,
                'json': json,
                'baseline': self.data_store.get_baseline(),
                'updates': self.data_store.updates,
                'latest': self.data_store.get_latest(),
                'get_latest': self.data_store.get_latest,
                'get_baseline': self.data_store.get_baseline,
                'get_update_by_date': self.data_store.get_update_by_date,
                'get_update_by_month': self.data_store.get_update_by_month,
                'hours_per_day': self.data_store.hours_per_day,
                'result': None
            }

            exec(code, context)
            result = context.get('result')

            if isinstance(result, pd.DataFrame):
                result = result.head(50).to_dict('records')

            return {'success': True, 'result': result}

        except Exception as e:
            return {'success': False, 'error': str(e)}


class XERAnalyzer:

    def __init__(self):
        self.data_store = XERDataStore()
        self.executor = XERQueryExecutor(self.data_store)

    def load_baseline(self, data: Dict, name: str = None, data_date: str = None):
        if name is None:
            name = data.get('project', {}).get('project_name', 'Baseline')
        if data_date is None:
            data_date = str(data.get('project', {}).get('data_date', ''))[:10]
        self.data_store.load_baseline(data, name, data_date)

    def add_update(self, data: Dict, name: str = None, data_date: str = None):
        if name is None:
            name = data.get('project', {}).get('project_name', 'Update')
        if data_date is None:
            data_date = str(data.get('project', {}).get('data_date', ''))[:10]
        self.data_store.add_update(data, name, data_date)

    def remove_update(self, index: int):
        self.data_store.remove_update(index)

    def get_basic_stats(self) -> Dict:
        return self.data_store.compute_basic_stats()

    def execute_code(self, code: str) -> Dict:
        return self.executor.execute(code)

    def get_system_prompt(self) -> str:
        return """You are an expert Primavera P6 Schedule Analyst. You analyze construction project schedules from XER files with precision and accuracy.

STRICT RULES — NON-NEGOTIABLE:
- Base ALL analysis exclusively on data provided in the context. Never fabricate, estimate, or invent values.
- Activity names, codes, and dates must match exactly what exists in the provided data.
- If the data is insufficient to answer a question, state that clearly rather than guessing.
- If analysis code failed, explain what you could determine from available statistics instead.
- Do not reference industry averages or general norms unless explicitly asked.

RESPONSE STYLE:
- Direct, professional, and precise
- All numbers must come from the provided data
- Use bullet points for lists, tables for comparisons
- Flag concerns with context — not just the number, but what it implies
- Provide specific recommendations when issues are identified

SCHEDULE ANALYSIS CONTEXT:
- Critical activities have zero or negative total float — any delay directly delays project completion
- Open-ended activities (no successor) break the logic network and inflate float unrealistically
- Dangling activities (no predecessor) can start at any time, making the schedule unreliable
- Hard constraints (CS_MSO, CS_MEO, CS_MEOB) override schedule logic and should be justified
- Negative lags are generally poor practice and should be replaced with proper logic
- Resource-loaded schedules allow earned value analysis; unloaded schedules limit forecasting"""

    def get_code_generation_prompt(self, user_query: str, basic_stats: Dict) -> str:
        source = self.data_store.get_latest()
        columns_info = ""
        if source and 'df' in source:
            for table_name, df in list(source['df'].items())[:10]:
                cols = list(df.columns)[:20]
                columns_info += f"\n{table_name.upper()} [{len(df)} rows]: {', '.join(cols)}"

        return f"""Generate Python code to answer this question about a Primavera P6 schedule.

USER QUESTION: {user_query}

PRE-COMPUTED STATISTICS (use as fallback if code analysis is unnecessary):
{json.dumps(basic_stats, indent=2, default=str)}

AVAILABLE DATAFRAMES AND COLUMNS:{columns_info}

AVAILABLE VARIABLES:
- latest: dict — latest schedule. Access DataFrames via latest['df']['table_name']
- baseline: dict — baseline schedule. Access via baseline['df']['table_name']  
- updates: list of dicts — all update schedules
- get_update_by_month('feb'): returns update dict for that month, or None
- get_update_by_date('2024-03'): returns update dict matching date string
- pd: pandas
- datetime: Python datetime
- hours_per_day: {self.data_store.hours_per_day} (hours in a working day)

KEY DATAFRAMES AND FIELDS:
tasks:
  task_id, task_code, task_name, task_type, status_code
  target_start_date, target_end_date, act_start_date, act_end_date
  target_drtn_hr_cnt (duration hours), total_float_hr_cnt (float hours)
  phys_complete_pct, cstr_type, cstr_date, wbs_id, clndr_id

taskpred:
  task_id (successor), pred_task_id (predecessor)
  pred_type (PR_FS, PR_SS, PR_FF, PR_SF), lag_hr_cnt

taskrsrc:
  task_id, rsrc_id, target_qty, act_reg_qty, target_cost, act_reg_cost

wbs (PROJWBS):
  wbs_id, wbs_name, parent_wbs_id, seq_num

rsrc (RSRC):
  rsrc_id, rsrc_name, rsrc_type, clndr_id, unit_id

TASK TYPES: TT_Task (regular), TT_Mile (start milestone), TT_FinMile (finish milestone), TT_LOE (level of effort)
STATUS CODES: TK_NotStart, TK_Active, TK_Complete
CONSTRAINT TYPES: CS_MSOA (start on or after), CS_MEOA (end on or after), CS_MEOB (end on or before), CS_MSO (must start on), CS_MEO (must end on)

CRITICAL CODING RULES:
1. Always .copy() before modifying a DataFrame
2. Date fields may be datetime objects or strings — always use pd.to_datetime(col, errors='coerce')
3. Numeric fields may already be numeric but always use pd.to_numeric(col, errors='coerce').fillna(0) to be safe
4. Duration/float are in HOURS — divide by hours_per_day for days
5. For open-ended: work task IDs NOT in taskpred['pred_task_id'] (no successor)
6. For dangling: work task IDs NOT in taskpred['task_id'] (no predecessor)
7. Exclude TT_LOE and milestones from float/duration analyses
8. Convert datetime results to string with .isoformat() or str() for JSON serialization
9. Set result = ... at end (dict, list, number, or string — must be JSON-serializable)
10. Limit list results to 50 items max

EXAMPLE PATTERNS:

tasks = latest['df']['tasks'].copy()
tasks['dur_days'] = pd.to_numeric(tasks['target_drtn_hr_cnt'], errors='coerce').fillna(0) / hours_per_day
tasks['float_hrs'] = pd.to_numeric(tasks['total_float_hr_cnt'], errors='coerce').fillna(0)
tasks['pct'] = pd.to_numeric(tasks['phys_complete_pct'], errors='coerce').fillna(0)
tasks['start_dt'] = pd.to_datetime(tasks['target_start_date'], errors='coerce')
tasks['end_dt'] = pd.to_datetime(tasks['target_end_date'], errors='coerce')

work = tasks[~tasks['task_type'].isin(['TT_LOE', 'TT_Mile', 'TT_FinMile'])]
critical = work[work['float_hrs'] <= 0]

pred = latest['df']['taskpred'].copy()
pred['lag_days'] = pd.to_numeric(pred['lag_hr_cnt'], errors='coerce').fillna(0) / hours_per_day

# Comparing two updates
feb = get_update_by_month('feb')
mar = get_update_by_month('mar')
if feb and mar:
    feb_tasks = feb['df']['tasks'].copy()
    mar_tasks = mar['df']['tasks'].copy()
    merged = feb_tasks[['task_id','target_end_date']].merge(
        mar_tasks[['task_id','target_end_date']], on='task_id', suffixes=('_feb','_mar')
    )
    merged['feb_end'] = pd.to_datetime(merged['target_end_date_feb'], errors='coerce')
    merged['mar_end'] = pd.to_datetime(merged['target_end_date_mar'], errors='coerce')
    merged['slip_days'] = (merged['mar_end'] - merged['feb_end']).dt.days
    slipped = merged[merged['slip_days'] > 0].sort_values('slip_days', ascending=False)
    result = slipped[['task_id','slip_days']].head(20).to_dict('records')

Return ONLY valid Python code. No markdown, no explanations. Must set result = ..."""

    def get_response_prompt(self, user_query: str, basic_stats: Dict, code_result: Any,
                            code_success: bool, code_error: str = None) -> str:

        context = f"""USER QUESTION: {user_query}

PROJECT OVERVIEW:
- Project: {basic_stats.get('data_source', 'N/A')} | Data Date: {basic_stats.get('data_date', 'N/A')}
- Total Activities: {basic_stats.get('total_activities', 'N/A')} | Regular Tasks: {basic_stats.get('regular_tasks', 'N/A')} | Milestones: {basic_stats.get('milestones', 'N/A')} | LOE: {basic_stats.get('loe_activities', 'N/A')}
- Project Period: {basic_stats.get('project_start', 'N/A')} → {basic_stats.get('project_finish', 'N/A')}

STATUS:
- Completed: {basic_stats.get('completed', 'N/A')} | In Progress: {basic_stats.get('in_progress', 'N/A')} | Not Started: {basic_stats.get('not_started', 'N/A')}

CRITICAL PATH & FLOAT:
- Critical (float ≤ 0): {basic_stats.get('critical_count', 'N/A')} ({basic_stats.get('critical_pct', 'N/A')}%)
- Near-Critical (0 < float ≤ 14d): {basic_stats.get('near_critical_count', 'N/A')}
- Negative Float: {basic_stats.get('negative_float_count', 'N/A')}
- High Float (>30d): {basic_stats.get('high_float_count', 'N/A')}

SCHEDULE QUALITY:
- Open-Ended (no successor): {basic_stats.get('open_ended_count', 'N/A')}
- Dangling (no predecessor): {basic_stats.get('dangling_count', 'N/A')}
- Long Duration (>20d): {basic_stats.get('long_duration_count', 'N/A')}
- Constrained Activities: {basic_stats.get('constrained_activities', 'N/A')}

DELAY INDICATORS:
- Not-Started but overdue: {basic_stats.get('overdue_not_started', 'N/A')}
- In-Progress past target end: {basic_stats.get('overdue_in_progress', 'N/A')}

RELATIONSHIPS:
- Total: {basic_stats.get('total_relationships', 'N/A')} | With Lag: {basic_stats.get('relationships_with_lag', 'N/A')} | Negative Lags: {basic_stats.get('negative_lags', 'N/A')}

RESOURCES:
- Tasks with Resources: {basic_stats.get('tasks_with_resources', 'N/A')} ({basic_stats.get('resource_loaded_pct', 'N/A')}%)

DATA QUALITY:
- Duplicate PKs: {basic_stats.get('duplicate_pk_count', 0)} | Orphan FKs: {basic_stats.get('orphan_fk_count', 0)}

FILES:
- Baseline: {basic_stats.get('baseline_name', 'N/A')} ({basic_stats.get('baseline_date', 'N/A')})
- Updates loaded: {basic_stats.get('update_count', 0)}
"""

        if code_success and code_result:
            context += f"""
SPECIFIC ANALYSIS RESULTS (from executed code — use this as primary data source):
{json.dumps(code_result, indent=2, default=str)}
"""
        elif code_error:
            context += f"""
NOTE: Dynamic code analysis failed ({code_error}).
Answer using the pre-computed statistics above only. State clearly if the question cannot be fully answered.
"""

        context += """
INSTRUCTIONS:
- Answer the question directly using only the data provided above
- Lead with the most important finding
- Include specific numbers; never estimate or round beyond what the data shows
- Flag any schedule concerns identified in the data
- Provide actionable recommendations when issues are found
- If comparing files, show explicit before/after values
- Do not mention data that is shown as N/A"""

        return context
