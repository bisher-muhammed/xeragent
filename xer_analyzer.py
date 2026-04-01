"""
XER Schedule Analyzer
Orchestrates data store, insights engine, query executor, and LLM prompts.
"""

import json
import threading
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

EXEC_TIMEOUT_SECONDS = 30


# =============================================================================
# XERInsightsEngine  —  pre-computed insights (delays, critical path, resources)
# =============================================================================

class XERInsightsEngine:
    """
    Computes three insight sets from structured task data.
    All computation is done against pre-parsed task objects — no LLM involved.

    Results are stored as plain dicts/lists so they can be:
      - shown directly in the UI
      - injected into LLM prompts as ground truth
      - queried by LLM-generated code via the executor context
    """

    def compute(self, tasks_list: List[Dict], data_date: str) -> Dict:
        """
        Returns:
          delays          — tasks with start or finish slip vs plan
          critical_path   — ordered critical work tasks (by planned_start)
          resources       — aggregated cost/qty per resource across all tasks
        """
        try:
            dd = datetime.strptime(data_date[:10], '%Y-%m-%d') if data_date else None
        except ValueError:
            dd = None

        return {
            'delays':        self._delays(tasks_list, dd),
            'critical_path': self._critical_path(tasks_list),
            'resources':     self._resources(tasks_list),
        }

    # ── Delays ────────────────────────────────────────────────────────────────

    def _delays(self, tasks_list: List[Dict], data_date: datetime) -> List[Dict]:
        """
        For every task that has started or finished, compute slip in days:
          start_slip  = actual_start  - planned_start   (positive = late)
          finish_slip = actual_finish - planned_finish  (positive = late)

        For in-progress tasks with no actual_finish, finish_slip is estimated
        as: start_slip + (remaining_days - original_remaining_days).
        We flag any task where start_slip > 0 OR finish_slip > 0.
        Results sorted worst finish slip first.
        """
        rows = []
        for t in tasks_list:
            if t['task_type'] in ('TT_LOE',):
                continue
            if t['status'] == 'TK_NotStart':
                continue

            dates   = t['dates']
            p_start = _parse_date(dates.get('planned_start'))
            p_fin   = _parse_date(dates.get('planned_finish'))
            a_start = _parse_date(dates.get('actual_start'))
            a_fin   = _parse_date(dates.get('actual_finish'))

            start_slip  = _day_diff(a_start, p_start)   # None if either missing
            finish_slip = _day_diff(a_fin,   p_fin)

            # For in-progress tasks use data_date as proxy for actual finish
            if finish_slip is None and t['status'] == 'TK_Active' and data_date and p_fin:
                finish_slip = _day_diff(data_date, p_fin)

            # Only include if there is some slip
            if (start_slip or 0) <= 0 and (finish_slip or 0) <= 0:
                continue

            rows.append({
                'task_code':       t['task_code'],
                'task_name':       t['task_name'],
                'wbs_path':        t['wbs_path'],
                'status':          t['status'],
                'planned_start':   dates.get('planned_start'),
                'actual_start':    dates.get('actual_start'),
                'planned_finish':  dates.get('planned_finish'),
                'actual_finish':   dates.get('actual_finish'),
                'start_slip_days': start_slip,
                'finish_slip_days': finish_slip,
                'is_critical':     t['float']['is_critical'],
            })

        rows.sort(key=lambda r: (r['finish_slip_days'] or 0), reverse=True)
        return rows[:50]  # cap at 50

    # ── Critical path ─────────────────────────────────────────────────────────

    def _critical_path(self, tasks_list: List[Dict]) -> List[Dict]:
        """
        Returns critical work tasks ordered by planned_start.
        This is NOT a CPM re-calculation — it uses P6's pre-computed is_critical flag.
        The ordered list shows the sequence of tasks driving the project end date.
        """
        critical = [
            t for t in tasks_list
            if t['float']['is_critical']
            and t['task_type'] == 'TT_Task'
            and t['status'] != 'TK_Complete'
        ]
        critical.sort(key=lambda t: t['dates'].get('planned_start') or '')

        return [
            {
                'task_code':       t['task_code'],
                'task_name':       t['task_name'],
                'wbs_path':        t['wbs_path'],
                'status':          t['status'],
                'planned_start':   t['dates'].get('planned_start'),
                'planned_finish':  t['dates'].get('planned_finish'),
                'duration_days':   t['duration']['planned_days'],
                'remaining_days':  t['duration']['remaining_days'],
                'float_days':      t['float']['total_float_days'],
                'has_negative_float': t['float']['has_negative_float'],
                'predecessor_count': t['relationships']['predecessor_count'],
                'successor_count':   t['relationships']['successor_count'],
            }
            for t in critical
        ]

    # ── Resources ─────────────────────────────────────────────────────────────

    def _resources(self, tasks_list: List[Dict]) -> List[Dict]:
        """
        Aggregate planned/actual/remaining cost and qty per resource
        across all task assignments. Sort by planned_cost descending.
        """
        agg: Dict[str, Dict] = {}

        for t in tasks_list:
            for r in t.get('resources', []):
                name = r.get('resource_name') or r.get('resource_id', 'Unknown')
                if name not in agg:
                    agg[name] = {
                        'resource_name':   name,
                        'resource_type':   r.get('resource_type', ''),
                        'task_count':      0,
                        'planned_cost':    0.0,
                        'actual_cost':     0.0,
                        'remaining_cost':  0.0,
                        'planned_qty':     0.0,
                        'actual_qty':      0.0,
                        'remaining_qty':   0.0,
                    }
                a = agg[name]
                a['task_count']     += 1
                a['planned_cost']   += r.get('planned_cost',   0.0)
                a['actual_cost']    += r.get('actual_cost',    0.0)
                a['remaining_cost'] += r.get('remaining_cost', 0.0)
                a['planned_qty']    += r.get('planned_qty',    0.0)
                a['actual_qty']     += r.get('actual_qty',     0.0)
                a['remaining_qty']  += r.get('remaining_qty',  0.0)

        result = sorted(agg.values(), key=lambda x: x['planned_cost'], reverse=True)

        # Round floats
        for r in result:
            for k in ('planned_cost', 'actual_cost', 'remaining_cost',
                      'planned_qty',  'actual_qty',  'remaining_qty'):
                r[k] = round(r[k], 2)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers (module-level, used by insights engine)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(value) -> Optional[datetime]:
    if not value:
        return None
    s = str(value)[:10]
    try:
        return datetime.strptime(s, '%Y-%m-%d')
    except ValueError:
        return None


def _day_diff(later: Optional[datetime], earlier: Optional[datetime]) -> Optional[float]:
    """Return (later - earlier).days, or None if either is missing."""
    if later is None or earlier is None:
        return None
    return float((later - earlier).days)


# =============================================================================
# XERDataStore
# =============================================================================

class XERDataStore:
    """Stores structured XER data and provides fast access."""

    def __init__(self):
        self.baseline: Optional[Dict]    = None
        self.updates:  List[Dict]        = []
        self._cached_stats:  Optional[Dict] = None
        self._cached_insights: Optional[Dict] = None
        self._insights_engine = XERInsightsEngine()

    # ── Load / add / remove ───────────────────────────────

    def load_baseline(self, data: Dict, name: str, data_date: str):
        self.baseline        = self._build_source(data, name, data_date)
        self._cached_stats   = None
        self._cached_insights = None

    def add_update(self, data: Dict, name: str, data_date: str):
        self.updates.append(self._build_source(data, name, data_date))
        self.updates.sort(key=lambda x: x['data_date'])
        self._cached_stats   = None
        self._cached_insights = None

    def remove_update(self, index: int):
        if 0 <= index < len(self.updates):
            self.updates.pop(index)
            self._cached_stats   = None
            self._cached_insights = None

    def _build_source(self, data: Dict, name: str, data_date: str) -> Dict:
        tasks_list = data.get('tasks', [])
        df         = self._build_flat_dataframe(tasks_list)
        indexes    = self._build_indexes(tasks_list)
        return {
            'name':       name,
            'data_date':  data_date,
            'data':       data,
            'tasks_list': tasks_list,
            'df':         df,
            'indexes':    indexes,
            'summary':    data.get('summary', {}),
        }

    # ── DataFrame builder ─────────────────────────────────

    def _build_flat_dataframe(self, tasks_list: List[Dict]) -> Dict[str, pd.DataFrame]:
        if not tasks_list:
            return {}

        rows = []
        for t in tasks_list:
            rows.append({
                'task_id':    t.get('task_id', ''),
                'task_code':  t.get('task_code', ''),
                'task_name':  t.get('task_name', ''),
                'task_type':  t.get('task_type', ''),
                'status':     t.get('status', ''),
                'wbs_id':     t.get('wbs_id', ''),
                'wbs_path':   t.get('wbs_path', ''),
                'calendar_id': t.get('calendar_id', ''),
                'duration_days':    t['duration']['planned_days'],
                'remaining_days':   t['duration']['remaining_days'],
                'percent_complete': t['duration']['percent_complete'],
                'total_float_days':   t['float']['total_float_days'],
                'free_float_days':    t['float']['free_float_days'],
                'is_critical':        t['float']['is_critical'],
                'is_near_critical':   t['float']['is_near_critical'],
                'has_negative_float': t['float']['has_negative_float'],
                'planned_start':  t['dates'].get('planned_start'),
                'planned_finish': t['dates'].get('planned_finish'),
                'actual_start':   t['dates'].get('actual_start'),
                'actual_finish':  t['dates'].get('actual_finish'),
                'predecessor_count': t['relationships']['predecessor_count'],
                'successor_count':   t['relationships']['successor_count'],
                'is_open_ended':     t['relationships']['is_open_ended'],
                'is_dangling':       t['relationships']['is_dangling'],
                'has_resources':  t.get('has_resources', False),
                'has_constraint': t.get('constraints') is not None,
                'constraint_type': t['constraints']['type'] if t.get('constraints') else '',
            })

        tasks_df = pd.DataFrame(rows)
        for col in ['planned_start', 'planned_finish', 'actual_start', 'actual_finish']:
            tasks_df[col] = pd.to_datetime(tasks_df[col], errors='coerce')

        return {'tasks': tasks_df}

    # ── Index builder ─────────────────────────────────────

    def _build_indexes(self, tasks_list: List[Dict]) -> Dict:
        indexes = {
            'tasks_by_id':  {},
            'tasks_by_wbs': defaultdict(list),
            'successors':   {},
            'predecessors': {},
        }
        succ_map = defaultdict(list)
        pred_map = defaultdict(list)

        for t in tasks_list:
            tid = t['task_id']
            indexes['tasks_by_id'][tid] = t
            indexes['tasks_by_wbs'][t.get('wbs_id', '')].append(tid)
            for s in t['relationships']['successors']:
                succ_map[tid].append(s['task_id'])
            for p in t['relationships']['predecessors']:
                pred_map[tid].append(p['task_id'])

        indexes['successors']   = dict(succ_map)
        indexes['predecessors'] = dict(pred_map)
        indexes['tasks_by_wbs'] = dict(indexes['tasks_by_wbs'])
        return indexes

    # ── Accessors ─────────────────────────────────────────

    def get_latest(self) -> Optional[Dict]:
        return self.updates[-1] if self.updates else self.baseline

    def get_baseline(self) -> Optional[Dict]:
        return self.baseline

    def get_update_by_date(self, date_str: str) -> Optional[Dict]:
        for u in self.updates:
            if date_str in u['data_date'] or date_str in u['name']:
                return u
        return None

    def get_update_by_month(self, month: str, year: str = None) -> Optional[Dict]:
        month_map = {
            'jan': '01', 'january': '01',  '1': '01',  '01': '01',
            'feb': '02', 'february': '02', '2': '02',  '02': '02',
            'mar': '03', 'march': '03',    '3': '03',  '03': '03',
            'apr': '04', 'april': '04',    '4': '04',  '04': '04',
            'may': '05',                   '5': '05',  '05': '05',
            'jun': '06', 'june': '06',     '6': '06',  '06': '06',
            'jul': '07', 'july': '07',     '7': '07',  '07': '07',
            'aug': '08', 'august': '08',   '8': '08',  '08': '08',
            'sep': '09', 'september': '09','9': '09',  '09': '09',
            'oct': '10', 'october': '10',  '10': '10',
            'nov': '11', 'november': '11', '11': '11',
            'dec': '12', 'december': '12', '12': '12',
        }
        mn = month_map.get(month.lower().strip(), month.zfill(2))
        for u in self.updates:
            try:
                parsed = datetime.strptime(u['data_date'][:10], '%Y-%m-%d')
                if f"{parsed.month:02d}" == mn:
                    if year is None or str(parsed.year) == str(year):
                        return u
            except ValueError:
                continue
        return None

    # ── Stats ─────────────────────────────────────────────

    def compute_basic_stats(self) -> Dict:
        if self._cached_stats:
            return self._cached_stats

        source = self.get_latest()
        if not source:
            return {'error': 'No data loaded'}

        s        = source.get('summary', {})
        tasks_df = source['df'].get('tasks', pd.DataFrame())
        stats    = {'data_source': source['name'], 'data_date': source['data_date']}

        if not tasks_df.empty:
            total     = len(tasks_df)
            work_mask = ~tasks_df['task_type'].isin(['TT_LOE', 'TT_Mile', 'TT_FinMile'])
            stats.update({
                'total_activities':       total,
                'milestones':             s.get('milestones',      int(tasks_df['task_type'].isin(['TT_Mile','TT_FinMile']).sum())),
                'loe_activities':         s.get('loe',             int((tasks_df['task_type'] == 'TT_LOE').sum())),
                'regular_tasks':          s.get('work_tasks',      int(work_mask.sum())),
                'completed':              s.get('completed',       int((tasks_df['status'] == 'TK_Complete').sum())),
                'in_progress':            s.get('in_progress',     int((tasks_df['status'] == 'TK_Active').sum())),
                'not_started':            s.get('not_started',     int((tasks_df['status'] == 'TK_NotStart').sum())),
                'critical_count':         s.get('critical',        int(tasks_df['is_critical'].sum())),
                'near_critical_count':    s.get('near_critical',   int(tasks_df['is_near_critical'].sum())),
                'negative_float_count':   s.get('negative_float',  int(tasks_df['has_negative_float'].sum())),
                'open_ended_count':       s.get('open_ended',      int(tasks_df['is_open_ended'].sum())),
                'dangling_count':         s.get('dangling',        int(tasks_df['is_dangling'].sum())),
                'long_duration_count':    s.get('long_duration',   int((tasks_df['duration_days'] > 30).sum())),
                'constrained_activities': s.get('constrained',     int(tasks_df['has_constraint'].sum())),
                'resource_loaded_count':  s.get('resource_loaded', int(tasks_df['has_resources'].sum())),
                'avg_duration_days':      round(tasks_df.loc[work_mask,'duration_days'].mean(), 1) if work_mask.any() else 0,
                'project_start':  str(tasks_df['planned_start'].min())[:10]  if tasks_df['planned_start'].notna().any()  else 'N/A',
                'project_finish': str(tasks_df['planned_finish'].max())[:10] if tasks_df['planned_finish'].notna().any() else 'N/A',
            })
            stats['critical_pct']        = round(stats['critical_count']       / total * 100, 1) if total else 0
            stats['resource_loaded_pct'] = round(stats['resource_loaded_count']/ total * 100, 1) if total else 0

        raw_tables = source.get('data', {}).get('tables', {})
        taskpred   = raw_tables.get('TASKPRED', [])
        if taskpred:
            rel_types, lags_pos, lags_neg = defaultdict(int), 0, 0
            for r in taskpred:
                rel_types[r.get('pred_type', '')] += 1
                try:
                    lag = float(r.get('lag_hr_cnt', 0) or 0)
                    if lag > 0: lags_pos += 1
                    if lag < 0: lags_neg += 1
                except (ValueError, TypeError):
                    pass
            stats['total_relationships']    = len(taskpred)
            stats['relationship_types']     = dict(rel_types)
            stats['relationships_with_lag'] = lags_pos
            stats['negative_lags']          = lags_neg
        else:
            stats['total_relationships']    = 'unavailable (raw tables not in source)'
            stats['relationships_with_lag'] = 'unavailable'
            stats['negative_lags']          = 'unavailable'

        stats['baseline_name'] = self.baseline['name']      if self.baseline else None
        stats['baseline_date'] = self.baseline['data_date'] if self.baseline else None
        stats['update_count']  = len(self.updates)
        stats['updates']       = [{'name': u['name'], 'date': u['data_date']} for u in self.updates]

        self._cached_stats = stats
        return stats

    # ── Insights ──────────────────────────────────────────

    def compute_insights(self) -> Dict:
        if self._cached_insights:
            return self._cached_insights

        source = self.get_latest()
        if not source:
            return {}

        self._cached_insights = self._insights_engine.compute(
            source['tasks_list'],
            source['data_date'],
        )
        return self._cached_insights


# =============================================================================
# XERQueryExecutor
# =============================================================================

class XERQueryExecutor:
    """Executes LLM-generated Python code with a timeout."""

    def __init__(self, data_store: XERDataStore, timeout: int = EXEC_TIMEOUT_SECONDS):
        self.data_store = data_store
        self.timeout    = timeout

    def execute(self, code: str) -> Dict[str, Any]:
        latest   = self.data_store.get_latest()
        baseline = self.data_store.get_baseline()

        def _idx(s): return s.get('indexes', {}) if s else {}

        context = {
            'pd':       pd,
            'datetime': datetime,
            'json':     json,

            'latest':   latest,
            'baseline': baseline,
            'updates':  self.data_store.updates,

            'tasks_list':          latest.get('tasks_list', [])                        if latest   else [],
            'baseline_tasks_list': baseline.get('tasks_list', [])                      if baseline else [],
            'tasks_df':            latest['df'].get('tasks', pd.DataFrame())            if latest   else pd.DataFrame(),
            'baseline_tasks_df':   baseline['df'].get('tasks', pd.DataFrame())          if baseline else pd.DataFrame(),

            'task_by_id':   _idx(latest).get('tasks_by_id', {}),
            'tasks_by_wbs': _idx(latest).get('tasks_by_wbs', {}),
            'successors':   _idx(latest).get('successors', {}),
            'predecessors': _idx(latest).get('predecessors', {}),

            # Pre-computed insights available for LLM code to reference
            'insights': self.data_store.compute_insights(),

            'get_update_by_month': self.data_store.get_update_by_month,
            'get_update_by_date':  self.data_store.get_update_by_date,
            'result': None,
        }

        exec_result: Dict[str, Any] = {}

        def _run():
            try:
                exec(code, context)  # noqa: S102
                exec_result['result'] = context.get('result')
            except Exception as e:
                exec_result['error'] = str(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            return {'success': False, 'error': f'Timed out after {self.timeout}s'}

        if 'error' in exec_result:
            return {'success': False, 'error': exec_result['error']}

        result = exec_result.get('result')
        if isinstance(result, pd.DataFrame):
            result = result.head(50).to_dict('records')

        try:
            result = json.loads(json.dumps(result, default=str))
        except Exception as e:
            return {'success': False, 'error': f'Result not JSON serialisable: {e}'}

        return {'success': True, 'result': result}


# =============================================================================
# XERAnalyzer  —  public API
# =============================================================================

class XERAnalyzer:

    def __init__(self):
        self.data_store = XERDataStore()
        self.executor   = XERQueryExecutor(self.data_store)

    def load_baseline(self, data: Dict, name: str = None, data_date: str = None):
        name      = name      or data.get('project', {}).get('project_name', 'Baseline')
        data_date = data_date or (data.get('project', {}).get('data_date') or '')[:10]
        self.data_store.load_baseline(data, name, data_date)

    def add_update(self, data: Dict, name: str = None, data_date: str = None):
        name      = name      or data.get('project', {}).get('project_name', 'Update')
        data_date = data_date or (data.get('project', {}).get('data_date') or '')[:10]
        self.data_store.add_update(data, name, data_date)

    def remove_update(self, index: int):
        self.data_store.remove_update(index)

    def get_basic_stats(self) -> Dict:
        return self.data_store.compute_basic_stats()

    def get_insights(self) -> Dict:
        return self.data_store.compute_insights()

    def execute_code(self, code: str) -> Dict:
        return self.executor.execute(code)

    # ── Prompts ───────────────────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        return """You are an expert Primavera P6 Schedule Analyst AI Assistant.
You analyze construction project schedules from XER files.

RULES — NEVER BREAK THESE:
1. Only use data from the variables provided. Never invent task names, dates, or numbers.
2. If data is unavailable, say so clearly. Never guess.
3. All analysis must reference actual task_code or task_name values from the data.
4. Numeric fields are already floats — never call pd.to_numeric() on them.
5. Boolean fields (is_critical, is_open_ended, etc.) are already True/False.

RESPONSE STYLE:
- Professional, direct, specific numbers and percentages
- Use markdown tables for lists of activities
- Bullet points for summaries
- Always recommend actions when issues are found"""

    def get_code_generation_prompt(
        self,
        user_query: str,
        basic_stats: Dict,
        previous_code: str = None,
        previous_error: str = None,
        conversation_history: List[Dict] = None,
    ) -> str:
        source  = self.data_store.get_latest()
        df_cols = ''
        if source and 'df' in source:
            df = source['df'].get('tasks', pd.DataFrame())
            if not df.empty:
                df_cols = ', '.join(df.columns.tolist())

        retry_block = ''
        if previous_code and previous_error:
            retry_block = f"""
PREVIOUS ATTEMPT FAILED — FIX THIS ERROR:
Error: {previous_error}
Failed code:
{previous_code}
Rewrite fixing the error above.
"""
        history_block = ''
        if conversation_history:
            lines = []
            for msg in conversation_history[-6:]:
                lines.append(f"{msg.get('role','user').upper()}: {str(msg.get('content',''))[:400]}")
            history_block = 'RECENT CONVERSATION:\n' + '\n'.join(lines) + '\n'

        return f"""Generate Python code to answer this question about a Primavera P6 schedule.

USER QUESTION: {user_query}

{history_block}
{retry_block}

── PRE-COMPUTED STATS ────────────────────────────────────
{json.dumps(basic_stats, indent=2, default=str)}

── AVAILABLE VARIABLES ───────────────────────────────────
tasks_list       : list of structured task dicts
tasks_df         : flat pandas DataFrame (bulk filtering)
baseline_tasks_list / baseline_tasks_df : same for baseline
get_update_by_month('mar') : returns update source dict
  → df:   get_update_by_month('mar')['df']['tasks']
  → list: get_update_by_month('mar')['tasks_list']

insights         : pre-computed insights dict with keys:
  insights['delays']         → list of delayed tasks (start_slip_days, finish_slip_days)
  insights['critical_path']  → ordered list of critical work tasks
  insights['resources']      → list of resources with aggregated cost/qty

FAST INDEXES:
  task_by_id[task_id]      → full task dict
  tasks_by_wbs[wbs_id]     → list of task_ids
  successors[task_id]      → list of successor task_ids
  predecessors[task_id]    → list of predecessor task_ids

── tasks_df COLUMNS ──────────────────────────────────────
{df_cols}

── TASK OBJECT STRUCTURE ─────────────────────────────────
task['task_id'], task['task_code'], task['task_name']
task['task_type']  → 'TT_Task', 'TT_Mile', 'TT_FinMile', 'TT_LOE'
task['status']     → 'TK_NotStart', 'TK_Active', 'TK_Complete'
task['wbs_path']   → e.g. "PROJECT > CIVIL > FOUNDATIONS"
task['duration']['planned_days'], task['duration']['remaining_days'], task['duration']['percent_complete']
task['float']['total_float_days'], task['float']['is_critical'], task['float']['has_negative_float']
task['relationships']['predecessors'] → [{{task_id, task_code, task_name, type, lag_days}}]
task['relationships']['successors']   → same format
task['relationships']['is_open_ended'], task['relationships']['is_dangling']
task['resources']   → [{{resource_name, planned_qty, actual_qty, planned_cost, actual_cost}}]
task['constraints'] → {{'type': 'CS_MSOA', 'date': '2024-03-01'}} or None
task['notes']       → [{{memo_type, text}}]
task['period_actuals'] → [{{period_name, period_start, period_end, actual_cost, actual_work_qty}}]

── CORRECT PATTERNS ──────────────────────────────────────

# Use insights directly for delay/critical/resource questions
result = insights['delays'][:20]

# Critical path sequence
result = insights['critical_path']

# Resource summary
result = insights['resources']

# Bulk filter
critical = tasks_df[tasks_df['is_critical'] & (tasks_df['task_type'] == 'TT_Task')]
result = critical[['task_code','task_name','wbs_path','total_float_days']].to_dict('records')

# Baseline comparison
merged = baseline_tasks_df[['task_id','duration_days']].merge(
    tasks_df[['task_id','task_code','task_name','duration_days']],
    on='task_id', suffixes=('_baseline','_latest')
)
merged['change'] = merged['duration_days_latest'] - merged['duration_days_baseline']
result = merged[merged['change'] != 0].sort_values('change', ascending=False).head(20).to_dict('records')

── RULES ─────────────────────────────────────────────────
1. Set result = ... at the end
2. Max 50 items in any list
3. result must be JSON serialisable
4. NEVER invent data

Return ONLY valid Python code. No explanations."""

    def get_response_prompt(
        self,
        user_query: str,
        basic_stats: Dict,
        code_result: Any,
        code_success: bool,
        code_error: str = None,
        insights: Dict = None,
    ) -> str:
        insights = insights or {}

        # Summarise delays for context
        delays       = insights.get('delays', [])
        delayed_count = len(delays)
        worst_delays  = delays[:3]  # top 3 for context block
        worst_str = ', '.join(
            d['task_code'] + ' (' + str(d['finish_slip_days']) + 'd slip)'
            for d in worst_delays
        )
        delay_line = str(delayed_count) + (' — worst: ' + worst_str if worst_str else '')

        # Resource totals
        resources    = insights.get('resources', [])
        total_planned_cost = sum(r.get('planned_cost', 0) for r in resources)

        # Critical path length
        cp           = insights.get('critical_path', [])
        cp_remaining = sum(t.get('remaining_days', 0) for t in cp)

        ctx = f"""USER QUESTION: {user_query}

PROJECT OVERVIEW:
- Project: {basic_stats.get('data_source','N/A')} | Data Date: {basic_stats.get('data_date','N/A')}
- Total Activities: {basic_stats.get('total_activities','N/A')}
- Period: {basic_stats.get('project_start','N/A')} to {basic_stats.get('project_finish','N/A')}

STATUS: Completed={basic_stats.get('completed','N/A')} | In Progress={basic_stats.get('in_progress','N/A')} | Not Started={basic_stats.get('not_started','N/A')}

SCHEDULE HEALTH:
- Critical: {basic_stats.get('critical_count','N/A')} ({basic_stats.get('critical_pct','N/A')}%) | Near-Critical: {basic_stats.get('near_critical_count','N/A')}
- Negative Float: {basic_stats.get('negative_float_count','N/A')} | Open-Ended: {basic_stats.get('open_ended_count','N/A')} | Dangling: {basic_stats.get('dangling_count','N/A')}
- Long Duration >30d: {basic_stats.get('long_duration_count','N/A')} | Constrained: {basic_stats.get('constrained_activities','N/A')}

RELATIONSHIPS: Total={basic_stats.get('total_relationships','N/A')} | With Lag={basic_stats.get('relationships_with_lag','N/A')} | Negative Lags={basic_stats.get('negative_lags','N/A')}
RESOURCES: {basic_stats.get('resource_loaded_count','N/A')} tasks loaded | Total Planned Cost: {total_planned_cost:,.0f}
FILES: Baseline={basic_stats.get('baseline_name','N/A')} ({basic_stats.get('baseline_date','N/A')}) | Updates={basic_stats.get('update_count',0)}

INSIGHTS SUMMARY:
- Delayed tasks: {delay_line}
- Critical path: {len(cp)} tasks remaining | {cp_remaining:.0f} days total remaining work
- Resources: {len(resources)} resources tracked
"""
        if code_success and code_result:
            ctx += f"\nSPECIFIC ANALYSIS RESULTS (use these — do not contradict them):\n{json.dumps(code_result, indent=2, default=str)}\n"
        elif code_error:
            ctx += f"\nNote: Code analysis failed ({code_error}). Use stats and insights above.\n"

        ctx += """
INSTRUCTIONS:
1. Answer ONLY from the data provided. Never invent activities, dates, or numbers.
2. If analysis results are present, base your answer primarily on those.
3. Include specific task codes and names when listing activities.
4. Highlight concerns with clear recommendations.
5. Use markdown tables for lists of 5+ items.
6. If comparing files, show a clear before/after table."""

        return ctx

