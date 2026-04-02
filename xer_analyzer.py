"""
XER Schedule Analyzer - Robust LLM-Powered Analysis
Comprehensive context and reliable code generation
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helper: safe column access on DataFrames
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column Series if it exists, else an empty Series of dtype object."""
    if name in df.columns:
        return df[name]
    return pd.Series(dtype=object)


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric column, coercing strings and filling NaN with 0."""
    if name not in df.columns:
        return pd.Series(0, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors='coerce').fillna(0)


def _safe_records(df: pd.DataFrame, columns: List[str]) -> List[Dict]:
    """Return records keeping only columns that exist; fills NaN with empty string."""
    available = [c for c in columns if c in df.columns]
    if not available or df.empty:
        return []
    return df[available].fillna('').to_dict('records')


# ---------------------------------------------------------------------------
# XERDataStore
# ---------------------------------------------------------------------------

class XERDataStore:
    """Stores all XER data with pre-computed statistics."""

    # Activities with these task_types are excluded from most quality checks
    LOE_TYPES = {'TT_LOE'}
    MILESTONE_TYPES = {'TT_Mile', 'TT_FinMile'}
    EXCLUDED_TYPES = LOE_TYPES | MILESTONE_TYPES

    def __init__(self):
        self.baseline: Optional[Dict] = None
        self.updates: List[Dict] = []
        self.hours_per_day: int = 10
        self._cached_stats: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Load / manage data
    # ------------------------------------------------------------------

    def load_baseline(self, data: Dict, name: str, data_date: str):
        """Load baseline data."""
        self.baseline = {
            'name': name,
            'data_date': data_date,
            'data': data,
            'df': self._create_dataframes(data),
        }
        self._cached_stats = None

    def add_update(self, data: Dict, name: str, data_date: str):
        """Add an update file (sorted chronologically)."""
        self.updates.append({
            'name': name,
            'data_date': data_date,
            'data': data,
            'df': self._create_dataframes(data),
        })
        self.updates.sort(key=lambda x: x['data_date'])
        self._cached_stats = None

    def remove_update(self, index: int):
        """Remove an update by index."""
        if 0 <= index < len(self.updates):
            self.updates.pop(index)
            self._cached_stats = None

    def get_latest(self) -> Optional[Dict]:
        """Get latest data (most recent update, or baseline)."""
        if self.updates:
            return self.updates[-1]
        return self.baseline

    def get_baseline(self) -> Optional[Dict]:
        return self.baseline

    def get_update_by_date(self, date_str: str) -> Optional[Dict]:
        """Find update by date (partial match on data_date or name)."""
        for update in self.updates:
            if date_str in update['data_date'] or date_str in update['name']:
                return update
        return None

    def get_update_by_month(self, month: str, year: str = None) -> Optional[Dict]:
        """Find update by month name or number, with optional year filter."""
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
            'dec': '12', 'december': '12', '12': '12',
        }
        month_num = month_map.get(month.lower(), month)

        for update in self.updates:
            data_date = update['data_date']
            if len(data_date) >= 7:
                file_month = data_date[5:7]
                file_year = data_date[:4]
                if file_month == month_num:
                    if year is None or file_year == str(year):
                        return update
        return None

    # ------------------------------------------------------------------
    # DataFrame creation
    # ------------------------------------------------------------------

    def _create_dataframes(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """Convert XER tables to pandas DataFrames."""
        dfs: Dict[str, pd.DataFrame] = {}

        # Tasks and WBS from top-level convenience lists
        if data.get('tasks'):
            dfs['tasks'] = pd.DataFrame(data['tasks'])

        if data.get('wbs'):
            dfs['wbs'] = pd.DataFrame(data['wbs'])

        # All raw tables (lower-cased key to avoid case sensitivity issues)
        for table_name, records in data.get('tables', {}).items():
            key = table_name.lower()
            if records and key not in dfs:
                try:
                    dfs[key] = pd.DataFrame(records)
                except Exception:
                    pass  # Skip malformed tables silently

        return dfs

    # ------------------------------------------------------------------
    # Statistics computation
    # ------------------------------------------------------------------

    def compute_basic_stats(self) -> Dict:
        """Compute comprehensive statistics that are always available."""
        if self._cached_stats is not None:
            return self._cached_stats

        source = self.get_latest()
        if not source or 'tasks' not in source.get('df', {}):
            return {'error': 'No data loaded'}

        tasks_df = source['df']['tasks'].copy()
        stats: Dict[str, Any] = {}

        # Basic counts
        stats['total_activities'] = len(tasks_df)
        stats['data_source'] = source['name']
        stats['data_date'] = source['data_date']

        # ----- Task types -----
        task_type_col = _col(tasks_df, 'task_type')
        if not task_type_col.empty:
            type_counts = task_type_col.value_counts().to_dict()
            stats['task_types'] = type_counts
            stats['milestones'] = (
                type_counts.get('TT_Mile', 0) + type_counts.get('TT_FinMile', 0)
            )
            stats['loe_activities'] = type_counts.get('TT_LOE', 0)
            stats['regular_tasks'] = type_counts.get('TT_Task', 0)
        else:
            stats['milestones'] = 0
            stats['loe_activities'] = 0
            stats['regular_tasks'] = 0

        # Work tasks mask (excludes LOE and milestones)
        if 'task_type' in tasks_df.columns:
            work_mask = ~tasks_df['task_type'].isin(self.EXCLUDED_TYPES)
            ns_mask = ~tasks_df['task_type'].isin(self.LOE_TYPES)  # keep milestones for NS check
        else:
            work_mask = pd.Series(True, index=tasks_df.index)
            ns_mask = work_mask

        work_tasks = tasks_df[work_mask].copy()

        # ----- Status breakdown -----
        status_col = _col(tasks_df, 'status_code')
        if not status_col.empty:
            status_counts = status_col.value_counts().to_dict()
            stats['status_breakdown'] = status_counts
            stats['completed'] = status_counts.get('TK_Complete', 0)
            stats['in_progress'] = status_counts.get('TK_Active', 0)
            stats['not_started'] = status_counts.get('TK_NotStart', 0)
        else:
            stats['completed'] = 0
            stats['in_progress'] = 0
            stats['not_started'] = 0

        # ----- Duration analysis -----
        work_tasks['_duration_hrs'] = _num_col(work_tasks, 'target_drtn_hr_cnt')
        work_tasks['_duration_days'] = work_tasks['_duration_hrs'] / self.hours_per_day

        if len(work_tasks) > 0:
            # Threshold shown in sidebar is 20 days
            stats['long_duration_count'] = int(
                (work_tasks['_duration_days'] > 20).sum()
            )
            stats['avg_duration_days'] = round(work_tasks['_duration_days'].mean(), 1)
            stats['max_duration_days'] = round(work_tasks['_duration_days'].max(), 1)
        else:
            stats['long_duration_count'] = 0
            stats['avg_duration_days'] = 0.0
            stats['max_duration_days'] = 0.0

        # ----- Float / Critical path -----
        work_tasks['_float_hrs'] = _num_col(work_tasks, 'total_float_hr_cnt')
        work_tasks['_float_days'] = work_tasks['_float_hrs'] / self.hours_per_day

        if len(work_tasks) > 0:
            critical_mask = work_tasks['_float_hrs'] <= 0
            near_critical_mask = (work_tasks['_float_hrs'] > 0) & (
                work_tasks['_float_hrs'] <= 100
            )
            negative_mask = work_tasks['_float_hrs'] < 0

            stats['critical_count'] = int(critical_mask.sum())
            stats['critical_pct'] = round(
                critical_mask.sum() / len(work_tasks) * 100, 1
            )
            stats['near_critical_count'] = int(near_critical_mask.sum())
            stats['negative_float_count'] = int(negative_mask.sum())
        else:
            stats['critical_count'] = 0
            stats['critical_pct'] = 0.0
            stats['near_critical_count'] = 0
            stats['negative_float_count'] = 0

        # ----- Relationships -----
        pred_df_key = 'taskpred'
        if pred_df_key in source['df']:
            pred_df = source['df'][pred_df_key].copy()
            stats['total_relationships'] = len(pred_df)

            if 'pred_type' in pred_df.columns:
                stats['relationship_types'] = pred_df['pred_type'].value_counts().to_dict()

            pred_df['_lag'] = _num_col(pred_df, 'lag_hr_cnt')
            stats['relationships_with_lag'] = int((pred_df['_lag'] > 0).sum())
            stats['negative_lags'] = int((pred_df['_lag'] < 0).sum())

            # Open-ended: no successor
            all_task_ids = set(_col(tasks_df, 'task_id').dropna().tolist())
            has_successor = set(_col(pred_df, 'pred_task_id').dropna().tolist())
            has_predecessor = set(_col(pred_df, 'task_id').dropna().tolist())

            if 'task_type' in tasks_df.columns:
                work_only_ids = set(
                    tasks_df.loc[work_mask, 'task_id'].dropna().tolist()
                )
            else:
                work_only_ids = all_task_ids

            stats['open_ended_count'] = len((all_task_ids - has_successor) & work_only_ids)
            stats['dangling_count'] = len((all_task_ids - has_predecessor) & work_only_ids)
        else:
            stats['total_relationships'] = 0
            stats['relationships_with_lag'] = 0
            stats['negative_lags'] = 0
            stats['open_ended_count'] = 0
            stats['dangling_count'] = 0

        # ----- Constraints -----
        cstr_col = _col(tasks_df, 'cstr_type')
        if not cstr_col.empty:
            constrained = tasks_df[cstr_col.notna() & (cstr_col != '')]
            stats['constrained_activities'] = len(constrained)
            if len(constrained) > 0:
                stats['constraint_types'] = cstr_col[
                    cstr_col.notna() & (cstr_col != '')
                ].value_counts().to_dict()
        else:
            stats['constrained_activities'] = 0

        # ----- Resources -----
        rsrc_df_key = 'taskrsrc'
        if rsrc_df_key in source['df']:
            rsrc_df = source['df'][rsrc_df_key]
            stats['resource_assignments'] = len(rsrc_df)
            tasks_with_resources = _col(rsrc_df, 'task_id').nunique()
            stats['tasks_with_resources'] = int(tasks_with_resources)
            total = len(tasks_df)
            stats['resource_loaded_pct'] = round(
                tasks_with_resources / total * 100, 1
            ) if total > 0 else 0.0
        else:
            stats['resource_assignments'] = 0
            stats['tasks_with_resources'] = 0
            stats['resource_loaded_pct'] = 0.0

        # ----- Overdue activities (relative to data date) -----
        data_date_str = source.get('data_date', '')
        stats['overdue_not_started'] = 0
        stats['overdue_in_progress'] = 0

        if data_date_str:
            try:
                data_date = pd.to_datetime(data_date_str, errors='coerce')
                if pd.notna(data_date):
                    # Not started but past target start
                    if 'status_code' in tasks_df.columns and 'target_start_date' in tasks_df.columns:
                        ns_df = tasks_df[tasks_df['status_code'] == 'TK_NotStart'].copy()
                        ns_df['_ts'] = pd.to_datetime(
                            _col(ns_df, 'target_start_date'), errors='coerce'
                        )
                        stats['overdue_not_started'] = int(
                            (ns_df['_ts'] < data_date).sum()
                        )

                    # In progress but past target end
                    if 'status_code' in tasks_df.columns and 'target_end_date' in tasks_df.columns:
                        ip_df = tasks_df[tasks_df['status_code'] == 'TK_Active'].copy()
                        ip_df['_te'] = pd.to_datetime(
                            _col(ip_df, 'target_end_date'), errors='coerce'
                        )
                        stats['overdue_in_progress'] = int(
                            (ip_df['_te'] < data_date).sum()
                        )
            except Exception:
                pass  # Data date parse failed; leave as 0

        # ----- Date range -----
        start_col = _col(tasks_df, 'target_start_date')
        if not start_col.empty:
            parsed = pd.to_datetime(start_col, errors='coerce').dropna()
            if not parsed.empty:
                stats['project_start'] = str(parsed.min())[:10]

        end_col = _col(tasks_df, 'target_end_date')
        if not end_col.empty:
            parsed = pd.to_datetime(end_col, errors='coerce').dropna()
            if not parsed.empty:
                stats['project_finish'] = str(parsed.max())[:10]

        # ----- Calendars -----
        cal_key = 'calendar'
        if cal_key in source['df']:
            cal_df = source['df'][cal_key]
            stats['calendar_count'] = len(cal_df)
            if 'clndr_name' in cal_df.columns:
                stats['calendars'] = cal_df['clndr_name'].dropna().tolist()

        # ----- Files info -----
        stats['baseline_name'] = self.baseline['name'] if self.baseline else None
        stats['baseline_date'] = self.baseline['data_date'] if self.baseline else None
        stats['update_count'] = len(self.updates)
        stats['updates'] = [
            {'name': u['name'], 'date': u['data_date']} for u in self.updates
        ]

        self._cached_stats = stats
        return stats


# ---------------------------------------------------------------------------
# InsightsDataLayer
# ---------------------------------------------------------------------------

class InsightsDataLayer:
    """
    Pre-computed analytical insights for delays, critical path, and resources.
    Sits on top of XERDataStore; feeds both the sidebar UI and LLM context.
    Cache is keyed per insight category and invalidated whenever data changes.
    """

    # Float threshold (hours) below which a task is "near-critical" (8 working days)
    NEAR_CRITICAL_HRS = 80

    def __init__(self, data_store: XERDataStore):
        self.data_store = data_store
        self._cache: Dict[str, Any] = {}

    def invalidate(self):
        """Drop all cached insight results — call after any data load/remove."""
        self._cache = {}

    # ------------------------------------------------------------------
    # Public façade
    # ------------------------------------------------------------------

    def get_all_insights(self) -> Dict:
        """Return all three insight categories in one dict."""
        return {
            'delays': self.get_delay_insights(),
            'critical_path': self.get_critical_path_insights(),
            'resources': self.get_resource_insights(),
        }

    # ------------------------------------------------------------------
    # 1. Delay Insights
    # ------------------------------------------------------------------

    def get_delay_insights(self) -> Dict:
        """
        Compute delay-related insights:
        - Project end-date slippage vs baseline
        - Per-activity slippage (most delayed activities)
        - Float erosion (activities that newly crossed into negative float)
        - Overdue not-started and in-progress activities with days overdue
        - Float trend across all loaded update files
        """
        key = 'delays'
        if key in self._cache:
            return self._cache[key]

        out: Dict[str, Any] = {
            'project_end_slippage_days': None,
            'baseline_project_end': None,
            'current_project_end': None,
            'slipped_activities_count': 0,
            'gained_negative_float_count': 0,
            'delayed_not_started': [],
            'delayed_in_progress': [],
            'most_slipped_activities': [],
            'float_trend': [],
        }

        source = self.data_store.get_latest()
        baseline = self.data_store.get_baseline()
        hpd = self.data_store.hours_per_day

        if not source or 'tasks' not in source.get('df', {}):
            self._cache[key] = out
            return out

        tasks = source['df']['tasks'].copy()

        # Work tasks mask (exclude LOE; keep milestones for overdue check)
        if 'task_type' in tasks.columns:
            work_mask = ~tasks['task_type'].isin({'TT_LOE', 'TT_Mile', 'TT_FinMile'})
        else:
            work_mask = pd.Series(True, index=tasks.index)
        work = tasks[work_mask].copy()

        data_date = self._parse_date(source.get('data_date', ''))

        # ---- Overdue not-started ----
        if (data_date is not None
                and 'status_code' in work.columns
                and 'target_start_date' in work.columns):
            ns = work[work['status_code'] == 'TK_NotStart'].copy()
            ns['_ts'] = pd.to_datetime(ns['target_start_date'], errors='coerce')
            ns_over = ns[ns['_ts'] < data_date].copy()
            if not ns_over.empty:
                ns_over['days_overdue'] = (data_date - ns_over['_ts']).dt.days
                out['delayed_not_started'] = _safe_records(
                    ns_over.nlargest(20, 'days_overdue'),
                    ['task_id', 'task_code', 'task_name', 'target_start_date', 'days_overdue'],
                )

        # ---- Overdue in-progress ----
        if (data_date is not None
                and 'status_code' in work.columns
                and 'target_end_date' in work.columns):
            ip = work[work['status_code'] == 'TK_Active'].copy()
            ip['_te'] = pd.to_datetime(ip['target_end_date'], errors='coerce')
            ip_over = ip[ip['_te'] < data_date].copy()
            if not ip_over.empty:
                ip_over['days_overdue'] = (data_date - ip_over['_te']).dt.days
                out['delayed_in_progress'] = _safe_records(
                    ip_over.nlargest(20, 'days_overdue'),
                    ['task_id', 'task_code', 'task_name', 'target_end_date', 'days_overdue'],
                )

        # ---- Baseline comparison ----
        if baseline and 'tasks' in baseline.get('df', {}):
            bl = baseline['df']['tasks'].copy()

            # Project end-date movement
            bl_end = pd.to_datetime(_col(bl, 'target_end_date'), errors='coerce').max()
            cu_end = pd.to_datetime(_col(tasks, 'target_end_date'), errors='coerce').max()
            if pd.notna(bl_end) and pd.notna(cu_end):
                out['project_end_slippage_days'] = int((cu_end - bl_end).days)
                out['baseline_project_end'] = str(bl_end)[:10]
                out['current_project_end'] = str(cu_end)[:10]

            # Activity-level slippage
            bl_slim = pd.DataFrame({'task_id': _col(bl, 'task_id')})
            if 'target_end_date' in bl.columns:
                bl_slim['bl_end_date'] = bl['target_end_date'].values
            if 'target_start_date' in bl.columns:
                bl_slim['bl_start_date'] = bl['target_start_date'].values
            if 'total_float_hr_cnt' in bl.columns:
                bl_slim['bl_float_hrs'] = bl['total_float_hr_cnt'].values

            merged = work.merge(bl_slim, on='task_id', how='inner')

            if 'bl_end_date' in merged.columns and 'target_end_date' in merged.columns:
                merged['_cu_end'] = pd.to_datetime(merged['target_end_date'], errors='coerce')
                merged['_bl_end'] = pd.to_datetime(merged['bl_end_date'], errors='coerce')
                merged['slippage_days'] = (
                    (merged['_cu_end'] - merged['_bl_end']).dt.days.fillna(0).astype(int)
                )
                slipped = merged[merged['slippage_days'] > 0]
                out['slipped_activities_count'] = len(slipped)
                if not slipped.empty:
                    out['most_slipped_activities'] = _safe_records(
                        slipped.nlargest(20, 'slippage_days'),
                        ['task_id', 'task_code', 'task_name',
                         'bl_end_date', 'target_end_date', 'slippage_days'],
                    )

            # Float erosion: was >= 0 in baseline, now < 0
            if 'bl_float_hrs' in merged.columns and 'total_float_hr_cnt' in merged.columns:
                merged['_cu_float'] = _num_col(merged, 'total_float_hr_cnt')
                merged['_bl_float'] = pd.to_numeric(
                    merged['bl_float_hrs'], errors='coerce'
                ).fillna(0)
                gained_neg = merged[
                    (merged['_bl_float'] >= 0) & (merged['_cu_float'] < 0)
                ]
                out['gained_negative_float_count'] = len(gained_neg)

        # ---- Float trend across updates ----
        if self.data_store.updates:
            trend = []
            for upd in self.data_store.updates:
                if 'tasks' in upd.get('df', {}):
                    t = upd['df']['tasks']
                    flt = _num_col(t, 'total_float_hr_cnt')
                    trend.append({
                        'update_name': upd['name'],
                        'data_date': upd['data_date'],
                        'negative_float_count': int((flt < 0).sum()),
                        'critical_count': int((flt <= 0).sum()),
                        'avg_float_days': round(float(flt.mean()) / hpd, 1),
                    })
            out['float_trend'] = trend

        self._cache[key] = out
        return out

    # ------------------------------------------------------------------
    # 2. Critical Path Insights
    # ------------------------------------------------------------------

    def get_critical_path_insights(self) -> Dict:
        """
        Compute critical path insights:
        - Full critical activity list with float / duration / WBS / status
        - Near-critical activities (float within NEAR_CRITICAL_HRS)
        - Critical activities grouped by WBS (count + total duration)
        - Changes vs baseline (newly critical / left critical path)
        - Hard-constrained activities on the critical path
        - Critical activities that have lags on their incoming logic ties
        """
        key = 'critical_path'
        if key in self._cache:
            return self._cache[key]

        hpd = self.data_store.hours_per_day
        out: Dict[str, Any] = {
            'critical_activities': [],
            'near_critical_activities': [],
            'critical_by_wbs': [],
            'critical_path_changes': [],
            'constrained_critical': [],
            'critical_with_lags': [],
            'near_critical_threshold_days': round(self.NEAR_CRITICAL_HRS / hpd, 1),
            'summary': {},
        }

        source = self.data_store.get_latest()
        if not source or 'tasks' not in source.get('df', {}):
            self._cache[key] = out
            return out

        tasks = source['df']['tasks'].copy()

        # Exclude LOE from critical path analysis (milestones are kept)
        if 'task_type' in tasks.columns:
            work = tasks[tasks['task_type'] != 'TT_LOE'].copy()
        else:
            work = tasks.copy()

        work['_float_hrs'] = _num_col(work, 'total_float_hr_cnt')
        work['_float_days'] = work['_float_hrs'] / hpd
        work['_dur_hrs'] = _num_col(work, 'target_drtn_hr_cnt')
        work['_dur_days'] = work['_dur_hrs'] / hpd
        work['_complete_pct'] = _num_col(work, 'phys_complete_pct')

        critical = work[work['_float_hrs'] <= 0].copy()
        near_crit = work[
            (work['_float_hrs'] > 0) & (work['_float_hrs'] <= self.NEAR_CRITICAL_HRS)
        ].copy()

        # ---- Full critical list ----
        out['critical_activities'] = _safe_records(
            critical.sort_values('_float_hrs'),
            ['task_id', 'task_code', 'task_name', 'task_type', 'status_code',
             '_float_days', '_dur_days', '_complete_pct',
             'target_start_date', 'target_end_date', 'wbs_id', 'cstr_type'],
        )

        # ---- Near-critical list ----
        out['near_critical_activities'] = _safe_records(
            near_crit.sort_values('_float_hrs'),
            ['task_id', 'task_code', 'task_name', 'status_code',
             '_float_days', '_dur_days', 'target_start_date', 'target_end_date'],
        )

        # ---- Summary ----
        out['summary'] = {
            'critical_count': len(critical),
            'near_critical_count': len(near_crit),
            'critical_pct': round(len(critical) / max(len(work), 1) * 100, 1),
            'total_critical_duration_days': round(float(critical['_dur_days'].sum()), 1),
            'avg_critical_float_days': round(float(critical['_float_hrs'].mean()) / hpd, 2)
            if not critical.empty else 0,
        }

        # ---- Critical by WBS ----
        if 'wbs_id' in critical.columns and not critical.empty:
            wbs_g = critical.groupby('wbs_id', as_index=False).agg(
                critical_count=('task_id', 'count'),
                total_duration_days=('_dur_days', 'sum'),
                avg_float_days=('_float_days', 'mean'),
            )
            wbs_g['total_duration_days'] = wbs_g['total_duration_days'].round(1)
            wbs_g['avg_float_days'] = wbs_g['avg_float_days'].round(2)

            # Join WBS names if available
            for key_wbs in ('projwbs', 'wbs'):
                if key_wbs in source['df']:
                    wdf = source['df'][key_wbs]
                    if 'wbs_id' in wdf.columns and 'wbs_name' in wdf.columns:
                        wbs_g = wbs_g.merge(
                            wdf[['wbs_id', 'wbs_name']].drop_duplicates('wbs_id'),
                            on='wbs_id', how='left',
                        )
                        break

            out['critical_by_wbs'] = wbs_g.nlargest(20, 'critical_count').to_dict('records')

        # ---- Constrained critical ----
        if 'cstr_type' in critical.columns:
            constr = critical[
                _col(critical, 'cstr_type').notna() & (_col(critical, 'cstr_type') != '')
            ]
            out['constrained_critical'] = _safe_records(
                constr,
                ['task_id', 'task_code', 'task_name',
                 'cstr_type', 'cstr_date', '_float_days'],
            )

        # ---- Critical activities with lags on incoming ties ----
        if 'taskpred' in source['df'] and 'task_id' in critical.columns:
            pred = source['df']['taskpred'].copy()
            pred['_lag_hrs'] = _num_col(pred, 'lag_hr_cnt')
            pred['_lag_days'] = pred['_lag_hrs'] / hpd
            crit_ids = set(critical['task_id'].tolist())
            lag_ties = pred[
                pred['task_id'].isin(crit_ids) & (pred['_lag_hrs'] != 0)
            ].copy()
            out['critical_with_lags'] = _safe_records(
                lag_ties.head(30),
                ['task_id', 'pred_task_id', 'pred_type', '_lag_days'],
            )

        # ---- Critical path changes vs baseline ----
        baseline = self.data_store.get_baseline()
        if baseline and 'tasks' in baseline.get('df', {}):
            bl_tasks = baseline['df']['tasks'].copy()
            bl_tasks['_bl_float'] = _num_col(bl_tasks, 'total_float_hr_cnt')
            bl_crit_ids = set(
                bl_tasks.loc[bl_tasks['_bl_float'] <= 0, 'task_id'].dropna().tolist()
            ) if 'task_id' in bl_tasks.columns else set()
            cu_crit_ids = set(
                critical['task_id'].dropna().tolist()
            ) if 'task_id' in critical.columns else set()

            newly = cu_crit_ids - bl_crit_ids
            left = bl_crit_ids - cu_crit_ids
            changes = []

            if 'task_id' in work.columns:
                for tid, label in (
                    list((tid, 'became_critical') for tid in list(newly)[:20])
                    + list((tid, 'left_critical') for tid in list(left)[:20])
                ):
                    row = work[work['task_id'] == tid]
                    if not row.empty:
                        r = row.iloc[0]
                        changes.append({
                            'task_id': tid,
                            'task_code': str(r.get('task_code', '')),
                            'task_name': str(r.get('task_name', '')),
                            'change': label,
                            'current_float_days': round(float(r.get('_float_days', 0)), 2),
                        })

            out['critical_path_changes'] = changes

        self._cache[key] = out
        return out

    # ------------------------------------------------------------------
    # 3. Resource Insights
    # ------------------------------------------------------------------

    def get_resource_insights(self) -> Dict:
        """
        Compute resource and cost insights:
        - Cost summary: budget / actual / remaining / EAC / CPI
        - Unresourced work activities (with duration for prioritisation)
        - Top resources by budgeted cost with utilisation %
        - Over-budget activities (actual > budget)
        - Cost breakdown by WBS
        - Resource type breakdown (labour / material / equipment)
        """
        key = 'resources'
        if key in self._cache:
            return self._cache[key]

        hpd = self.data_store.hours_per_day
        out: Dict[str, Any] = {
            'cost_summary': {},
            'unresourced_count': 0,
            'unresourced_activities': [],
            'top_resources_by_cost': [],
            'overbudget_activities': [],
            'resource_type_breakdown': {},
            'cost_by_wbs': [],
        }

        source = self.data_store.get_latest()
        if not source:
            self._cache[key] = out
            return out

        tasks_df = source['df'].get('tasks')
        rsrc_assign = source['df'].get('taskrsrc')   # assignments
        rsrc_def = source['df'].get('rsrc')           # resource definitions

        if tasks_df is None:
            self._cache[key] = out
            return out

        # Work task mask
        if 'task_type' in tasks_df.columns:
            work_tasks = tasks_df[
                ~tasks_df['task_type'].isin({'TT_LOE', 'TT_Mile', 'TT_FinMile'})
            ].copy()
        else:
            work_tasks = tasks_df.copy()

        # ---- Unresourced activities ----
        if rsrc_assign is not None and 'task_id' in rsrc_assign.columns and 'task_id' in work_tasks.columns:
            resourced_ids = set(_col(rsrc_assign, 'task_id').dropna().tolist())
            unres = work_tasks[~work_tasks['task_id'].isin(resourced_ids)].copy()
            out['unresourced_count'] = len(unres)
            if not unres.empty:
                unres['_dur_hrs'] = _num_col(unres, 'target_drtn_hr_cnt')
                unres['_dur_days'] = unres['_dur_hrs'] / hpd
                unres['_float_hrs'] = _num_col(unres, 'total_float_hr_cnt')
                out['unresourced_activities'] = _safe_records(
                    unres.nlargest(20, '_dur_hrs'),
                    ['task_id', 'task_code', 'task_name', 'status_code',
                     '_dur_days', '_float_hrs', 'target_start_date'],
                )

        if rsrc_assign is None:
            self._cache[key] = out
            return out

        ra = rsrc_assign.copy()
        ra['_budget'] = _num_col(ra, 'target_cost')
        ra['_actual'] = _num_col(ra, 'act_reg_cost')
        ra['_remain'] = _num_col(ra, 'remain_cost')
        ra['_bqty'] = _num_col(ra, 'target_qty')
        ra['_aqty'] = _num_col(ra, 'act_reg_qty')
        ra['_rqty'] = _num_col(ra, 'remain_qty')

        # ---- Cost summary ----
        b = float(ra['_budget'].sum())
        a = float(ra['_actual'].sum())
        r = float(ra['_remain'].sum())
        out['cost_summary'] = {
            'total_budgeted_cost': round(b, 2),
            'total_actual_cost': round(a, 2),
            'total_remaining_cost': round(r, 2),
            'estimate_at_completion': round(a + r, 2),
            'cost_variance': round(a - b, 2),
            'cost_performance_index': round(b / a, 3) if a > 0 else None,
            'budget_spent_pct': round(a / b * 100, 1) if b > 0 else None,
        }

        # ---- Over-budget activities ----
        if 'task_id' in ra.columns:
            task_cost = ra.groupby('task_id', as_index=False).agg(
                _budget=('_budget', 'sum'),
                _actual=('_actual', 'sum'),
            )
            task_cost['_variance'] = task_cost['_actual'] - task_cost['_budget']
            over = task_cost[
                (task_cost['_variance'] > 0) & (task_cost['_budget'] > 0)
            ].copy()

            if not over.empty and 'task_id' in tasks_df.columns:
                name_cols = ['task_id'] + [
                    c for c in ('task_name', 'task_code', 'status_code')
                    if c in tasks_df.columns
                ]
                over = over.merge(tasks_df[name_cols], on='task_id', how='left')

            out['overbudget_activities'] = over.nlargest(20, '_variance').to_dict('records')

        # ---- Top resources by budgeted cost ----
        if 'rsrc_id' in ra.columns:
            rg = ra.groupby('rsrc_id', as_index=False).agg(
                total_budget=('_budget', 'sum'),
                total_actual=('_actual', 'sum'),
                total_remain=('_remain', 'sum'),
                assignments=('task_id', 'count'),
                budget_qty=('_bqty', 'sum'),
                actual_qty=('_aqty', 'sum'),
            )
            rg['utilization_pct'] = (
                rg['actual_qty'] / rg['budget_qty'].replace(0, float('nan')) * 100
            ).fillna(0).round(1)
            rg['cost_variance'] = (rg['total_actual'] - rg['total_budget']).round(2)

            if rsrc_def is not None and 'rsrc_id' in rsrc_def.columns:
                keep = ['rsrc_id'] + [
                    c for c in ('rsrc_name', 'rsrc_short_name', 'rsrc_type')
                    if c in rsrc_def.columns
                ]
                rg = rg.merge(rsrc_def[keep].drop_duplicates('rsrc_id'), on='rsrc_id', how='left')

                # Resource type breakdown
                if 'rsrc_type' in rsrc_def.columns:
                    type_cost = rg.copy()
                    type_cost['rsrc_type'] = type_cost.get('rsrc_type', 'Unknown')
                    breakdown = type_cost.groupby('rsrc_type', as_index=False).agg(
                        budget=('total_budget', 'sum'),
                        actual=('total_actual', 'sum'),
                        count=('rsrc_id', 'count'),
                    )
                    out['resource_type_breakdown'] = breakdown.to_dict('records')

            out['top_resources_by_cost'] = rg.nlargest(20, 'total_budget').to_dict('records')

        # ---- Cost by WBS ----
        if 'task_id' in ra.columns and 'task_id' in tasks_df.columns and 'wbs_id' in tasks_df.columns:
            task_wbs = tasks_df[['task_id', 'wbs_id']].drop_duplicates('task_id')
            ra_wbs = ra.merge(task_wbs, on='task_id', how='left')

            if 'wbs_id' in ra_wbs.columns:
                wbs_cost = ra_wbs.groupby('wbs_id', as_index=False).agg(
                    budget=('_budget', 'sum'),
                    actual=('_actual', 'sum'),
                    remain=('_remain', 'sum'),
                )
                wbs_cost['variance'] = (wbs_cost['actual'] - wbs_cost['budget']).round(2)

                for wbs_key in ('projwbs', 'wbs'):
                    if wbs_key in source['df']:
                        wdf = source['df'][wbs_key]
                        if 'wbs_id' in wdf.columns and 'wbs_name' in wdf.columns:
                            wbs_cost = wbs_cost.merge(
                                wdf[['wbs_id', 'wbs_name']].drop_duplicates('wbs_id'),
                                on='wbs_id', how='left',
                            )
                            break

                out['cost_by_wbs'] = wbs_cost.nlargest(20, 'budget').to_dict('records')

        self._cache[key] = out
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str: str):
        """Parse a date string; return None if invalid."""
        if not date_str:
            return None
        dt = pd.to_datetime(date_str, errors='coerce')
        return dt if pd.notna(dt) else None


# ---------------------------------------------------------------------------
# XERQueryExecutor
# ---------------------------------------------------------------------------

class XERQueryExecutor:
    """Executes Python code safely against XER data."""

    def __init__(self, data_store: XERDataStore, insights: 'InsightsDataLayer' = None):
        self.data_store = data_store
        self._insights_ref: Dict = {}
        self._insights_layer = insights  # set after InsightsDataLayer is created

    def set_insights(self, insights_layer: 'InsightsDataLayer'):
        self._insights_layer = insights_layer

    def _get_insights_snapshot(self) -> Dict:
        """Return a lazily-evaluated snapshot of insights for code execution."""
        if self._insights_layer is None:
            return {}
        try:
            return self._insights_layer.get_all_insights()
        except Exception:
            return {}

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute generated code and return results."""
        try:
            latest_data = self.data_store.get_latest()
            baseline_data = self.data_store.get_baseline()

            context = {
                'pd': pd,
                'datetime': datetime,
                'json': json,
                'baseline': baseline_data,
                'updates': self.data_store.updates,
                'latest': latest_data,
                'get_latest': self.data_store.get_latest,
                'get_baseline': self.data_store.get_baseline,
                'get_update_by_date': self.data_store.get_update_by_date,
                'get_update_by_month': self.data_store.get_update_by_month,
                'hours_per_day': self.data_store.hours_per_day,
                'insights': self._get_insights_snapshot(),   # pre-computed insight dict
                'result': None,
            }

            exec(code, context)
            result = context.get('result')

            if isinstance(result, pd.DataFrame):
                result = result.head(50).to_dict('records')

            return {'success': True, 'result': result}

        except Exception as exc:
            return {'success': False, 'error': str(exc)}


# ---------------------------------------------------------------------------
# XERAnalyzer
# ---------------------------------------------------------------------------

class XERAnalyzer:
    """Main analyzer with comprehensive LLM support."""

    def __init__(self):
        self.data_store = XERDataStore()
        self.executor = XERQueryExecutor(self.data_store)
        self.insights = InsightsDataLayer(self.data_store)
        self.executor.set_insights(self.insights)

    def load_baseline(self, data: Dict, name: str = None, data_date: str = None):
        if name is None:
            name = data.get('project', {}).get('project_name', 'Baseline')
        if data_date is None:
            raw_date = data.get('project', {}).get('data_date', '')
            data_date = str(raw_date)[:10] if raw_date else ''
        self.data_store.load_baseline(data, name, data_date)
        self.insights.invalidate()

    def add_update(self, data: Dict, name: str = None, data_date: str = None):
        if name is None:
            name = data.get('project', {}).get('project_name', 'Update')
        if data_date is None:
            raw_date = data.get('project', {}).get('data_date', '')
            data_date = str(raw_date)[:10] if raw_date else ''
        self.data_store.add_update(data, name, data_date)
        self.insights.invalidate()

    def remove_update(self, index: int):
        self.data_store.remove_update(index)
        self.insights.invalidate()

    def get_basic_stats(self) -> Dict:
        """Get pre-computed statistics."""
        return self.data_store.compute_basic_stats()

    # Insight shortcuts
    def get_all_insights(self) -> Dict:
        return self.insights.get_all_insights()

    def get_delay_insights(self) -> Dict:
        return self.insights.get_delay_insights()

    def get_critical_path_insights(self) -> Dict:
        return self.insights.get_critical_path_insights()

    def get_resource_insights(self) -> Dict:
        return self.insights.get_resource_insights()

    def execute_code(self, code: str) -> Dict:
        return self.executor.execute(code)

    # ------------------------------------------------------------------
    # LLM prompt builders
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return """You are an expert Primavera P6 Schedule Analyst AI Assistant. You analyze construction project schedules from XER files and provide professional, insightful analysis.

YOUR CAPABILITIES:
1. Analyze schedule quality metrics
2. Compare baseline vs update files
3. Identify schedule issues and risks
4. Provide actionable recommendations
5. Answer any question about the schedule data

RESPONSE STYLE:
- Be direct, professional, and precise
- Include specific numbers and percentages
- Highlight key findings and concerns
- Provide recommendations when appropriate
- Use bullet points for clarity
- Format tables for comparisons

ANALYSIS APPROACH:
- Always consider schedule best practices
- Flag potential issues (negative float, open-ended activities, hard constraints)
- Interpret data in context of construction project management
- Suggest improvements when you see problems"""

    def get_code_generation_prompt(self, user_query: str, basic_stats: Dict) -> str:
        """Generate prompt for code generation."""
        source = self.data_store.get_latest()

        # Collect column names from actual DataFrames
        columns_info = ""
        if source and 'df' in source:
            for table_name, df in list(source['df'].items())[:10]:
                cols = list(df.columns)[:20]
                columns_info += f"\n{table_name.upper()}: {', '.join(cols)}"

        return f"""Generate Python code to answer this question about a Primavera P6 schedule.

USER QUESTION: {user_query}

CURRENT PROJECT STATISTICS (always available as fallback):
{json.dumps(basic_stats, indent=2, default=str)}

AVAILABLE DATA TABLES AND COLUMNS:{columns_info}

AVAILABLE VARIABLES IN CODE:
- latest: dict with latest schedule data; access DataFrames via latest['df']['table_name']
- baseline: dict with baseline data; access via baseline['df']['table_name']
- updates: list of update dicts (sorted chronologically)
- get_update_by_month('feb'): returns the update file for February
- get_update_by_month('mar', '2024'): filtered by year
- insights: pre-computed insights dict with keys 'delays', 'critical_path', 'resources'
  - insights['delays']        → project_end_slippage_days, slipped_activities_count,
                                 most_slipped_activities, delayed_not_started,
                                 delayed_in_progress, gained_negative_float_count, float_trend
  - insights['critical_path'] → critical_activities (full list with float/duration),
                                 near_critical_activities, critical_by_wbs,
                                 critical_path_changes, constrained_critical, summary
  - insights['resources']     → cost_summary (budget/actual/remain/CPI),
                                 unresourced_activities, top_resources_by_cost,
                                 overbudget_activities, cost_by_wbs
- pd: pandas library
- hours_per_day: 10 (for duration conversion)

KEY FIELDS (all string values — ALWAYS convert to numeric before math):
Tasks DataFrame:
  task_id, task_name, task_code, task_type, status_code,
  target_start_date, target_end_date, act_start_date, act_end_date,
  target_drtn_hr_cnt  (duration in HOURS - divide by hours_per_day for days),
  total_float_hr_cnt  (total float in HOURS),
  free_float_hr_cnt   (free float in HOURS),
  phys_complete_pct   (physical % complete),
  cstr_type, cstr_date, cstr_type2, cstr_date2,
  wbs_id, clndr_id, rsrc_id, driving_path_flag

Taskpred DataFrame (taskpred):
  task_id, pred_task_id,
  pred_type  (PR_FS / PR_SS / PR_FF / PR_SF),
  lag_hr_cnt (lag in hours — can be negative)

Taskrsrc DataFrame (taskrsrc):
  task_id, rsrc_id, role_id,
  target_qty, act_reg_qty, remain_qty,
  target_cost, act_reg_cost, remain_cost

Projwbs DataFrame (projwbs):
  wbs_id, wbs_name, parent_wbs_id, proj_id

Calendar DataFrame (calendar):
  clndr_id, clndr_name, day_hr_cnt, week_hr_cnt

TASK TYPES:
  TT_Task    = regular task
  TT_Mile    = start milestone
  TT_FinMile = finish milestone
  TT_LOE     = level of effort (exclude from most analyses)

STATUS CODES:
  TK_NotStart = not started
  TK_Active   = in progress
  TK_Complete = complete

CONSTRAINT TYPES:
  CS_MSOA (as late as possible), CS_MEOA (as soon as possible),
  CS_MEOB (must finish before), CS_MSO (must start on),
  CS_MEO (must end on), CS_MFOB (mandatory finish)

CODING RULES:
1. Always call .copy() when modifying DataFrames
2. ALWAYS convert numeric fields: pd.to_numeric(df['col'], errors='coerce').fillna(0)
3. Duration is in HOURS → divide by hours_per_day for days
4. For open-ended: tasks NOT in taskpred['pred_task_id'] (no successor)
5. For dangling: tasks NOT in taskpred['task_id'] (no predecessor)
6. Exclude TT_LOE and milestones from quality analyses unless the question is specifically about them
7. Always store final result in a variable named 'result'
8. result must be JSON-serializable (dict, list, number, or string)
9. Limit lists to 50 items max
10. Check if columns exist before using them: if 'col' in df.columns

COMMON CODE PATTERNS:

# Convert columns
tasks = latest['df']['tasks'].copy()
tasks['dur_hrs']     = pd.to_numeric(tasks['target_drtn_hr_cnt'], errors='coerce').fillna(0)
tasks['dur_days']    = tasks['dur_hrs'] / hours_per_day
tasks['float_hrs']   = pd.to_numeric(tasks['total_float_hr_cnt'], errors='coerce').fillna(0)
tasks['complete_pct']= pd.to_numeric(tasks['phys_complete_pct'], errors='coerce').fillna(0)

# Work tasks only
work = tasks[~tasks['task_type'].isin(['TT_LOE', 'TT_Mile', 'TT_FinMile'])].copy()

# Predecessor frame
preds = latest['df']['taskpred'].copy()
preds['lag_hrs'] = pd.to_numeric(preds['lag_hr_cnt'], errors='coerce').fillna(0)

# Open-ended (no successor)
has_succ = set(preds['pred_task_id'].tolist())
open_ended = work[~work['task_id'].isin(has_succ)]

# Negative lags
neg_lags = preds[preds['lag_hrs'] < 0]

# Long duration
long_dur = work[work['dur_days'] > 20]

# Comparing two updates
feb = get_update_by_month('feb')
mar = get_update_by_month('mar')
if feb and mar:
    feb_tasks = feb['df']['tasks'].copy()
    mar_tasks = mar['df']['tasks'].copy()
    # merge on task_id and compare...

Return ONLY valid Python code that sets result = ... at the end. No markdown fences."""

    def get_response_prompt(
        self,
        user_query: str,
        basic_stats: Dict,
        code_result: Any,
        code_success: bool,
        code_error: str = None,
    ) -> str:
        """Generate prompt for final response, injecting relevant insights."""

        context = f"""USER QUESTION: {user_query}

PROJECT OVERVIEW:
- Project: {basic_stats.get('data_source', 'N/A')} (Data Date: {basic_stats.get('data_date', 'N/A')})
- Total Activities: {basic_stats.get('total_activities', 'N/A')}
- Project Period: {basic_stats.get('project_start', 'N/A')} to {basic_stats.get('project_finish', 'N/A')}
- Milestones: {basic_stats.get('milestones', 'N/A')}
- LOE Activities: {basic_stats.get('loe_activities', 'N/A')}

STATUS BREAKDOWN:
- Completed: {basic_stats.get('completed', 'N/A')}
- In Progress: {basic_stats.get('in_progress', 'N/A')}
- Not Started: {basic_stats.get('not_started', 'N/A')}

SCHEDULE HEALTH METRICS:
- Critical Activities: {basic_stats.get('critical_count', 'N/A')} ({basic_stats.get('critical_pct', 'N/A')}%)
- Near-Critical: {basic_stats.get('near_critical_count', 'N/A')}
- Negative Float: {basic_stats.get('negative_float_count', 'N/A')}
- Open-Ended: {basic_stats.get('open_ended_count', 'N/A')}
- Dangling Activities: {basic_stats.get('dangling_count', 'N/A')}
- Long Duration (>20d): {basic_stats.get('long_duration_count', 'N/A')}
- Constrained Activities: {basic_stats.get('constrained_activities', 'N/A')}

OVERDUE:
- Not Started Past Target Start: {basic_stats.get('overdue_not_started', 'N/A')}
- In Progress Past Target End: {basic_stats.get('overdue_in_progress', 'N/A')}

RELATIONSHIPS:
- Total: {basic_stats.get('total_relationships', 'N/A')}
- With Lag: {basic_stats.get('relationships_with_lag', 'N/A')}
- Negative Lags: {basic_stats.get('negative_lags', 'N/A')}

RESOURCE LOADING:
- Tasks with Resources: {basic_stats.get('tasks_with_resources', 'N/A')} ({basic_stats.get('resource_loaded_pct', 'N/A')}%)

FILES LOADED:
- Baseline: {basic_stats.get('baseline_name', 'N/A')} ({basic_stats.get('baseline_date', 'N/A')})
- Updates: {basic_stats.get('update_count', 0)} file(s) — {[u['name'] for u in basic_stats.get('updates', [])]}
"""

        # Selectively inject pre-computed insights based on query topic.
        # This keeps token count reasonable while giving the LLM richer data.
        q = user_query.lower()

        _delay_kw = {'delay', 'slip', 'late', 'overdue', 'behind', 'slippage',
                     'float', 'behind schedule', 'trend', 'erosion'}
        _cp_kw = {'critical', 'near-critical', 'near critical', 'critical path',
                  'longest path', 'constraint', 'float', 'driving'}
        _rsrc_kw = {'resource', 'cost', 'budget', 'spend', 'loaded', 'unloaded',
                    'over budget', 'labour', 'labor', 'utiliz', 'cpi', 'eac'}

        if any(kw in q for kw in _delay_kw):
            try:
                d = self.insights.get_delay_insights()
                context += f"\nDELAY INSIGHTS:\n{json.dumps(d, indent=2, default=str)}\n"
            except Exception:
                pass

        if any(kw in q for kw in _cp_kw):
            try:
                cp = self.insights.get_critical_path_insights()
                context += f"\nCRITICAL PATH INSIGHTS:\n{json.dumps(cp, indent=2, default=str)}\n"
            except Exception:
                pass

        if any(kw in q for kw in _rsrc_kw):
            try:
                rs = self.insights.get_resource_insights()
                context += f"\nRESOURCE INSIGHTS:\n{json.dumps(rs, indent=2, default=str)}\n"
            except Exception:
                pass

        if code_success and code_result is not None:
            context += f"""
SPECIFIC ANALYSIS RESULTS:
{json.dumps(code_result, indent=2, default=str)}
"""
        elif code_error:
            context += f"""
Note: Detailed code analysis failed ({code_error}). Respond based on pre-computed statistics and insights above.
"""

        context += """
INSTRUCTIONS:
1. Answer the user's question directly and professionally
2. Include specific numbers from the analysis results or statistics
3. Highlight any concerns or issues found
4. Provide actionable recommendations when problems are identified
5. Be concise but thorough
6. Use bullet points and formatting for clarity
7. If the question asks for a list, provide the items found
8. If comparing files, show clear before/after differences with delta values"""

        return context
