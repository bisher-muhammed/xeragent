"""
XER Schedule Analyzer
Orchestrates data store, insights engine, query executor, and LLM prompts.
"""

import json
import threading
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

EXEC_TIMEOUT_SECONDS = 30


# =============================================================================
# XERInsightsEngine  —  pre-computed insights (delays, critical path, resources)
# =============================================================================

class XERInsightsEngine:
    """
    Computes insight sets from structured task data.

    Runs a full CPM forward/backward pass (not P6's pre-computed flags) so
    float and criticality are independently verified and can support what-if
    queries. Delay propagation uses BFS through the successor graph, bleeding
    slip through downstream float buffers.
    """

    def compute(self, tasks_list: List[Dict], data_date: str) -> Dict:
        """
        Returns:
          delays            — tasks with start or finish slip vs plan
          critical_path     — ordered critical work tasks with driving predecessor
          resources         — aggregated cost/qty per resource with variance flags
          delay_propagation — downstream cascade analysis for each delayed task
          cpm               — per-task CPM results {task_id: {float, is_critical, ...}}
        """
        try:
            dd = datetime.strptime(data_date[:10], '%Y-%m-%d') if data_date else None
        except ValueError:
            dd = None

        # Build indexes once; shared across all methods
        task_map: Dict[str, Dict] = {t['task_id']: t for t in tasks_list}
        succ_map: Dict[str, List[Dict]] = defaultdict(list)
        pred_map: Dict[str, List[Dict]] = defaultdict(list)
        for t in tasks_list:
            tid = t['task_id']
            for s in t['relationships']['successors']:
                succ_map[tid].append(s)
            for p in t['relationships']['predecessors']:
                pred_map[tid].append(p)

        delays         = self._delays(tasks_list, dd)
        cpm            = self._run_cpm(tasks_list, task_map, succ_map, pred_map)
        critical_path  = self._critical_path(tasks_list, task_map, succ_map, pred_map, cpm)
        resources      = self._resources(tasks_list)
        delay_prop     = self._delay_propagation(tasks_list, delays, task_map, succ_map)

        return {
            'delays':            delays,
            'critical_path':     critical_path,
            'resources':         resources,
            'delay_propagation': delay_prop,
            'cpm':               cpm,
        }

    # ── Real CPM ──────────────────────────────────────────────────────────────

    def _run_cpm(
        self,
        tasks_list: List[Dict],
        task_map:   Dict,
        succ_map:   Dict,
        pred_map:   Dict,
    ) -> Dict:
        """
        Independent forward/backward pass CPM using ordinal-day arithmetic.

        Relationship types handled: FS, SS, FF, SF (with lags).
        Completed tasks are pinned to their actual dates.
        In-progress tasks use actual start + remaining duration.
        Not-started tasks are scheduled from their latest predecessor.

        Returns {task_id: {total_float_days, is_critical, has_negative_float}}
        """
        all_ids = set(task_map.keys())

        # ── Helpers ───────────────────────────────────────────────────────────

        def remaining(tid: str) -> float:
            t = task_map[tid]
            return 0.0 if t['status'] == 'TK_Complete' else max(
                0.0, t['duration'].get('remaining_days', 0.0)
            )

        def fixed_es(tid: str) -> Optional[int]:
            """Return actual_start ordinal for started tasks; None otherwise."""
            t = task_map[tid]
            if t['status'] in ('TK_Complete', 'TK_Active'):
                d = _parse_date(t['dates'].get('actual_start'))
                if d:
                    return d.toordinal()
            return None

        def fixed_ef(tid: str) -> Optional[int]:
            """Return actual_finish ordinal for complete tasks; None otherwise."""
            t = task_map[tid]
            if t['status'] == 'TK_Complete':
                d = _parse_date(t['dates'].get('actual_finish'))
                if d:
                    return d.toordinal()
            return None

        # ── Topological sort (Kahn's) ─────────────────────────────────────────

        in_deg: Dict[str, int] = {
            tid: sum(1 for p in pred_map.get(tid, []) if p['task_id'] in all_ids)
            for tid in all_ids
        }
        queue: deque = deque(tid for tid in all_ids if in_deg[tid] == 0)
        topo:  List[str] = []
        seen:  set = set()

        while queue:
            tid = queue.popleft()
            if tid in seen:
                continue
            seen.add(tid)
            topo.append(tid)
            for s in succ_map.get(tid, []):
                sid = s['task_id']
                if sid in in_deg:
                    in_deg[sid] -= 1
                    if in_deg[sid] <= 0 and sid not in seen:
                        queue.append(sid)

        # Append any tasks in cycles (skip-safe)
        for tid in all_ids:
            if tid not in seen:
                topo.append(tid)

        # Project reference start (earliest planned start across all tasks)
        p_dates = [
            _parse_date(t['dates'].get('planned_start') or t['dates'].get('early_start'))
            for t in tasks_list
        ]
        proj_start = min(d.toordinal() for d in p_dates if d) if any(p_dates) else 0

        # ── Forward pass ──────────────────────────────────────────────────────

        es_ord: Dict[str, float] = {}
        ef_ord: Dict[str, float] = {}

        for tid in topo:
            dur = remaining(tid)
            fe  = fixed_ef(tid)
            fs  = fixed_es(tid)

            # Completed: pin to actual dates
            if fe is not None:
                ef_ord[tid] = float(fe)
                es_ord[tid] = float(fs) if fs is not None else float(fe) - dur
                continue

            # In-progress: ES is fixed (actual start), EF floats
            if fs is not None:
                es_ord[tid] = float(fs)
            else:
                # Not started: drive from predecessors
                candidates: List[float] = []
                for p in pred_map.get(tid, []):
                    pid = p['task_id']
                    if pid not in ef_ord:
                        continue
                    rel = p.get('type', 'FS')
                    lag = float(p.get('lag_days', 0))

                    # Standard CPM forward constraints:
                    # FS: ES_j >= EF_i + lag
                    # SS: ES_j >= ES_i + lag
                    # FF: EF_j >= EF_i + lag  →  ES_j >= EF_i + lag - dur_j
                    # SF: EF_j >= ES_i + lag  →  ES_j >= ES_i + lag - dur_j
                    if rel == 'FS':
                        candidates.append(ef_ord[pid] + lag)
                    elif rel == 'SS':
                        candidates.append(es_ord.get(pid, ef_ord[pid]) + lag)
                    elif rel == 'FF':
                        candidates.append(ef_ord[pid] + lag - dur)
                    elif rel == 'SF':
                        candidates.append(es_ord.get(pid, ef_ord[pid]) + lag - dur)
                    else:
                        candidates.append(ef_ord[pid] + lag)

                if candidates:
                    es_ord[tid] = max(candidates)
                else:
                    t    = task_map[tid]
                    pd_d = _parse_date(
                        t['dates'].get('planned_start') or t['dates'].get('early_start')
                    )
                    es_ord[tid] = float(pd_d.toordinal()) if pd_d else float(proj_start)

            ef_ord[tid] = es_ord[tid] + dur

        if not ef_ord:
            return {}

        proj_end = max(ef_ord.values())

        # ── Backward pass ─────────────────────────────────────────────────────

        lf_ord: Dict[str, float] = {}
        ls_ord: Dict[str, float] = {}

        for tid in reversed(topo):
            dur = remaining(tid)

            # Completed: pin to actual
            if task_map[tid]['status'] == 'TK_Complete':
                lf_ord[tid] = ef_ord.get(tid, proj_end)
                ls_ord[tid] = es_ord.get(tid, lf_ord[tid] - dur)
                continue

            constraints: List[float] = []
            for s in succ_map.get(tid, []):
                sid = s['task_id']
                if sid not in ls_ord:
                    continue
                rel = s.get('type', 'FS')
                lag = float(s.get('lag_days', 0))

                # Standard CPM backward constraints (tid = predecessor, sid = successor):
                # FS: LF_pred = LS_succ - lag
                # SS: LS_pred = LS_succ - lag  →  LF_pred = LS_succ - lag + dur_pred
                # FF: LF_pred = LF_succ - lag
                # SF: LS_pred = LF_succ - lag  →  LF_pred = LF_succ - lag + dur_pred
                if rel == 'FS':
                    constraints.append(ls_ord[sid] - lag)
                elif rel == 'SS':
                    constraints.append(ls_ord[sid] - lag + dur)
                elif rel == 'FF':
                    constraints.append(lf_ord.get(sid, proj_end) - lag)
                elif rel == 'SF':
                    constraints.append(lf_ord.get(sid, proj_end) - lag + dur)
                else:
                    constraints.append(ls_ord[sid] - lag)

            lf_ord[tid] = min(constraints) if constraints else proj_end
            ls_ord[tid] = lf_ord[tid] - dur

        # ── Assemble results ──────────────────────────────────────────────────

        cpm_results: Dict[str, Dict] = {}
        for tid in all_ids:
            if tid not in ef_ord:
                continue
            ef = ef_ord.get(tid, 0.0)
            lf = lf_ord.get(tid, proj_end)
            tf = lf - ef
            cpm_results[tid] = {
                'total_float_days':   round(tf, 1),
                'is_critical':        tf <= 0,
                'has_negative_float': tf < 0,
            }

        return cpm_results

    # ── Delays ────────────────────────────────────────────────────────────────

    def _delays(self, tasks_list: List[Dict], data_date: datetime) -> List[Dict]:
        """
        Start/finish slip for every task that has begun.
        In-progress tasks use data_date as a proxy for actual finish.
        """
        rows = []
        for t in tasks_list:
            if t['task_type'] == 'TT_LOE':
                continue
            if t['status'] == 'TK_NotStart':
                continue

            dates   = t['dates']
            p_start = _parse_date(dates.get('planned_start'))
            p_fin   = _parse_date(dates.get('planned_finish'))
            a_start = _parse_date(dates.get('actual_start'))
            a_fin   = _parse_date(dates.get('actual_finish'))

            start_slip  = _day_diff(a_start, p_start)
            finish_slip = _day_diff(a_fin,   p_fin)

            if finish_slip is None and t['status'] == 'TK_Active' and data_date and p_fin:
                finish_slip = _day_diff(data_date, p_fin)

            if (start_slip or 0) <= 0 and (finish_slip or 0) <= 0:
                continue

            rows.append({
                'task_code':        t['task_code'],
                'task_name':        t['task_name'],
                'wbs_path':         t['wbs_path'],
                'status':           t['status'],
                'planned_start':    dates.get('planned_start'),
                'actual_start':     dates.get('actual_start'),
                'planned_finish':   dates.get('planned_finish'),
                'actual_finish':    dates.get('actual_finish'),
                'start_slip_days':  start_slip,
                'finish_slip_days': finish_slip,
                'is_critical':      t['float']['is_critical'],
            })

        rows.sort(key=lambda r: (r['finish_slip_days'] or 0), reverse=True)
        return rows[:50]

    # ── Delay Propagation ─────────────────────────────────────────────────────

    def _delay_propagation(
        self,
        tasks_list: List[Dict],
        delays:     List[Dict],
        task_map:   Dict,
        succ_map:   Dict,
    ) -> List[Dict]:
        """
        BFS from each delayed source task through its successors.

        At each hop, the propagated slip is reduced by the successor's
        available float and the incoming relationship lag. If net impact > 0
        the successor is affected and becomes a new frontier node.

        Result is sorted by source slip (worst first), capped at top-20 sources
        and top-10 impacted tasks each.
        """
        code_to_id = {t['task_code']: t['task_id'] for t in tasks_list}
        propagation: List[Dict] = []

        for delay in delays[:20]:
            slip = float(delay.get('finish_slip_days') or delay.get('start_slip_days') or 0)
            if slip <= 0:
                continue

            source_id = code_to_id.get(delay['task_code'])
            if not source_id:
                continue

            visited:  set   = {source_id}
            frontier: deque = deque([(source_id, slip)])
            impacted: List[Dict] = []

            while frontier:
                current_id, propagated_slip = frontier.popleft()

                for s in succ_map.get(current_id, []):
                    sid = s['task_id']
                    if sid in visited:
                        continue
                    visited.add(sid)

                    st = task_map.get(sid)
                    if not st or st['status'] == 'TK_Complete':
                        continue  # Already done — not at risk

                    available_float = st['float'].get('total_float_days', 0.0)
                    rel_lag         = float(s.get('lag_days', 0))
                    net_impact      = propagated_slip - available_float - rel_lag

                    if net_impact > 0:
                        impacted.append({
                            'task_code':            st['task_code'],
                            'task_name':            st['task_name'],
                            'status':               st['status'],
                            'available_float':      round(available_float, 1),
                            'estimated_impact_days': round(net_impact, 1),
                            'is_critical':          st['float']['is_critical'],
                        })
                        frontier.append((sid, net_impact))

            if impacted:
                impacted.sort(key=lambda x: -x['estimated_impact_days'])
                propagation.append({
                    'source_task_code':    delay['task_code'],
                    'source_task_name':    delay['task_name'],
                    'slip_days':           slip,
                    'impacted_task_count': len(impacted),
                    'impacted_tasks':      impacted[:10],
                })

        propagation.sort(key=lambda x: -x['slip_days'])
        return propagation

    # ── Critical path ─────────────────────────────────────────────────────────

    def _critical_path(
        self,
        tasks_list: List[Dict],
        task_map:   Dict,
        succ_map:   Dict,
        pred_map:   Dict,
        cpm:        Dict,
    ) -> List[Dict]:
        """
        Incomplete critical work tasks ordered by planned_start.

        Uses CPM-computed float where available; falls back to P6's flag.
        Each task includes its driving predecessor (the predecessor with the
        tightest float, i.e., the one actually driving this task's late start).
        """
        critical: List[Dict] = []

        for t in tasks_list:
            if t['task_type'] != 'TT_Task':
                continue
            if t['status'] == 'TK_Complete':
                continue

            tid      = t['task_id']
            cpm_data = cpm.get(tid, {})

            # Prefer independently computed criticality
            is_crit  = cpm_data.get('is_critical',        t['float']['is_critical'])
            has_neg  = cpm_data.get('has_negative_float', t['float']['has_negative_float'])
            cpm_float = cpm_data.get('total_float_days',  t['float']['total_float_days'])

            if not is_crit:
                continue

            driving_pred = self._find_driving_predecessor(tid, pred_map, task_map, cpm)

            critical.append({
                'task_code':          t['task_code'],
                'task_name':          t['task_name'],
                'wbs_path':           t['wbs_path'],
                'status':             t['status'],
                'planned_start':      t['dates'].get('planned_start'),
                'planned_finish':     t['dates'].get('planned_finish'),
                'duration_days':      t['duration']['planned_days'],
                'remaining_days':     t['duration']['remaining_days'],
                'float_days':         cpm_float,
                'has_negative_float': has_neg,
                'predecessor_count':  t['relationships']['predecessor_count'],
                'successor_count':    t['relationships']['successor_count'],
                'driving_predecessor': driving_pred,
            })

        critical.sort(key=lambda t: t['planned_start'] or '')
        return critical

    def _find_driving_predecessor(
        self,
        tid:      str,
        pred_map: Dict,
        task_map: Dict,
        cpm:      Dict,
    ) -> Optional[Dict]:
        """
        The driving predecessor is the one with the least float —
        it has no schedule cushion to absorb further delay, so it is
        the constraint that is actually driving this task's late start.
        """
        best_float = float('inf')
        driving: Optional[Dict] = None

        for p in pred_map.get(tid, []):
            pid = p['task_id']
            pt  = task_map.get(pid)
            if not pt:
                continue
            pred_float = cpm.get(pid, {}).get(
                'total_float_days', pt['float'].get('total_float_days', 0.0)
            )
            if pred_float < best_float:
                best_float = pred_float
                driving    = {
                    'task_code':  pt['task_code'],
                    'task_name':  pt['task_name'],
                    'float_days': round(pred_float, 1),
                    'rel_type':   p.get('type', 'FS'),
                    'lag_days':   p.get('lag_days', 0),
                }

        return driving

    # ── Resources ─────────────────────────────────────────────────────────────

    def _resources(self, tasks_list: List[Dict]) -> List[Dict]:
        """
        Aggregate cost and qty per resource with variance analysis:
          cost_variance  = planned - actual - remaining  (negative = over budget)
          spend_pct      = actual / planned × 100
          is_overloaded  = actual_qty > planned_qty by more than 10%
          is_over_budget = cost_variance < 0
        Sorted by planned_cost descending.
        """
        agg: Dict[str, Dict] = {}

        for t in tasks_list:
            for r in t.get('resources', []):
                name = r.get('resource_name') or r.get('resource_id', 'Unknown')
                if name not in agg:
                    agg[name] = {
                        'resource_name':  name,
                        'resource_type':  r.get('resource_type', ''),
                        'task_count':     0,
                        'planned_cost':   0.0,
                        'actual_cost':    0.0,
                        'remaining_cost': 0.0,
                        'planned_qty':    0.0,
                        'actual_qty':     0.0,
                        'remaining_qty':  0.0,
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

        for r in result:
            for k in ('planned_cost', 'actual_cost', 'remaining_cost',
                      'planned_qty',  'actual_qty',  'remaining_qty'):
                r[k] = round(r[k], 2)

            # Variance: positive means we have budget left; negative = over
            r['cost_variance']  = round(
                r['planned_cost'] - r['actual_cost'] - r['remaining_cost'], 2
            )
            r['spend_pct']      = round(
                r['actual_cost'] / r['planned_cost'] * 100, 1
            ) if r['planned_cost'] else 0.0
            r['is_overloaded']  = (
                r['planned_qty'] > 0 and r['actual_qty'] > r['planned_qty'] * 1.1
            )
            r['is_over_budget'] = r['cost_variance'] < 0

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
        self._cached_stats:   Optional[Dict] = None
        self._cached_insights: Optional[Dict] = None
        self._insights_engine = XERInsightsEngine()

    # ── Load / add / remove ───────────────────────────────

    def load_baseline(self, data: Dict, name: str, data_date: str):
        self.baseline          = self._build_source(data, name, data_date)
        self._cached_stats     = None
        self._cached_insights  = None

    def add_update(self, data: Dict, name: str, data_date: str):
        self.updates.append(self._build_source(data, name, data_date))
        self.updates.sort(key=lambda x: x['data_date'])
        self._cached_stats     = None
        self._cached_insights  = None

    def remove_update(self, index: int):
        if 0 <= index < len(self.updates):
            self.updates.pop(index)
            self._cached_stats     = None
            self._cached_insights  = None

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
                'early_start':    t['dates'].get('early_start'),
                'early_finish':   t['dates'].get('early_finish'),
                'late_start':     t['dates'].get('late_start'),
                'late_finish':    t['dates'].get('late_finish'),
                'predecessor_count': t['relationships']['predecessor_count'],
                'successor_count':   t['relationships']['successor_count'],
                'is_open_ended':     t['relationships']['is_open_ended'],
                'is_dangling':       t['relationships']['is_dangling'],
                'has_resources':  t.get('has_resources', False),
                'has_constraint': t.get('constraints') is not None,
                'constraint_type': t['constraints']['type'] if t.get('constraints') else '',
            })

        tasks_df = pd.DataFrame(rows)
        for col in ['planned_start', 'planned_finish', 'actual_start', 'actual_finish',
                    'early_start', 'early_finish', 'late_start', 'late_finish']:
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
            # Full relationship objects (not just IDs) for CPM/path traversal
            'successors':   _idx(latest).get('successors', {}),
            'predecessors': _idx(latest).get('predecessors', {}),

            # Pre-computed insights — use these for delay/critical/resource/propagation questions
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
  insights['delays']            → list of delayed tasks (start_slip_days, finish_slip_days)
  insights['critical_path']     → ordered list of critical work tasks (CPM-computed float)
                                   each task includes: driving_predecessor dict
  insights['resources']         → list of resources with cost_variance, spend_pct,
                                   is_overloaded, is_over_budget flags
  insights['delay_propagation'] → BFS cascade analysis per delayed task:
                                   source_task_code, slip_days, impacted_task_count,
                                   impacted_tasks[{{task_code, estimated_impact_days, available_float}}]
  insights['cpm']               → dict keyed by task_id:
                                   {{total_float_days, is_critical, has_negative_float}}
                                   This is an INDEPENDENT CPM calculation, not P6's flag.

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
task['dates']['early_start'], task['dates']['early_finish'],
task['dates']['late_start'],  task['dates']['late_finish']

── CORRECT PATTERNS ──────────────────────────────────────

# Use insights directly for delay/critical/resource questions
result = insights['delays'][:20]

# Cascade impact from the most delayed task
result = insights['delay_propagation'][:5]

# Critical path sequence with driving predecessors
result = insights['critical_path']

# Resources over budget or overloaded
result = [r for r in insights['resources'] if r['is_over_budget'] or r['is_overloaded']]

# CPM-computed float for a specific task
task_id = next((t['task_id'] for t in tasks_list if t['task_code'] == 'A1000'), None)
if task_id:
    result = insights['cpm'].get(task_id)

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

        delays        = insights.get('delays', [])
        delayed_count = len(delays)
        worst_delays  = delays[:3]
        worst_str     = ', '.join(
            d['task_code'] + ' (' + str(d['finish_slip_days']) + 'd slip)'
            for d in worst_delays
        )
        delay_line = str(delayed_count) + (' — worst: ' + worst_str if worst_str else '')

        resources          = insights.get('resources', [])
        total_planned_cost = sum(r.get('planned_cost', 0) for r in resources)
        over_budget_count  = sum(1 for r in resources if r.get('is_over_budget'))
        overloaded_count   = sum(1 for r in resources if r.get('is_overloaded'))

        cp           = insights.get('critical_path', [])
        cp_remaining = sum(t.get('remaining_days', 0) for t in cp)
        neg_float_cp = sum(1 for t in cp if t.get('has_negative_float'))

        delay_prop   = insights.get('delay_propagation', [])
        cascade_str  = ''
        if delay_prop:
            top = delay_prop[0]
            cascade_str = (
                f" | Top cascade: {top['source_task_code']} ({top['slip_days']:.0f}d slip)"
                f" impacts {top['impacted_task_count']} downstream tasks"
            )

        ctx = f"""USER QUESTION: {user_query}

PROJECT OVERVIEW:
- Project: {basic_stats.get('data_source','N/A')} | Data Date: {basic_stats.get('data_date','N/A')}
- Total Activities: {basic_stats.get('total_activities','N/A')}
- Period: {basic_stats.get('project_start','N/A')} to {basic_stats.get('project_finish','N/A')}

STATUS: Completed={basic_stats.get('completed','N/A')} | In Progress={basic_stats.get('in_progress','N/A')} | Not Started={basic_stats.get('not_started','N/A')}

SCHEDULE HEALTH:
- Critical (CPM-computed): {basic_stats.get('critical_count','N/A')} ({basic_stats.get('critical_pct','N/A')}%) | Near-Critical: {basic_stats.get('near_critical_count','N/A')}
- Negative Float: {basic_stats.get('negative_float_count','N/A')} | Open-Ended: {basic_stats.get('open_ended_count','N/A')} | Dangling: {basic_stats.get('dangling_count','N/A')}
- Long Duration >30d: {basic_stats.get('long_duration_count','N/A')} | Constrained: {basic_stats.get('constrained_activities','N/A')}

RELATIONSHIPS: Total={basic_stats.get('total_relationships','N/A')} | With Lag={basic_stats.get('relationships_with_lag','N/A')} | Negative Lags={basic_stats.get('negative_lags','N/A')}
RESOURCES: {basic_stats.get('resource_loaded_count','N/A')} tasks loaded | Total Planned Cost: {total_planned_cost:,.0f} | Over Budget: {over_budget_count} | Overloaded: {overloaded_count}
FILES: Baseline={basic_stats.get('baseline_name','N/A')} ({basic_stats.get('baseline_date','N/A')}) | Updates={basic_stats.get('update_count',0)}

INSIGHTS SUMMARY:
- Delayed tasks: {delay_line}
- Cascade risk: {len(delay_prop)} delayed tasks propagate downstream{cascade_str}
- Critical path: {len(cp)} tasks remaining | {cp_remaining:.0f}d total remaining | {neg_float_cp} with negative float
- Resources: {len(resources)} tracked | {over_budget_count} over budget | {overloaded_count} overloaded
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
4. For delayed tasks, always mention downstream cascade impact where relevant.
5. For critical path tasks, mention the driving predecessor where available.
6. Highlight concerns with clear recommendations.
7. Use markdown tables for lists of 5+ items.
8. If comparing files, show a clear before/after table."""

        return ctx