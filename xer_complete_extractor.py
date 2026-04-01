#!/usr/bin/env python3
"""
Complete XER Data Extractor  —  v2.0
Parses all 35 known Primavera P6 XER tables into structured JSON.

Output schema (self.structured):
{
  schema_version   : '2.0'
  project          : project header + scheduling options
  tasks            : list — each task has dates, duration, float,
                     relationships, resources, notes, steps, period_actuals,
                     cost_account, activity_codes (labelled), custom_fields
  wbs              : {tree, nodes, paths}
  obs              : {tree, nodes}           ← OBS hierarchy
  resources        : list — resource master with rates
  roles            : list — role master with rates
  calendars        : list — calendar header + non-work exceptions
  cost_accounts    : list — account hierarchy
  financial_periods: list — FINDATES period definitions
  project_codes    : {types, values, assignments}
  resource_categories: {types, values, assignments}
  currencies       : list
  summary          : pre-computed counts
  parse_report     : per-table status, coverage %, warnings
  tables           : raw string tables (always present for edge-case access)
}
"""

from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional
import json


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION       = '2.0'
DEFAULT_HOURS_PER_DAY = 10.0
DEFAULT_NEAR_CRITICAL_DAYS = 5.0

# Tables the extractor fully processes into typed structures.
# Everything else is kept as raw string rows in self.tables.
PROCESSED_TABLES = {
    # Core
    'PROJECT', 'PROJWBS', 'SCHEDOPTION',
    # Tasks
    'TASK', 'TASKPRED', 'TASKRSRC', 'TASKACTV', 'TASKMEMO', 'TASKFIN', 'TASKPROC',
    # Resources / roles
    'RSRC', 'RSRCRATE', 'ROLE', 'RSRCROLE', 'ROLERATE',
    # Calendars
    'CALENDAR', 'NONWORK',
    # Activity codes
    'ACTVTYPE', 'ACTVCODE',
    # UDFs
    'UDFTYPE', 'UDFVALUE',
    # Cost
    'ACCOUNT',
    # Financial
    'FINDATES', 'FINTMPL',
    # OBS
    'POBS',
    # Project codes
    'PCATTYPE', 'PCATVAL', 'PROJPCAT',
    # Resource categories
    'RCATTYPE', 'RCATVAL', 'RSRCCAT',
    # Memos
    'MEMOTYPE',
    # Currency
    'CURRTYPE',
}

# CLNDRDATA stores calendar working hours as a proprietary encoded string.
# We store it raw; parsing the internal format is out of scope.
RAW_ONLY_TABLES = {'CLNDRDATA', 'PROJCOST'}

NUMERIC_FIELDS = {
    # Task
    'target_drtn_hr_cnt', 'remain_drtn_hr_cnt', 'total_float_hr_cnt',
    'free_float_hr_cnt', 'phys_complete_pct', 'lag_hr_cnt',
    # Resources
    'target_cost', 'act_reg_cost', 'remain_cost',
    'target_qty',  'act_reg_qty',  'remain_qty',
    # Calendar
    'day_hr_cnt', 'week_hr_cnt', 'month_hr_cnt',
    # Resource rates
    'cost_per_qty', 'cost_per_qty2', 'cost_per_qty3',
    'cost_per_qty4', 'cost_per_qty5', 'max_qty_per_hr',
    # Financial
    'act_work_qty', 'act_equip_qty', 'act_reg_cost', 'act_ot_cost',
    'target_work_qty', 'target_equip_qty',
    # Misc
    'orig_cost', 'indep_remain_cost', 'acct_seq_num',
}

DATE_FIELDS = {
    'target_start_date', 'target_end_date',
    'act_start_date',    'act_end_date',
    'early_start_date',  'early_end_date',
    'late_start_date',   'late_end_date',
    'cstr_date',         'cstr_date2',
    'plan_start_date',   'plan_end_date',
    'last_recalc_date',  'scd_end_date',
    'start_date',        'end_date',
    'fin_dates_start_date', 'fin_dates_end_date',
    'nonwork_date',
}

REL_TYPE_MAP = {
    'PR_FS': 'FS', 'PR_SS': 'SS',
    'PR_FF': 'FF', 'PR_SF': 'SF',
}


# ─────────────────────────────────────────────────────────────────────────────
# Type conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_float(value) -> float:
    if value is None or value == '':
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _to_date(value) -> Optional[str]:
    """Return ISO date string YYYY-MM-DD or None. Validated via strptime."""
    if not value:
        return None
    s = str(value).strip()[:10]
    try:
        datetime.strptime(s, '%Y-%m-%d')
        return s
    except ValueError:
        return None


def _to_bool(value) -> bool:
    if value is None:
        return False
    return str(value).strip().upper() in ('Y', 'YES', '1', 'TRUE')


def _convert_row(row: Dict) -> Dict:
    """Convert a raw string row to proper Python types."""
    out = {}
    for key, val in row.items():
        if key.startswith('_'):
            continue
        if key in NUMERIC_FIELDS:
            out[key] = _to_float(val)
        elif key in DATE_FIELDS:
            out[key] = _to_date(val)
        else:
            out[key] = str(val).strip() if val is not None else ''
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class CompleteXERExtractor:
    """
    Parses a Primavera P6 XER file.

    Accepts filepath (str/Path) or raw bytes via `file_bytes` kwarg.
    Bytes mode preserves windows-1252 encoding — avoids corrupting
    special characters in task/WBS names when called from a web app.

    Usage:
        ex = CompleteXERExtractor('project.xer').extract_all()
        data = ex.structured      # full output dict
        report = data['parse_report']
        tasks  = data['tasks']
    """

    def __init__(
        self,
        filepath: str = None,
        file_type: str = 'baseline',
        near_critical_days: float = DEFAULT_NEAR_CRITICAL_DAYS,
        file_bytes: bytes = None,
        filename: str = None,
    ):
        self.filepath    = Path(filepath) if filepath else None
        self.file_bytes  = file_bytes
        self.file_type   = file_type
        self.filename    = filename or (self.filepath.name if self.filepath else 'unknown.xer')
        self.near_critical_hrs = near_critical_days * DEFAULT_HOURS_PER_DAY

        self.tables: Dict[str, List[Dict]]  = {}
        self.structured: Dict               = {}
        self.metadata: Dict                 = {}
        self.parsing_errors: List[str]      = []
        self._warnings: List[str]           = []
        self._calendar_hpd: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_all(self) -> 'CompleteXERExtractor':
        self._read_and_parse()
        self._build_calendar_hpd()

        lk       = self._build_lookups()
        wbs_data = self._build_hierarchy(lk['wbs'],  'wbs_id',    'parent_wbs_id',  'wbs_name')
        obs_data = self._build_hierarchy(lk['obs'],  'obs_id',    'parent_obs_id',  'obs_name')
        acc_data = self._build_account_hierarchy(lk['accounts'])

        tasks = self._build_all_task_objects(lk, wbs_data['paths'])

        self.structured = {
            'schema_version':       SCHEMA_VERSION,
            'project':              self._extract_project_info(lk),
            'tasks':                tasks,
            'wbs':                  wbs_data,
            'obs':                  obs_data,
            'resources':            self._build_resource_list(lk),
            'roles':                self._build_role_list(lk),
            'calendars':            self._build_calendar_list(lk),
            'cost_accounts':        acc_data,
            'financial_periods':    self._build_financial_periods(lk),
            'project_codes':        self._build_project_codes(lk),
            'resource_categories':  self._build_resource_categories(lk),
            'currencies':           [_convert_row(r) for r in self.tables.get('CURRTYPE', [])],
            'summary':              self._compute_summary(tasks),
            'parse_report':         self._generate_parse_report(),
            'tables':               self.tables,
        }

        report = self.structured['parse_report']
        print(
            f"[{self.filename}] Extracted {len(tasks)} tasks | "
            f"{report['tables_found']} tables | "
            f"coverage {report['coverage_pct']}% | "
            f"{len(self.parsing_errors)} errors"
        )
        return self

    # Compatibility
    def get_project_info(self)     -> Dict:        return self.structured.get('project', {})
    def get_all_tasks(self)        -> List[Dict]:  return self.structured.get('tasks', [])
    def get_wbs_structure(self)    -> List[Dict]:  return self.structured.get('wbs', {}).get('tree', [])
    def get_parse_report(self)     -> Dict:        return self.structured.get('parse_report', {})

    def save_to_json(self, output_path: str, include_raw_tables: bool = False):
        payload = {k: v for k, v in self.structured.items() if k != 'tables'}
        if include_raw_tables:
            payload['tables'] = self.tables
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved: {output_path}")

    # ── Step 1: Read & parse ──────────────────────────────────────────────────

    def _get_lines(self) -> List[str]:
        if self.file_bytes is not None:
            return self.file_bytes.decode('windows-1252', errors='replace').splitlines()
        with open(self.filepath, 'rb') as f:
            return f.read().decode('windows-1252', errors='replace').splitlines()

    def _read_and_parse(self):
        try:
            lines = self._get_lines()
        except Exception as e:
            self.parsing_errors.append(f"File read failed: {e}")
            raise

        if lines:
            parts = lines[0].strip().split('\t')
            if parts[0] == 'ERMHDR':
                self.metadata = {
                    'version':     parts[1] if len(parts) > 1 else '',
                    'export_date': parts[2] if len(parts) > 2 else '',
                    'username':    parts[4] if len(parts) > 4 else '',
                    'database':    parts[6] if len(parts) > 6 else '',
                    'file_type':   self.file_type,
                    'filename':    self.filename,
                }

        current_table  = None
        current_fields: List[str] = []

        for line_num, raw in enumerate(lines[1:], start=2):
            line = raw.rstrip('\r\n')
            if not line:
                continue
            parts  = line.split('\t')
            marker = parts[0]

            try:
                if marker == '%T':
                    current_table  = parts[1] if len(parts) > 1 else ''
                    self.tables.setdefault(current_table, [])
                    current_fields = []

                elif marker == '%F':
                    current_fields = parts[1:]

                elif marker == '%R':
                    if current_table and current_fields:
                        row = {
                            field: (parts[i + 1] if i + 1 < len(parts) else '')
                            for i, field in enumerate(current_fields)
                        }
                        self.tables[current_table].append(row)

            except Exception as e:
                self.parsing_errors.append(f"Line {line_num}: {e}")

    # ── Step 2: Calendar HPD ──────────────────────────────────────────────────

    def _build_calendar_hpd(self):
        for row in self.tables.get('CALENDAR', []):
            r   = _convert_row(row)
            cid = r.get('clndr_id', '')
            if not cid:
                continue
            hpd = r.get('day_hr_cnt', 0.0)
            self._calendar_hpd[cid] = hpd if hpd > 0 else DEFAULT_HOURS_PER_DAY

    def _hpd(self, clndr_id: str) -> float:
        return self._calendar_hpd.get(clndr_id, DEFAULT_HOURS_PER_DAY)

    # ── Step 3: Build all lookup dicts ────────────────────────────────────────

    def _build_lookups(self) -> Dict:
        lk: Dict[str, Any] = {
            # Single-record lookups
            'tasks':        {},
            'wbs':          {},
            'obs':          {},
            'resources':    {},
            'roles':        {},
            'calendars':    {},
            'accounts':     {},
            'actv_types':   {},   # ACTVTYPE  — type_id → {label, ...}
            'actv_codes':   {},   # ACTVCODE  — code_id → {short_name, type_id, ...}
            'udf_types':    {},   # UDFTYPE   — type_id → label string
            'fin_dates':    {},   # FINDATES  — period_id → {name, start, end}
            'memo_types':   {},   # MEMOTYPE  — type_id → label
            'currencies':   {},   # CURRTYPE  — currency_id → row
            'pcat_types':   {},   # PCATTYPE  — type_id → row
            'pcat_vals':    {},   # PCATVAL   — val_id  → row
            'rcat_types':   {},   # RCATTYPE  — type_id → row
            'rcat_vals':    {},   # RCATVAL   — val_id  → row
            # Multi-record lookups (keyed by task_id or rsrc_id)
            'predecessors':   defaultdict(list),
            'successors':     defaultdict(list),
            'task_resources': defaultdict(list),
            'task_actv':      defaultdict(list),
            'task_udfs':      defaultdict(dict),
            'task_memos':     defaultdict(list),
            'task_steps':     defaultdict(list),
            'task_fin':       defaultdict(list),   # TASKFIN  — period actuals
            'rsrc_rates':     defaultdict(list),   # RSRCRATE — resource rates
            'rsrc_roles':     defaultdict(list),   # RSRCROLE — resource→role
            'role_rates':     defaultdict(list),   # ROLERATE — role rates
            'nonwork':        defaultdict(list),   # NONWORK  — per calendar
            'proj_pcats':     defaultdict(list),   # PROJPCAT
            'rsrc_cats':      defaultdict(list),   # RSRCCAT
        }

        # ── Single-row tables ─────────────────────────────
        for row in self.tables.get('TASK', []):
            r = _convert_row(row); tid = r.get('task_id', '')
            if tid:
                if tid in lk['tasks']:
                    self._warnings.append(f"Duplicate task_id skipped: {tid}")
                    continue
                lk['tasks'][tid] = r

        for row in self.tables.get('PROJWBS', []):
            r = _convert_row(row); wid = r.get('wbs_id', '')
            if wid: lk['wbs'][wid] = r

        for row in self.tables.get('POBS', []):
            r = _convert_row(row); oid = r.get('obs_id', '')
            if oid: lk['obs'][oid] = r

        for row in self.tables.get('RSRC', []):
            r = _convert_row(row); rid = r.get('rsrc_id', '')
            if rid: lk['resources'][rid] = r

        for row in self.tables.get('ROLE', []):
            r = _convert_row(row); rid = r.get('role_id', '')
            if rid: lk['roles'][rid] = r

        for row in self.tables.get('CALENDAR', []):
            r = _convert_row(row); cid = r.get('clndr_id', '')
            if cid: lk['calendars'][cid] = r

        for row in self.tables.get('ACCOUNT', []):
            r = _convert_row(row); aid = r.get('acct_id', '')
            if aid: lk['accounts'][aid] = r

        for row in self.tables.get('ACTVTYPE', []):
            r = _convert_row(row); tid = r.get('actv_code_type_id', '')
            if tid: lk['actv_types'][tid] = r

        for row in self.tables.get('ACTVCODE', []):
            r = _convert_row(row); vid = r.get('actv_code_id', '')
            if vid: lk['actv_codes'][vid] = r

        for row in self.tables.get('UDFTYPE', []):
            r = _convert_row(row); uid = r.get('udf_type_id', '')
            if uid: lk['udf_types'][uid] = r.get('udf_type_label', uid)

        for row in self.tables.get('FINDATES', []):
            r = _convert_row(row); fid = r.get('fin_dates_id', '')
            if fid:
                lk['fin_dates'][fid] = {
                    'id':    fid,
                    'name':  r.get('fin_dates_name', ''),
                    'start': r.get('fin_dates_start_date'),
                    'end':   r.get('fin_dates_end_date'),
                }

        for row in self.tables.get('MEMOTYPE', []):
            r = _convert_row(row); mid = r.get('memo_type_id', '')
            if mid: lk['memo_types'][mid] = r.get('memo_type', mid)

        for row in self.tables.get('CURRTYPE', []):
            r = _convert_row(row); cid = r.get('curr_id', '')
            if cid: lk['currencies'][cid] = r

        for row in self.tables.get('PCATTYPE', []):
            r = _convert_row(row); pid = r.get('proj_catg_type_id', '')
            if pid: lk['pcat_types'][pid] = r

        for row in self.tables.get('PCATVAL', []):
            r = _convert_row(row); vid = r.get('proj_catg_id', '')
            if vid: lk['pcat_vals'][vid] = r

        for row in self.tables.get('RCATTYPE', []):
            r = _convert_row(row); rid = r.get('rsrc_catg_type_id', '')
            if rid: lk['rcat_types'][rid] = r

        for row in self.tables.get('RCATVAL', []):
            r = _convert_row(row); vid = r.get('rsrc_catg_id', '')
            if vid: lk['rcat_vals'][vid] = r

        # ── Multi-row lookups ─────────────────────────────
        for row in self.tables.get('TASKPRED', []):
            r = _convert_row(row)
            tid, pid = r.get('task_id', ''), r.get('pred_task_id', '')
            if tid and pid:
                lk['predecessors'][tid].append(r)
                lk['successors'][pid].append(r)

        for row in self.tables.get('TASKRSRC', []):
            r = _convert_row(row); tid = r.get('task_id', '')
            if tid: lk['task_resources'][tid].append(r)

        for row in self.tables.get('TASKACTV', []):
            r = _convert_row(row); tid = r.get('task_id', '')
            if tid: lk['task_actv'][tid].append(r)

        for row in self.tables.get('UDFVALUE', []):
            r = _convert_row(row)
            tid         = r.get('fk_id', '')
            udf_type_id = r.get('udf_type_id', '')
            if tid and udf_type_id:
                label = lk['udf_types'].get(udf_type_id, udf_type_id)
                # Explicit None check — avoids skipping '0' / 'False' strings
                value = (
                    r['udf_text']   if r.get('udf_text')   not in (None, '') else
                    r['udf_number'] if r.get('udf_number') not in (None, '') else
                    r.get('udf_date', '')
                )
                lk['task_udfs'][tid][label] = value

        for row in self.tables.get('TASKMEMO', []):
            r = _convert_row(row); tid = r.get('task_id', '')
            if tid:
                memo_type_id = r.get('memo_type_id', '')
                lk['task_memos'][tid].append({
                    'memo_type': lk['memo_types'].get(memo_type_id, memo_type_id),
                    'text':      r.get('task_memo', '').strip(),
                })

        for row in self.tables.get('TASKPROC', []):
            r = _convert_row(row); tid = r.get('task_id', '')
            if tid:
                lk['task_steps'][tid].append({
                    'step_id':    r.get('proc_id', ''),
                    'step_name':  r.get('proc_name', ''),
                    'description': r.get('proc_descr', ''),
                    'is_complete': _to_bool(r.get('complete_flag', '')),
                    'weight':      _to_float(r.get('proc_wt', 0)),
                })

        for row in self.tables.get('TASKFIN', []):
            r = _convert_row(row); tid = r.get('task_id', '')
            if tid:
                fin_id = r.get('fin_dates_id', '')
                period = lk['fin_dates'].get(fin_id, {'id': fin_id, 'name': fin_id})
                lk['task_fin'][tid].append({
                    'period_id':        fin_id,
                    'period_name':      period.get('name', ''),
                    'period_start':     period.get('start'),
                    'period_end':       period.get('end'),
                    'actual_work_qty':  r.get('act_work_qty', 0.0),
                    'actual_equip_qty': r.get('act_equip_qty', 0.0),
                    'actual_cost':      r.get('act_reg_cost', 0.0),
                    'actual_ot_cost':   r.get('act_ot_cost', 0.0),
                    'target_work_qty':  r.get('target_work_qty', 0.0),
                })

        for row in self.tables.get('RSRCRATE', []):
            r = _convert_row(row); rid = r.get('rsrc_id', '')
            if rid:
                lk['rsrc_rates'][rid].append({
                    'start_date':   r.get('start_date'),
                    'end_date':     r.get('end_date'),
                    'price_per_unit':    r.get('cost_per_qty', 0.0),
                    'price_per_unit_2':  r.get('cost_per_qty2', 0.0),
                    'price_per_unit_3':  r.get('cost_per_qty3', 0.0),
                    'max_qty_per_hr':    r.get('max_qty_per_hr', 0.0),
                })

        for row in self.tables.get('RSRCROLE', []):
            r = _convert_row(row); rid = r.get('rsrc_id', '')
            if rid:
                role_id   = r.get('role_id', '')
                role_info = lk['roles'].get(role_id, {})
                lk['rsrc_roles'][rid].append({
                    'role_id':   role_id,
                    'role_name': role_info.get('role_name', role_id),
                    'is_primary': _to_bool(r.get('prim_role_flag', '')),
                })

        for row in self.tables.get('ROLERATE', []):
            r = _convert_row(row); rid = r.get('role_id', '')
            if rid:
                lk['role_rates'][rid].append({
                    'start_date':      r.get('start_date'),
                    'end_date':        r.get('end_date'),
                    'price_per_unit':  r.get('cost_per_qty', 0.0),
                })

        for row in self.tables.get('NONWORK', []):
            r = _convert_row(row); cid = r.get('clndr_id', '')
            if cid:
                lk['nonwork'][cid].append({
                    'date':        r.get('nonwork_date'),
                    'nonwork_type': r.get('nonwork_type_id', ''),
                })

        for row in self.tables.get('PROJPCAT', []):
            r = _convert_row(row); pid = r.get('proj_id', '')
            if pid: lk['proj_pcats'][pid].append(r)

        for row in self.tables.get('RSRCCAT', []):
            r = _convert_row(row); rid = r.get('rsrc_id', '')
            if rid: lk['rsrc_cats'][rid].append(r)

        return lk

    # ── Step 4: Generic hierarchy builder ────────────────────────────────────

    def _build_hierarchy(
        self,
        node_lookup: Dict,
        id_field: str,
        parent_field: str,
        name_field: str,
    ) -> Dict:
        """
        Builds a tree + flat path dict from any parent-child node lookup.
        Used for both WBS (PROJWBS) and OBS (POBS).
        Includes cycle detection.
        """
        if not node_lookup:
            return {'tree': [], 'nodes': {}, 'paths': {}}

        all_ids      = set(node_lookup.keys())
        parent_map   = {nid: n.get(parent_field, '') for nid, n in node_lookup.items()}
        children_map = defaultdict(list)
        for nid in node_lookup:
            children_map[parent_map.get(nid, '')].append(nid)

        paths       = {}
        in_progress = set()

        def get_path(nid: str) -> str:
            if nid in paths:
                return paths[nid]
            if nid in in_progress:
                self._warnings.append(f"Circular hierarchy reference at id={nid} ({id_field})")
                paths[nid] = node_lookup[nid].get(name_field, nid)
                return paths[nid]
            in_progress.add(nid)
            parent = parent_map.get(nid, '')
            name   = node_lookup[nid].get(name_field, nid)
            paths[nid] = (get_path(parent) + ' > ' + name) if (parent and parent in all_ids) else name
            in_progress.discard(nid)
            return paths[nid]

        for nid in node_lookup:
            get_path(nid)

        # For WBS, include wbs_short_name as wbs_code; for OBS use obs_short_name
        code_field = 'wbs_short_name' if id_field == 'wbs_id' else 'obs_short_name'

        def build_node(nid):
            n = node_lookup[nid]
            return {
                id_field:   nid,
                'code':     n.get(code_field, ''),
                'name':     n.get(name_field, ''),
                'path':     paths.get(nid, ''),
                'children': [build_node(c) for c in sorted(children_map.get(nid, []))],
            }

        roots = [nid for nid in all_ids if parent_map.get(nid, '') not in all_ids]
        return {
            'tree':  [build_node(r) for r in sorted(roots)],
            'nodes': node_lookup,
            'paths': paths,
        }

    # ── Step 5: Account hierarchy ─────────────────────────────────────────────

    def _build_account_hierarchy(self, account_lookup: Dict) -> List[Dict]:
        """Flatten account hierarchy into a list with full path strings."""
        if not account_lookup:
            return []

        all_ids    = set(account_lookup.keys())
        parent_map = {aid: a.get('parent_acct_id', '') for aid, a in account_lookup.items()}
        paths      = {}
        in_progress: set = set()

        def get_path(aid: str) -> str:
            if aid in paths: return paths[aid]
            if aid in in_progress:
                paths[aid] = account_lookup[aid].get('acct_name', aid)
                return paths[aid]
            in_progress.add(aid)
            parent = parent_map.get(aid, '')
            name   = account_lookup[aid].get('acct_name', aid)
            paths[aid] = (get_path(parent) + ' > ' + name) if (parent and parent in all_ids) else name
            in_progress.discard(aid)
            return paths[aid]

        result = []
        for aid, acct in account_lookup.items():
            result.append({
                'acct_id':    aid,
                'acct_code':  acct.get('acct_short_name', ''),
                'acct_name':  acct.get('acct_name', ''),
                'path':       get_path(aid),
                'seq_num':    acct.get('acct_seq_num', 0.0),
                'description': acct.get('acct_descr', ''),
            })
        return sorted(result, key=lambda x: x['path'])

    # ── Step 6: Build task objects ────────────────────────────────────────────

    def _build_task_object(self, task: Dict, lk: Dict, wbs_paths: Dict) -> Dict:
        tid   = task.get('task_id', '')
        cid   = task.get('clndr_id', '')
        HPD   = self._hpd(cid)
        nct   = self.near_critical_hrs

        # Durations and float
        planned_hrs  = task.get('target_drtn_hr_cnt', 0.0)
        remain_hrs   = task.get('remain_drtn_hr_cnt', 0.0)
        float_hrs    = task.get('total_float_hr_cnt', 0.0)
        free_flt_hrs = task.get('free_float_hr_cnt', 0.0)
        pct          = task.get('phys_complete_pct', 0.0)

        # Predecessors
        pred_list = []
        for p in lk['predecessors'].get(tid, []):
            pt     = lk['tasks'].get(p.get('pred_task_id', ''), {})
            pt_hpd = self._hpd(pt.get('clndr_id', cid))
            pred_list.append({
                'task_id':   p.get('pred_task_id', ''),
                'task_code': pt.get('task_code', ''),
                'task_name': pt.get('task_name', ''),
                'type':      REL_TYPE_MAP.get(p.get('pred_type', ''), p.get('pred_type', '')),
                'lag_days':  round(p.get('lag_hr_cnt', 0.0) / pt_hpd, 2),
            })

        # Successors
        # Lag is applied on the predecessor's calendar (this task = predecessor here),
        # so use HPD (current task's HPD), not the successor's HPD.
        succ_list = []
        for s in lk['successors'].get(tid, []):
            st = lk['tasks'].get(s.get('task_id', ''), {})
            succ_list.append({
                'task_id':   s.get('task_id', ''),
                'task_code': st.get('task_code', ''),
                'task_name': st.get('task_name', ''),
                'type':      REL_TYPE_MAP.get(s.get('pred_type', ''), s.get('pred_type', '')),
                'lag_days':  round(s.get('lag_hr_cnt', 0.0) / HPD, 2),  # HPD = predecessor (this task)
            })

        # Resources (with rates)
        resource_list = []
        for r in lk['task_resources'].get(tid, []):
            rsrc      = lk['resources'].get(r.get('rsrc_id', ''), {})
            rsrc_id   = r.get('rsrc_id', '')
            # Roles assigned to this resource
            roles     = lk['rsrc_roles'].get(rsrc_id, [])
            resource_list.append({
                'resource_id':    rsrc_id,
                'resource_name':  rsrc.get('rsrc_name', rsrc_id),
                'resource_type':  rsrc.get('rsrc_type', ''),
                'calendar_id':    rsrc.get('clndr_id', ''),
                'planned_qty':    r.get('target_qty', 0.0),
                'actual_qty':     r.get('act_reg_qty', 0.0),
                'remaining_qty':  r.get('remain_qty', 0.0),
                'planned_cost':   r.get('target_cost', 0.0),
                'actual_cost':    r.get('act_reg_cost', 0.0),
                'remaining_cost': r.get('remain_cost', 0.0),
                'roles':          roles,
            })

        # Activity codes — labelled with type name from ACTVTYPE
        activity_codes = {}
        for ac in lk['task_actv'].get(tid, []):
            code_info  = lk['actv_codes'].get(ac.get('actv_code_id', ''), {})
            type_id    = code_info.get('actv_code_type_id', '')
            type_info  = lk['actv_types'].get(type_id, {})
            type_label = type_info.get('actv_code_type', type_id) or type_id
            code_label = code_info.get('short_name', ac.get('actv_code_id', ''))
            activity_codes[type_label] = code_label

        # Cost account
        acct_id   = task.get('acct_id', '')
        acct_info = lk['accounts'].get(acct_id, {})
        cost_account = {
            'acct_id':   acct_id,
            'acct_code': acct_info.get('acct_short_name', ''),
            'acct_name': acct_info.get('acct_name', ''),
        } if acct_id else None

        wbs_id = task.get('wbs_id', '')
        cstr   = task.get('cstr_type', '')

        return {
            # Identity
            'task_id':    tid,
            'task_code':  task.get('task_code', ''),
            'task_name':  task.get('task_name', ''),
            'task_type':  task.get('task_type', ''),
            'status':     task.get('status_code', ''),
            'wbs_id':     wbs_id,
            'wbs_path':   wbs_paths.get(wbs_id, wbs_id),
            'calendar_id':   cid,
            'hours_per_day': HPD,

            # Dates
            'dates': {
                'planned_start':  task.get('target_start_date'),
                'planned_finish': task.get('target_end_date'),
                'actual_start':   task.get('act_start_date'),
                'actual_finish':  task.get('act_end_date'),
                'early_start':    task.get('early_start_date'),
                'early_finish':   task.get('early_end_date'),
                'late_start':     task.get('late_start_date'),
                'late_finish':    task.get('late_end_date'),
            },

            # Duration (days, calendar-aware)
            'duration': {
                'planned_days':     round(planned_hrs / HPD, 2) if HPD else 0.0,
                'remaining_days':   round(remain_hrs  / HPD, 2) if HPD else 0.0,
                'percent_complete': pct,
            },

            # Float
            'float': {
                'total_float_days':   round(float_hrs    / HPD, 2) if HPD else 0.0,
                'free_float_days':    round(free_flt_hrs / HPD, 2) if HPD else 0.0,
                'is_critical':        float_hrs <= 0,
                'is_near_critical':   0 < float_hrs <= nct,
                'has_negative_float': float_hrs < 0,
            },

            # Relationships
            # is_open_ended: finish milestones legitimately have no successors → not flagged
            # is_dangling:   start milestones legitimately have no predecessors → not flagged
            'relationships': {
                'predecessors':      pred_list,
                'successors':        succ_list,
                'predecessor_count': len(pred_list),
                'successor_count':   len(succ_list),
                'is_open_ended':     len(succ_list) == 0 and task.get('task_type') != 'TT_FinMile',
                'is_dangling':       len(pred_list) == 0 and task.get('task_type') != 'TT_Mile',
            },

            # Resources
            'resources':     resource_list,
            'has_resources': len(resource_list) > 0,

            # Constraint
            'constraints': {'type': cstr, 'date': task.get('cstr_date')} if cstr else None,

            # Cost account
            'cost_account': cost_account,

            # Task notes/narratives (from TASKMEMO)
            'notes': lk['task_memos'].get(tid, []),

            # Task steps/procedures (from TASKPROC)
            'steps': lk['task_steps'].get(tid, []),

            # Financial period actuals (from TASKFIN + FINDATES)
            'period_actuals': lk['task_fin'].get(tid, []),

            # Activity codes (labelled)
            'activity_codes': activity_codes,

            # UDFs
            'custom_fields': lk['task_udfs'].get(tid, {}),
        }

    def _build_all_task_objects(self, lk: Dict, wbs_paths: Dict) -> List[Dict]:
        return [self._build_task_object(t, lk, wbs_paths) for t in lk['tasks'].values()]

    # ── Step 7: Top-level section builders ───────────────────────────────────

    def _extract_project_info(self, lk: Dict) -> Dict:
        projects = self.tables.get('PROJECT', [])
        if not projects:
            return {}
        proj = _convert_row(projects[0])

        # Scheduling options (one row per project)
        sched_opts = {}
        for row in self.tables.get('SCHEDOPTION', []):
            r = _convert_row(row)
            if r.get('proj_id', '') == proj.get('proj_id', ''):
                sched_opts = {
                    'lag_calendar':             r.get('sched_calendar_on_relationship_lag', ''),
                    'use_expected_finish':       _to_bool(r.get('sched_use_expect_end_flag', '')),
                    'retained_logic':            _to_bool(r.get('sched_retained_logic', '')),
                    'progressive_override':      _to_bool(r.get('sched_progress_override', '')),
                    'float_type':                r.get('sched_float_type', ''),
                    'critical_float_threshold':  r.get('sched_critical_float_type', ''),
                    'max_percent_complete':       _to_float(r.get('sched_use_project_end_date_for_float', 0)),
                    'out_of_sequence_logic':      r.get('sched_outer_depend_type', ''),
                }
                break

        return {
            'project_id':        proj.get('proj_id', ''),
            'project_name':      proj.get('proj_short_name', ''),
            'data_date':         proj.get('last_recalc_date', ''),
            'plan_start':        proj.get('plan_start_date'),
            'plan_finish':       proj.get('scd_end_date'),
            'status':            proj.get('status_code', ''),
            'scheduling_options': sched_opts,
        }

    def _build_resource_list(self, lk: Dict) -> List[Dict]:
        result = []
        for rid, rsrc in lk['resources'].items():
            result.append({
                'resource_id':   rid,
                'resource_name': rsrc.get('rsrc_name', ''),
                'resource_type': rsrc.get('rsrc_type', ''),
                'calendar_id':   rsrc.get('clndr_id', ''),
                'parent_rsrc_id': rsrc.get('parent_rsrc_id', ''),
                'roles':         lk['rsrc_roles'].get(rid, []),
                'rates':         lk['rsrc_rates'].get(rid, []),
                'categories':    self._resource_categories_for(rid, lk),
            })
        return result

    def _resource_categories_for(self, rsrc_id: str, lk: Dict) -> List[Dict]:
        cats = []
        for cat in lk['rsrc_cats'].get(rsrc_id, []):
            cat_id    = cat.get('rsrc_catg_id', '')
            cat_val   = lk['rcat_vals'].get(cat_id, {})
            type_id   = cat_val.get('rsrc_catg_type_id', '')
            type_info = lk['rcat_types'].get(type_id, {})
            cats.append({
                'category_type':  type_info.get('rsrc_catg_type', type_id),
                'category_value': cat_val.get('rsrc_catg_name', cat_id),
            })
        return cats

    def _build_role_list(self, lk: Dict) -> List[Dict]:
        result = []
        for rid, role in lk['roles'].items():
            result.append({
                'role_id':       rid,
                'role_name':     role.get('role_name', ''),
                'role_code':     role.get('role_short_name', ''),
                'parent_role_id': role.get('parent_role_id', ''),
                'rates':         lk['role_rates'].get(rid, []),
            })
        return result

    def _build_calendar_list(self, lk: Dict) -> List[Dict]:
        result = []
        for cid, cal in lk['calendars'].items():
            result.append({
                'calendar_id':   cid,
                'calendar_name': cal.get('clndr_name', ''),
                'calendar_type': cal.get('clndr_type', ''),
                'hours_per_day': self._hpd(cid),
                'hours_per_week': cal.get('week_hr_cnt', 0.0),
                'non_work_days': lk['nonwork'].get(cid, []),
            })
        return result

    def _build_financial_periods(self, lk: Dict) -> List[Dict]:
        return sorted(lk['fin_dates'].values(), key=lambda x: x.get('start') or '')

    def _build_project_codes(self, lk: Dict) -> Dict:
        """Structure project codes as {type_name: [{value, assignments}]}"""
        types  = {tid: r.get('proj_catg_type', tid) for tid, r in lk['pcat_types'].items()}
        values: Dict[str, List] = defaultdict(list)

        for vid, val in lk['pcat_vals'].items():
            type_id    = val.get('proj_catg_type_id', '')
            type_label = types.get(type_id, type_id)
            values[type_label].append({
                'code_id':    vid,
                'code_value': val.get('proj_catg_name', ''),
                'short_name': val.get('short_name', ''),
                'seq_num':    _to_float(val.get('seq_num', 0)),
            })

        return dict(values)

    def _build_resource_categories(self, lk: Dict) -> Dict:
        types  = {tid: r.get('rsrc_catg_type', tid) for tid, r in lk['rcat_types'].items()}
        values: Dict[str, List] = defaultdict(list)

        for vid, val in lk['rcat_vals'].items():
            type_id    = val.get('rsrc_catg_type_id', '')
            type_label = types.get(type_id, type_id)
            values[type_label].append({
                'cat_id':   vid,
                'cat_name': val.get('rsrc_catg_name', ''),
            })

        return dict(values)

    # ── Step 8: Summary ───────────────────────────────────────────────────────

    def _compute_summary(self, tasks: List[Dict]) -> Dict:
        work = [t for t in tasks if t['task_type'] not in ('TT_LOE', 'TT_Mile', 'TT_FinMile')]
        tasks_with_notes  = sum(1 for t in tasks if t.get('notes'))
        tasks_with_steps  = sum(1 for t in tasks if t.get('steps'))
        tasks_with_actuals = sum(1 for t in tasks if t.get('period_actuals'))

        return {
            'total':              len(tasks),
            'work_tasks':         len(work),
            'milestones':         sum(1 for t in tasks if 'Mile' in t['task_type']),
            'loe':                sum(1 for t in tasks if t['task_type'] == 'TT_LOE'),
            'completed':          sum(1 for t in tasks if t['status'] == 'TK_Complete'),
            'in_progress':        sum(1 for t in tasks if t['status'] == 'TK_Active'),
            'not_started':        sum(1 for t in tasks if t['status'] == 'TK_NotStart'),
            'critical':           sum(1 for t in work if t['float']['is_critical']),
            'near_critical':      sum(1 for t in work if t['float']['is_near_critical']),
            'negative_float':     sum(1 for t in work if t['float']['has_negative_float']),
            'open_ended':         sum(1 for t in work if t['relationships']['is_open_ended']),
            'dangling':           sum(1 for t in work if t['relationships']['is_dangling']),
            'long_duration':      sum(1 for t in work if t['duration']['planned_days'] > 30),
            'constrained':        sum(1 for t in tasks if t['constraints'] is not None),
            'resource_loaded':    sum(1 for t in work if t['has_resources']),
            'with_notes':         tasks_with_notes,
            'with_steps':         tasks_with_steps,
            'with_period_actuals': tasks_with_actuals,
            'with_cost_account':  sum(1 for t in tasks if t.get('cost_account')),
        }

    # ── Step 9: Parse report ──────────────────────────────────────────────────

    def _generate_parse_report(self) -> Dict:
        """
        Per-table accounting of what was found, processed, and left raw.
        Gives a clear picture of data coverage for any given XER file.
        """
        # Where each table's data ends up in the structured output
        TABLE_DESTINATIONS = {
            'PROJECT':     'project',
            'PROJWBS':     'wbs',
            'SCHEDOPTION': 'project.scheduling_options',
            'TASK':        'tasks[]',
            'TASKPRED':    'tasks[].relationships',
            'TASKRSRC':    'tasks[].resources[]',
            'TASKACTV':    'tasks[].activity_codes',
            'TASKMEMO':    'tasks[].notes[]',
            'TASKPROC':    'tasks[].steps[]',
            'TASKFIN':     'tasks[].period_actuals[]',
            'RSRC':        'resources[]',
            'RSRCRATE':    'resources[].rates[]  +  tasks[].resources[].rates[]',
            'ROLE':        'roles[]',
            'RSRCROLE':    'resources[].roles[]  +  tasks[].resources[].roles[]',
            'ROLERATE':    'roles[].rates[]',
            'CALENDAR':    'calendars[]',
            'NONWORK':     'calendars[].non_work_days[]',
            'CLNDRDATA':   'tables.CLNDRDATA  (raw — proprietary format)',
            'ACTVTYPE':    'tasks[].activity_codes  (type labels)',
            'ACTVCODE':    'tasks[].activity_codes  (code values)',
            'UDFTYPE':     'tasks[].custom_fields  (field labels)',
            'UDFVALUE':    'tasks[].custom_fields',
            'ACCOUNT':     'cost_accounts[]  +  tasks[].cost_account',
            'FINDATES':    'financial_periods[]  +  tasks[].period_actuals[].period_*',
            'FINTMPL':     'financial_periods  (template reference — raw)',
            'POBS':        'obs',
            'PCATTYPE':    'project_codes  (type labels)',
            'PCATVAL':     'project_codes  (code values)',
            'PROJPCAT':    'project_codes  (assignments)',
            'RCATTYPE':    'resource_categories  (type labels)',
            'RCATVAL':     'resource_categories  (cat values)',
            'RSRCCAT':     'resources[].categories[]',
            'MEMOTYPE':    'tasks[].notes[].memo_type  (labels)',
            'CURRTYPE':    'currencies[]',
            'PROJCOST':    'tables.PROJCOST  (raw — period cost summary)',
        }

        tables_detail = {}
        total_rows    = 0
        processed_rows = 0

        for table_name, rows in self.tables.items():
            row_count = len(rows)
            total_rows += row_count

            if table_name in RAW_ONLY_TABLES:
                status = 'raw_only'
            elif table_name in PROCESSED_TABLES:
                status = 'processed'
                processed_rows += row_count
            else:
                status = 'unknown_raw'

            tables_detail[table_name] = {
                'rows':        row_count,
                'status':      status,
                'destination': TABLE_DESTINATIONS.get(table_name, 'tables (raw — unknown table)'),
            }

        # Tables declared in spec but absent from this file
        absent = []
        for t in (PROCESSED_TABLES | RAW_ONLY_TABLES):
            if t not in self.tables:
                absent.append(t)

        unknown_tables = [t for t in self.tables if t not in PROCESSED_TABLES | RAW_ONLY_TABLES]

        coverage = round(processed_rows / total_rows * 100, 1) if total_rows > 0 else 0.0

        return {
            'tables_found':    len(self.tables),
            'total_rows':      total_rows,
            'processed_rows':  processed_rows,
            'coverage_pct':    coverage,
            'tables_detail':   tables_detail,      # per-table breakdown
            'absent_tables':   sorted(absent),     # in spec, not in this file
            'unknown_tables':  unknown_tables,     # in file, not in spec
            'parsing_errors':  self.parsing_errors,
            'warnings':        self._warnings,
            'near_critical_threshold_hrs': self.near_critical_hrs,
        }
