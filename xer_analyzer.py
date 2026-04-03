"""
XER Schedule Analyzer — Fixed LLM Pipeline
============================================
Fixes applied vs original:
  1. _create_dataframes: skips unused bulk tables (191k-row TASKACTV, etc.)
  2. XERAnalyzer.__init__: loud warning when XERContextBuilder import fails
  3. get_system_prompt: strict anti-hallucination prompt
  4. get_response_prompt: validates context has real rows; flags aggregate-only fallback
  5. Near-critical threshold: single constant (NEAR_CRITICAL_HRS) used everywhere
  6. compute_basic_stats: open-ended/dangling computed on work tasks only
"""

from __future__ import annotations

import json
import textwrap
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(dtype=object)


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(0, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(0)


def _safe_records(df: pd.DataFrame, cols: List[str]) -> List[Dict]:
    available = [c for c in cols if c in df.columns]
    if not available or df.empty:
        return []
    return df[available].fillna("").to_dict("records")


def _fmt_float(v) -> str:
    try:
        return f"{float(v):,.1f}"
    except Exception:
        return "N/A"


def _fmt_int(v) -> str:
    try:
        return f"{int(float(v)):,}"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Near-critical threshold — single source of truth
# ---------------------------------------------------------------------------

NEAR_CRITICAL_HRS  = 80   # hours  (8 working days at 10 h/day)
NEAR_CRITICAL_DAYS = 8    # days   (derived — update both together)


# ---------------------------------------------------------------------------
# 1. Intent Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    INTENT_PATTERNS: Dict[str, List[str]] = {
        "critical_path": [
            "critical", "critical path", "float", "near-critical", "near critical",
            "longest path", "driving", "zero float", "negative float", "on the critical",
        ],
        "quality": [
            "quality", "open end", "open-end", "dangling", "dcma", "14 point",
            "constraint", "lag", "hard constraint", "soft logic", "relationship",
            "logic", "check", "audit", "review", "issue", "problem", "health",
            "loe", "level of effort",
        ],
        "delays": [
            "delay", "slip", "slippage", "late", "overdue", "behind schedule",
            "behind", "missed", "trend", "erosion", "drift",
        ],
        "resources": [
            "resource", "cost", "budget", "spend", "loaded", "unloaded",
            "manpower", "labour", "labor", "equipment", "material",
            "utiliz", "cpi", "eac", "spi", "ev", "earned value",
        ],
        "progress": [
            "progress", "complete", "percent complete", "% complete", "finished",
            "status", "done", "how much", "completion",
        ],
        "milestones": [
            "milestone", "key date", "completion date", "handover", "finish date",
            "target date", "key event", "start date",
        ],
        "comparison": [
            "compare", "comparison", "vs", "versus", "baseline vs",
            "change", "difference", "delta", "before", "after", "update",
            "month", "period",
        ],
        "wbs": [
            "wbs", "work breakdown", "area", "zone", "section", "package",
            "discipline", "phase", "level",
        ],
        "duration": [
            "long duration", "short duration", "how long", "duration",
            "days long", "longest", "shortest", "average duration",
        ],
        "list_activities": [
            "list", "show me", "give me", "what activities", "which activities",
            "find activities", "enumerate", "all activities",
        ],
    }

    @classmethod
    def classify(cls, query: str) -> List[str]:
        q = query.lower()
        matched = [
            intent
            for intent, patterns in cls.INTENT_PATTERNS.items()
            if any(p in q for p in patterns)
        ]
        return matched if matched else ["general"]

    @classmethod
    def primary_intent(cls, query: str) -> str:
        intents = cls.classify(query)
        priority = [
            "critical_path", "quality", "delays", "resources",
            "progress", "milestones", "comparison", "wbs",
            "duration", "list_activities", "general",
        ]
        for p in priority:
            if p in intents:
                return p
        return intents[0] if intents else "general"


# ---------------------------------------------------------------------------
# 2. Anti-hallucination system prompt
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """
You are a Primavera P6 Schedule Analyst answering questions about a specific project.

═══ ABSOLUTE RULES — NEVER BREAK THESE ══════════════════════════════════════

1. ONLY USE DATA FROM THE TABLES PROVIDED.
   Every task code, activity name, date, and number you state MUST appear
   verbatim in the context tables. Do not infer, estimate, or invent.

2. NEVER GENERATE TASK CODES.
   If a table shows 421 critical activities but only 40 are listed, you may
   say "421 critical activities; the 40 shown here are:" and list only those 40.
   Do NOT produce codes for the remaining 381.

3. IF DATA IS ABSENT, SAY SO EXPLICITLY.
   State exactly what is missing and what the user would need to load to
   answer the question fully.

4. TRUNCATION DISCLOSURE.
   When a table says "(up to N shown)" or "top N of M", state how many total
   exist vs how many are shown.

5. NO GENERIC ADVICE unless explicitly asked for recommendations.

6. NUMBERS FROM DATA ONLY.
   All counts, percentages, durations and dates must come from the provided
   tables. Never compute values yourself unless doing a simple sum/difference
   of values already shown.

══════════════════════════════════════════════════════════════════════════════

FORMAT RULES:
- Lead with the direct answer in one sentence.
- Then support with a table or bullet list using only data from context.
- Float values are in WORKING DAYS unless stated otherwise.
""".strip()

_INTENT_SUFFIX: Dict[str, str] = {
    "critical_path": (
        "\nFORMAT: Summary line → critical activities table "
        "(Code|Name|Float(d)|Dur(d)|Status) → WBS breakdown if available."
    ),
    "quality": "\nFORMAT: DCMA-style PASS/FAIL table → detail only for FAIL items.",
    "delays":  "\nFORMAT: Project slippage headline → top slipped activities table → overdue list.",
    "resources": (
        "\nFORMAT: Cost summary table (BAC|AC|Remain|EAC|CPI) → "
        "unresourced list → top resources by budget."
    ),
    "milestones": "\nFORMAT: All milestones sorted by date (Code|Name|Date|Float(d)|Status).",
    "comparison": "\nFORMAT: Header of what is being compared → delta table → changed activities.",
    "wbs":      "\nFORMAT: WBS table (Name|Activity Count|Critical Count|Avg Float(d)).",
    "duration": (
        "\nFORMAT: Summary stats → long-duration table (Code|Name|Dur(d)|Float(d)|Status) "
        "→ DCMA threshold result."
    ),
}


class SystemPromptLibrary:
    @classmethod
    def get(cls, primary_intent: str) -> str:
        return _SYSTEM_BASE + _INTENT_SUFFIX.get(primary_intent, "")


# ---------------------------------------------------------------------------
# 3. Legacy ContextBuilder (fallback only — no real rows)
# ---------------------------------------------------------------------------

class ContextBuilder:
    HPD = 10

    def build(self, query, intents, basic_stats, insights,
              code_result=None, code_error=None):
        sections = [f"USER QUESTION: {query}\n"]
        sections.append(self._project_header(basic_stats))
        for intent in intents:
            builder = getattr(self, f"_build_{intent}", None)
            if builder:
                section = builder(basic_stats, insights)
                if section:
                    sections.append(section)
        if code_result is not None:
            sections.append(self._code_result_section(code_result))
        elif code_error:
            sections.append(
                f"\n[Note: Dynamic query failed ({code_error}). "
                f"Use pre-computed data above.]\n"
            )
        sections.append(
            "\nINSTRUCTION: Answer using ONLY the data above. "
            "Be specific — use exact activity codes, counts, and dates. "
            "Never make up numbers not present in the context."
        )
        return "\n".join(sections)

    def _project_header(self, s):
        return textwrap.dedent(f"""
            ── PROJECT OVERVIEW ──────────────────────────────────────
            Project      : {s.get('data_source', 'N/A')}
            Data Date    : {s.get('data_date', 'N/A')}
            Period       : {s.get('project_start', 'N/A')} → {s.get('project_finish', 'N/A')}
            Total Activities : {_fmt_int(s.get('total_activities', 0))}
              Work tasks   : {_fmt_int(s.get('regular_tasks', 0))}
              Milestones   : {_fmt_int(s.get('milestones', 0))}
              LOE          : {_fmt_int(s.get('loe_activities', 0))}
            Status       : {_fmt_int(s.get('completed', 0))} complete | {_fmt_int(s.get('in_progress', 0))} in-progress | {_fmt_int(s.get('not_started', 0))} not-started
            Files loaded : Baseline={s.get('baseline_name', 'none')} | Updates={s.get('update_count', 0)}
            ──────────────────────────────────────────────────────────
        """).strip()

    def _build_critical_path(self, s, insights):
        cp   = insights.get("critical_path", {})
        summ = cp.get("summary", {})
        lines = ["\n── CRITICAL PATH DATA ──────────────────────────────────"]
        lines.append(
            f"Critical activities : {_fmt_int(summ.get('critical_count', s.get('critical_count', 0)))} "
            f"({summ.get('critical_pct', s.get('critical_pct', 0))}% of work tasks)"
        )
        lines.append(
            f"Near-critical (≤{NEAR_CRITICAL_DAYS}d float) : "
            f"{_fmt_int(summ.get('near_critical_count', s.get('near_critical_count', 0)))}"
        )
        lines.append(f"Negative float activities : {_fmt_int(s.get('negative_float_count', 0))}")
        crit_list = cp.get("critical_activities", [])
        if crit_list:
            lines.append(f"\nTop critical activities (up to 20 of {len(crit_list)} shown):")
            lines.append(f"{'Code':<30} {'Name':<45} {'Float(d)':>9} {'Dur(d)':>7} {'Status':<12}")
            lines.append("-" * 106)
            for act in crit_list[:20]:
                code = str(act.get("task_code", ""))[:29]
                name = str(act.get("task_name", ""))[:44]
                flt  = _fmt_float(act.get("_float_days", act.get("float_days", 0)))
                dur  = _fmt_float(act.get("_dur_days",   act.get("dur_days",   0)))
                stat = str(act.get("status_code", "")).replace("TK_", "")[:11]
                lines.append(f"{code:<30} {name:<45} {flt:>9} {dur:>7} {stat:<12}")
            if len(crit_list) > 20:
                lines.append(f"  ... and {len(crit_list) - 20} more (not shown)")
        by_wbs = cp.get("critical_by_wbs", [])
        if by_wbs:
            lines.append("\nCritical activities by WBS (top 10):")
            lines.append(f"{'WBS Name':<50} {'Critical':>8} {'Dur(d)':>8}")
            lines.append("-" * 68)
            for w in by_wbs[:10]:
                name = str(w.get("wbs_name") or w.get("wbs_id", "?"))[:49]
                cnt  = _fmt_int(w.get("critical_count", 0))
                dur  = _fmt_float(w.get("total_duration_days", 0))
                lines.append(f"{name:<50} {cnt:>8} {dur:>8}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_quality(self, s, insights):
        total_work = max(s.get("regular_tasks", s.get("total_activities", 1)), 1)
        lines = ["\n── SCHEDULE QUALITY METRICS ────────────────────────────"]
        checks = [
            ("Open-ended (no successor)",  s.get("open_ended_count", 0),        "< 5%", total_work),
            ("Dangling (no predecessor)",  s.get("dangling_count", 0),          "< 5%", total_work),
            ("Negative float",             s.get("negative_float_count", 0),    "= 0",  None),
            ("Long duration (>20d)",       s.get("long_duration_count", 0),     "< 5%", total_work),
            ("Constrained activities",     s.get("constrained_activities", 0),  "< 5%", total_work),
            ("Relationships with lag",     s.get("relationships_with_lag", 0),  "< 5%", max(s.get("total_relationships", 1), 1)),
            ("Negative lags",              s.get("negative_lags", 0),           "= 0",  None),
        ]
        lines.append(f"{'Check':<35} {'Count':>7} {'Pct':>7} {'Threshold':>11} {'Result':>8}")
        lines.append("-" * 72)
        for name, count, threshold, denom in checks:
            pct_str = f"{count / denom * 100:.1f}%" if denom else "—"
            if threshold == "= 0":
                result = "✓ PASS" if count == 0 else "✗ FAIL"
            elif "5%" in threshold:
                pct = (count / denom * 100) if denom else 0
                result = "✓ PASS" if pct < 5 else "✗ FAIL"
            else:
                result = "—"
            lines.append(f"{name:<35} {_fmt_int(count):>7} {pct_str:>7} {threshold:>11} {result:>8}")
        rel_types = s.get("relationship_types", {})
        if rel_types:
            total_rels = max(s.get("total_relationships", 1), 1)
            lines.append(f"\nRelationship type breakdown (total: {_fmt_int(total_rels)}):")
            for rtype, cnt in sorted(rel_types.items(), key=lambda x: -x[1]):
                lines.append(f"  {rtype:<10}: {_fmt_int(cnt):>6} ({cnt/total_rels*100:.1f}%)")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_delays(self, s, insights):
        di    = insights.get("delays", {})
        lines = ["\n── DELAY & SLIPPAGE DATA ───────────────────────────────"]
        slip  = di.get("project_end_slippage_days")
        if slip is not None:
            direction = "behind schedule" if slip > 0 else ("ahead of schedule" if slip < 0 else "on schedule")
            lines.append(f"Project end slippage : {slip:+d} days ({direction})")
            lines.append(f"  Baseline end : {di.get('baseline_project_end', 'N/A')}")
            lines.append(f"  Current end  : {di.get('current_project_end', 'N/A')}")
        else:
            lines.append("Project end slippage : N/A (requires baseline + update)")
        lines.append(f"Slipped activities   : {_fmt_int(di.get('slipped_activities_count', 0))}")
        lines.append(f"Overdue not-started  : {_fmt_int(s.get('overdue_not_started', 0))}")
        lines.append(f"Overdue in-progress  : {_fmt_int(s.get('overdue_in_progress', 0))}")
        slipped = di.get("most_slipped_activities", [])
        if slipped:
            lines.append(f"\nTop slipped activities (up to 15 of {len(slipped)} shown):")
            lines.append(f"{'Code':<30} {'Name':<40} {'BL Finish':<12} {'Curr Finish':<12} {'Slip':>6}")
            lines.append("-" * 103)
            for a in slipped[:15]:
                code = str(a.get("task_code", ""))[:29]
                name = str(a.get("task_name", ""))[:39]
                bl   = str(a.get("bl_end_date", ""))[:10]
                cu   = str(a.get("target_end_date", ""))[:10]
                sl   = str(a.get("slippage_days", ""))
                lines.append(f"{code:<30} {name:<40} {bl:<12} {cu:<12} {sl:>6}d")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_resources(self, s, insights):
        ri   = insights.get("resources", {})
        cs   = ri.get("cost_summary", {})
        lines = ["\n── RESOURCE & COST DATA ─────────────────────────────────"]
        if cs:
            b   = cs.get("total_budgeted_cost", 0)
            a   = cs.get("total_actual_cost",   0)
            r   = cs.get("total_remaining_cost",0)
            eac = cs.get("estimate_at_completion", 0)
            cpi = cs.get("cost_performance_index")
            cv  = cs.get("cost_variance", 0)
            lines.append(f"Budget (BAC)  : {b:>15,.2f}")
            lines.append(f"Actual (AC)   : {a:>15,.2f}")
            lines.append(f"Remaining     : {r:>15,.2f}")
            lines.append(f"EAC           : {eac:>15,.2f}")
            lines.append(f"Cost Variance : {cv:>+15,.2f}  ({'over' if cv > 0 else 'under'} budget)")
            lines.append(f"CPI           : {f'{cpi:.3f}' if cpi else 'N/A':>15}")
        else:
            lines.append("No cost data loaded (resource costs are zero or not assigned).")
        lines.append(f"\nResource loading : {_fmt_int(s.get('tasks_with_resources', 0))} activities ({s.get('resource_loaded_pct', 0)}%) have resource assignments")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_progress(self, s, insights):
        total    = max(s.get("total_activities", 1), 1)
        completed = s.get("completed", 0)
        in_prog   = s.get("in_progress", 0)
        lines = ["\n── PROGRESS DATA ────────────────────────────────────────"]
        lines.append(f"Overall completion : {completed/total*100:.1f}%")
        lines.append(f"  Completed        : {_fmt_int(completed)}")
        lines.append(f"  In progress      : {_fmt_int(in_prog)}")
        lines.append(f"  Not started      : {_fmt_int(s.get('not_started', 0))}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_milestones(self, s, insights):
        cp    = insights.get("critical_path", {})
        lines = ["\n── MILESTONE DATA ───────────────────────────────────────"]
        lines.append(f"Total milestones : {_fmt_int(s.get('milestones', 0))}")
        crit_list  = cp.get("critical_activities", [])
        ms_types   = {"TT_Mile", "TT_FinMile"}
        critical_ms = [a for a in crit_list if a.get("task_type") in ms_types]
        if critical_ms:
            lines.append(f"Critical milestones: {len(critical_ms)}")
            lines.append(f"{'Code':<30} {'Name':<45} {'Target Date':<13}")
            lines.append("-" * 90)
            for m in critical_ms[:20]:
                lines.append(f"{str(m.get('task_code','')):<30} {str(m.get('task_name','')):<45} {str(m.get('target_end_date',''))[:10]:<13}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_comparison(self, s, insights):
        di    = insights.get("delays", {})
        cp    = insights.get("critical_path", {})
        lines = ["\n── COMPARISON DATA ──────────────────────────────────────"]
        lines.append(f"Baseline   : {s.get('baseline_name', 'N/A')} ({s.get('baseline_date', 'N/A')})")
        lines.append(f"Updates    : {s.get('update_count', 0)}")
        slip = di.get("project_end_slippage_days")
        if slip is not None:
            lines.append(f"\nProject end slippage : {slip:+d} days")
            lines.append(f"  Baseline end : {di.get('baseline_project_end', 'N/A')}")
            lines.append(f"  Current end  : {di.get('current_project_end', 'N/A')}")
        changes = cp.get("critical_path_changes", [])
        if changes:
            newly = [c for c in changes if c["change"] == "became_critical"]
            left  = [c for c in changes if c["change"] == "left_critical"]
            lines.append(f"\nCritical path changes:")
            lines.append(f"  Newly critical : {len(newly)}")
            lines.append(f"  Left critical  : {len(left)}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_wbs(self, s, insights):
        cp    = insights.get("critical_path", {})
        lines = ["\n── WBS DATA ─────────────────────────────────────────────"]
        by_wbs = cp.get("critical_by_wbs", [])
        if by_wbs:
            lines.append(f"{'WBS Name':<50} {'Critical':>9} {'Dur(d)':>8} {'AvgFloat':>9}")
            lines.append("-" * 80)
            for w in by_wbs[:20]:
                name = str(w.get("wbs_name") or w.get("wbs_id", "?"))[:49]
                cnt  = _fmt_int(w.get("critical_count", 0))
                dur  = _fmt_float(w.get("total_duration_days", 0))
                flt  = _fmt_float(w.get("avg_float_days", 0))
                lines.append(f"{name:<50} {cnt:>9} {dur:>8} {flt:>9}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_duration(self, s, insights):
        lines = ["\n── DURATION ANALYSIS DATA ───────────────────────────────"]
        lines.append(f"Average duration      : {_fmt_float(s.get('avg_duration_days', 0))} days")
        lines.append(f"Maximum duration      : {_fmt_float(s.get('max_duration_days', 0))} days")
        lines.append(f"Long duration (>20d)  : {_fmt_int(s.get('long_duration_count', 0))} activities")
        total_work = max(s.get("regular_tasks", 1), 1)
        ld_count   = s.get("long_duration_count", 0)
        ld_pct     = ld_count / total_work * 100
        lines.append(
            f"DCMA threshold (>20d) : {ld_count}/{total_work} = {ld_pct:.1f}% "
            f"({'⚠ FAIL' if ld_pct > 5 else '✓ PASS'})"
        )
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def _build_general(self, s, insights):
        lines = ["\n── SCHEDULE SNAPSHOT ────────────────────────────────────"]
        lines.append(f"Critical path  : {_fmt_int(s.get('critical_count', 0))} activities ({s.get('critical_pct', 0)}%)")
        lines.append(f"Near-critical  : {_fmt_int(s.get('near_critical_count', 0))} activities")
        lines.append(f"Negative float : {_fmt_int(s.get('negative_float_count', 0))} activities")
        lines.append(f"Open-ended     : {_fmt_int(s.get('open_ended_count', 0))} activities")
        lines.append(f"Long duration  : {_fmt_int(s.get('long_duration_count', 0))} activities (>20d)")
        lines.append(f"Constrained    : {_fmt_int(s.get('constrained_activities', 0))} activities")
        lines.append(f"Relationships  : {_fmt_int(s.get('total_relationships', 0))} total | {_fmt_int(s.get('relationships_with_lag', 0))} with lag")
        lines.append(f"Resource load  : {s.get('resource_loaded_pct', 0)}%")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    # aliases for intent names that don't directly map
    _build_list_activities = _build_general

    def _code_result_section(self, result):
        lines = ["\n── DYNAMIC QUERY RESULT ─────────────────────────────────"]
        if isinstance(result, list) and result and isinstance(result[0], dict):
            keys = list(result[0].keys())[:8]
            lines.append("  " + " | ".join(f"{k[:20]:<20}" for k in keys))
            lines.append("  " + "-" * min(len(keys) * 23, 120))
            for row in result[:30]:
                cells = [str(row.get(k, ""))[:20] for k in keys]
                lines.append("  " + " | ".join(f"{c:<20}" for c in cells))
            if len(result) > 30:
                lines.append(f"  ... and {len(result) - 30} more rows")
        elif isinstance(result, dict):
            for k, v in list(result.items())[:20]:
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {result}")
        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. XERDataStore
# ---------------------------------------------------------------------------

# Tables we never need as DataFrames — skip them on load to save memory/time
_SKIP_TABLES = {
    "taskactv",    # 191k rows — activity code assignments, not used in analytics DFs
    "udfvalue",    # 40k rows  — UDF values, accessed via raw dicts if needed
    "projcost",
    "finimpl",
    "fintmpl",
    "obs",
    "umeasure",
    "currtype",
    "rsrcrate",
    "wbsbudg",
    "schedoptions",
}

_KEEP_TABLES = {
    "task", "taskpred", "taskrsrc", "projwbs",
    "rsrc", "calendar", "project",
}


class XERDataStore:
    LOE_TYPES       = {"TT_LOE"}
    MILESTONE_TYPES = {"TT_Mile", "TT_FinMile"}
    EXCLUDED_TYPES  = LOE_TYPES | MILESTONE_TYPES

    def __init__(self):
        self.baseline:      Optional[Dict] = None
        self.updates:       List[Dict]     = []
        self.hours_per_day: int            = 10
        self._cached_stats: Optional[Dict] = None

    def load_baseline(self, data: Dict, name: str, data_date: str):
        self.baseline = {
            "name": name, "data_date": data_date, "data": data,
            "df":   self._create_dataframes(data),
        }
        self._cached_stats = None

    def add_update(self, data: Dict, name: str, data_date: str):
        self.updates.append({
            "name": name, "data_date": data_date, "data": data,
            "df":   self._create_dataframes(data),
        })
        self.updates.sort(key=lambda x: x["data_date"])
        self._cached_stats = None

    def remove_update(self, index: int):
        if 0 <= index < len(self.updates):
            self.updates.pop(index)
            self._cached_stats = None

    def get_latest(self) -> Optional[Dict]:
        return self.updates[-1] if self.updates else self.baseline

    def get_baseline(self) -> Optional[Dict]:
        return self.baseline

    def get_update_by_date(self, date_str: str) -> Optional[Dict]:
        for u in self.updates:
            if date_str in u["data_date"] or date_str in u["name"]:
                return u
        return None

    def get_update_by_month(self, month: str, year: str = None) -> Optional[Dict]:
        month_map = {
            "jan":"01","january":"01","feb":"02","february":"02","mar":"03","march":"03",
            "apr":"04","april":"04","may":"05","jun":"06","june":"06","jul":"07","july":"07",
            "aug":"08","august":"08","sep":"09","september":"09","oct":"10","october":"10",
            "nov":"11","november":"11","dec":"12","december":"12",
        }
        mn = month_map.get(month.lower(), month.zfill(2))
        for u in self.updates:
            dd = u["data_date"]
            if len(dd) >= 7 and dd[5:7] == mn:
                if year is None or dd[:4] == str(year):
                    return u
        return None

    def _create_dataframes(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """
        Only materialise DataFrames for tables used by the analytics layer.
        Skips bulk unused tables (notably TASKACTV at 191k rows).
        """
        dfs: Dict[str, pd.DataFrame] = {}

        # Convenience lists populated by CompleteXERExtractor
        if data.get("tasks"):
            dfs["tasks"] = pd.DataFrame(data["tasks"])
        if data.get("wbs"):
            dfs["wbs"] = pd.DataFrame(data["wbs"])

        # Raw tables — load only what analytics code touches
        for table_name, records in data.get("tables", {}).items():
            key = table_name.lower()
            if key in _SKIP_TABLES:
                continue
            if key not in _KEEP_TABLES:
                continue
            if records and key not in dfs:
                try:
                    dfs[key] = pd.DataFrame(records)
                except Exception as exc:
                    print(f"[XERDataStore] Could not create DF for {table_name}: {exc}")

        return dfs

    def compute_basic_stats(self) -> Dict:
        if self._cached_stats is not None:
            return self._cached_stats

        source = self.get_latest()
        if not source or "tasks" not in source.get("df", {}):
            return {"error": "No data loaded"}

        tasks_df = source["df"]["tasks"].copy()
        hpd      = self.hours_per_day
        stats:   Dict[str, Any] = {}

        stats["total_activities"] = len(tasks_df)
        stats["data_source"]      = source["name"]
        stats["data_date"]        = source["data_date"]

        if "task_type" in tasks_df.columns:
            tc = tasks_df["task_type"].value_counts().to_dict()
            stats["task_types"]    = tc
            stats["milestones"]    = tc.get("TT_Mile", 0) + tc.get("TT_FinMile", 0)
            stats["loe_activities"]= tc.get("TT_LOE",  0)
            stats["regular_tasks"] = tc.get("TT_Task",  0)
            work_mask = ~tasks_df["task_type"].isin(self.EXCLUDED_TYPES)
        else:
            stats["milestones"] = stats["loe_activities"] = 0
            stats["regular_tasks"] = len(tasks_df)
            work_mask = pd.Series(True, index=tasks_df.index)

        work = tasks_df[work_mask].copy()

        if "status_code" in tasks_df.columns:
            sc = tasks_df["status_code"].value_counts().to_dict()
            stats["status_breakdown"] = sc
            stats["completed"]   = sc.get("TK_Complete", 0)
            stats["in_progress"] = sc.get("TK_Active",   0)
            stats["not_started"] = sc.get("TK_NotStart", 0)
        else:
            stats.update({"completed": 0, "in_progress": 0, "not_started": 0})

        # Duration
        work["_dur_hrs"]  = _num(work, "target_drtn_hr_cnt")
        work["_dur_days"] = work["_dur_hrs"] / hpd
        if len(work) > 0:
            stats["long_duration_count"] = int((work["_dur_days"] > 20).sum())
            stats["avg_duration_days"]   = round(work["_dur_days"].mean(), 1)
            stats["max_duration_days"]   = round(work["_dur_days"].max(),  1)
        else:
            stats.update({"long_duration_count": 0, "avg_duration_days": 0.0, "max_duration_days": 0.0})

        # Float
        work["_float_hrs"]  = _num(work, "total_float_hr_cnt")
        work["_float_days"] = work["_float_hrs"] / hpd
        if len(work) > 0:
            crit_mask = work["_float_hrs"] <= 0
            nc_mask   = (work["_float_hrs"] > 0) & (work["_float_hrs"] <= NEAR_CRITICAL_HRS)
            stats["critical_count"]       = int(crit_mask.sum())
            stats["critical_pct"]         = round(crit_mask.sum() / len(work) * 100, 1)
            stats["near_critical_count"]  = int(nc_mask.sum())
            stats["negative_float_count"] = int((work["_float_hrs"] < 0).sum())
        else:
            stats.update({"critical_count": 0, "critical_pct": 0.0,
                          "near_critical_count": 0, "negative_float_count": 0})

        # Relationships — FIX: open-ended/dangling computed on WORK tasks only
        pred_key = "taskpred"
        if pred_key in source["df"]:
            pred_df = source["df"][pred_key].copy()
            stats["total_relationships"] = len(pred_df)
            if "pred_type" in pred_df.columns:
                stats["relationship_types"] = pred_df["pred_type"].value_counts().to_dict()
            pred_df["_lag"] = _num(pred_df, "lag_hr_cnt")
            stats["relationships_with_lag"] = int((pred_df["_lag"] != 0).sum())
            stats["negative_lags"]          = int((pred_df["_lag"] <  0).sum())

            # Only consider relationships where at least one endpoint is a work task
            work_ids = set(work["task_id"].dropna().tolist()) if "task_id" in work.columns else set()
            pred_work = pred_df[
                pred_df["task_id"].isin(work_ids) | pred_df["pred_task_id"].isin(work_ids)
            ] if "task_id" in pred_df.columns and "pred_task_id" in pred_df.columns else pred_df

            has_succ = set(_col(pred_work, "pred_task_id").dropna().tolist())
            has_pred = set(_col(pred_work, "task_id").dropna().tolist())

            stats["open_ended_count"] = len(work_ids - has_succ)
            stats["dangling_count"]   = len(work_ids - has_pred)
        else:
            stats.update({"total_relationships": 0, "relationships_with_lag": 0,
                          "negative_lags": 0, "open_ended_count": 0, "dangling_count": 0})

        # Constraints
        if "cstr_type" in tasks_df.columns:
            cstr_mask = tasks_df["cstr_type"].notna() & (tasks_df["cstr_type"] != "")
            stats["constrained_activities"] = int(cstr_mask.sum())
            if cstr_mask.sum() > 0:
                stats["constraint_types"] = tasks_df.loc[cstr_mask, "cstr_type"].value_counts().to_dict()
        else:
            stats["constrained_activities"] = 0

        # Resources
        rsrc_key = "taskrsrc"
        if rsrc_key in source["df"]:
            ra = source["df"][rsrc_key]
            stats["resource_assignments"]  = len(ra)
            tw = _col(ra, "task_id").nunique()
            stats["tasks_with_resources"]  = int(tw)
            stats["resource_loaded_pct"]   = round(tw / max(len(tasks_df), 1) * 100, 1)
        else:
            stats.update({"resource_assignments": 0, "tasks_with_resources": 0, "resource_loaded_pct": 0.0})

        # Overdue
        dd_str = source.get("data_date", "")
        stats.update({"overdue_not_started": 0, "overdue_in_progress": 0})
        if dd_str:
            try:
                dd = pd.to_datetime(dd_str, errors="coerce")
                if pd.notna(dd):
                    if "status_code" in tasks_df.columns and "target_start_date" in tasks_df.columns:
                        ns = tasks_df[tasks_df["status_code"] == "TK_NotStart"].copy()
                        ns["_ts"] = pd.to_datetime(_col(ns, "target_start_date"), errors="coerce")
                        stats["overdue_not_started"] = int((ns["_ts"] < dd).sum())
                    if "status_code" in tasks_df.columns and "target_end_date" in tasks_df.columns:
                        ip = tasks_df[tasks_df["status_code"] == "TK_Active"].copy()
                        ip["_te"] = pd.to_datetime(_col(ip, "target_end_date"), errors="coerce")
                        stats["overdue_in_progress"] = int((ip["_te"] < dd).sum())
            except Exception:
                pass

        # Project date range
        for col, key in [("target_start_date", "project_start"), ("target_end_date", "project_finish")]:
            if col in tasks_df.columns:
                parsed = pd.to_datetime(tasks_df[col], errors="coerce").dropna()
                if not parsed.empty:
                    stats[key] = str((parsed.min() if key == "project_start" else parsed.max()))[:10]

        # Calendars
        if "calendar" in source["df"]:
            cal_df = source["df"]["calendar"]
            stats["calendar_count"] = len(cal_df)
            if "clndr_name" in cal_df.columns:
                stats["calendars"] = cal_df["clndr_name"].dropna().tolist()

        # File info
        stats["baseline_name"]  = self.baseline["name"]      if self.baseline else None
        stats["baseline_date"]  = self.baseline["data_date"] if self.baseline else None
        stats["update_count"]   = len(self.updates)
        stats["updates"]        = [{"name": u["name"], "date": u["data_date"]} for u in self.updates]

        self._cached_stats = stats
        return stats


# ---------------------------------------------------------------------------
# 5. Insights layer
# ---------------------------------------------------------------------------

class InsightsDataLayer:

    def __init__(self, data_store: XERDataStore):
        self.data_store = data_store
        self._cache:    Dict[str, Any] = {}

    def invalidate(self):
        self._cache = {}

    def get_all_insights(self) -> Dict:
        return {
            "delays":        self.get_delay_insights(),
            "critical_path": self.get_critical_path_insights(),
            "resources":     self.get_resource_insights(),
        }

    def get_delay_insights(self) -> Dict:
        key = "delays"
        if key in self._cache:
            return self._cache[key]
        out = {
            "project_end_slippage_days": None,
            "baseline_project_end": None,
            "current_project_end":  None,
            "slipped_activities_count": 0,
            "gained_negative_float_count": 0,
            "delayed_not_started":  [],
            "delayed_in_progress":  [],
            "most_slipped_activities": [],
            "float_trend": [],
        }
        source   = self.data_store.get_latest()
        baseline = self.data_store.get_baseline()
        hpd      = self.data_store.hours_per_day
        if not source or "tasks" not in source.get("df", {}):
            self._cache[key] = out
            return out

        tasks = source["df"]["tasks"].copy()
        if "task_type" in tasks.columns:
            work_mask = ~tasks["task_type"].isin({"TT_LOE", "TT_Mile", "TT_FinMile"})
        else:
            work_mask = pd.Series(True, index=tasks.index)
        work = tasks[work_mask].copy()
        dd   = pd.to_datetime(source.get("data_date", ""), errors="coerce")

        if pd.notna(dd):
            if "status_code" in work.columns and "target_start_date" in work.columns:
                ns = work[work["status_code"] == "TK_NotStart"].copy()
                ns["_ts"] = pd.to_datetime(ns["target_start_date"], errors="coerce")
                ns_over   = ns[ns["_ts"] < dd].copy()
                if not ns_over.empty:
                    ns_over["days_overdue"] = (dd - ns_over["_ts"]).dt.days
                    out["delayed_not_started"] = _safe_records(
                        ns_over.nlargest(20, "days_overdue"),
                        ["task_id","task_code","task_name","target_start_date","days_overdue"],
                    )
            if "status_code" in work.columns and "target_end_date" in work.columns:
                ip = work[work["status_code"] == "TK_Active"].copy()
                ip["_te"] = pd.to_datetime(ip["target_end_date"], errors="coerce")
                ip_over   = ip[ip["_te"] < dd].copy()
                if not ip_over.empty:
                    ip_over["days_overdue"] = (dd - ip_over["_te"]).dt.days
                    out["delayed_in_progress"] = _safe_records(
                        ip_over.nlargest(20, "days_overdue"),
                        ["task_id","task_code","task_name","target_end_date","days_overdue"],
                    )

        if baseline and "tasks" in baseline.get("df", {}):
            bl     = baseline["df"]["tasks"].copy()
            bl_end = pd.to_datetime(_col(bl, "target_end_date"), errors="coerce").max()
            cu_end = pd.to_datetime(_col(tasks, "target_end_date"), errors="coerce").max()
            if pd.notna(bl_end) and pd.notna(cu_end):
                out["project_end_slippage_days"] = int((cu_end - bl_end).days)
                out["baseline_project_end"]      = str(bl_end)[:10]
                out["current_project_end"]       = str(cu_end)[:10]

            bl_slim = pd.DataFrame({"task_id": _col(bl, "task_id")})
            if "target_end_date" in bl.columns:
                bl_slim["bl_end_date"] = bl["target_end_date"].values
            if "total_float_hr_cnt" in bl.columns:
                bl_slim["bl_float_hrs"] = bl["total_float_hr_cnt"].values

            merged = work.merge(bl_slim, on="task_id", how="inner")
            if "bl_end_date" in merged.columns and "target_end_date" in merged.columns:
                merged["_cu_end"] = pd.to_datetime(merged["target_end_date"], errors="coerce")
                merged["_bl_end"] = pd.to_datetime(merged["bl_end_date"],     errors="coerce")
                merged["slippage_days"] = (
                    (merged["_cu_end"] - merged["_bl_end"]).dt.days.fillna(0).astype(int)
                )
                slipped = merged[merged["slippage_days"] > 0]
                out["slipped_activities_count"] = len(slipped)
                if not slipped.empty:
                    out["most_slipped_activities"] = _safe_records(
                        slipped.nlargest(20, "slippage_days"),
                        ["task_id","task_code","task_name","bl_end_date","target_end_date","slippage_days"],
                    )
            if "bl_float_hrs" in merged.columns and "total_float_hr_cnt" in merged.columns:
                merged["_cu_float"] = _num(merged, "total_float_hr_cnt")
                merged["_bl_float"] = pd.to_numeric(merged["bl_float_hrs"], errors="coerce").fillna(0)
                out["gained_negative_float_count"] = int(
                    ((merged["_bl_float"] >= 0) & (merged["_cu_float"] < 0)).sum()
                )

        if self.data_store.updates:
            trend = []
            for upd in self.data_store.updates:
                if "tasks" in upd.get("df", {}):
                    t   = upd["df"]["tasks"]
                    flt = _num(t, "total_float_hr_cnt")
                    trend.append({
                        "update_name":           upd["name"],
                        "data_date":             upd["data_date"],
                        "negative_float_count":  int((flt < 0).sum()),
                        "critical_count":        int((flt <= 0).sum()),
                        "avg_float_days":        round(float(flt.mean()) / hpd, 1),
                    })
            out["float_trend"] = trend

        self._cache[key] = out
        return out

    def get_critical_path_insights(self) -> Dict:
        key = "critical_path"
        if key in self._cache:
            return self._cache[key]
        hpd = self.data_store.hours_per_day
        out = {
            "critical_activities":      [],
            "near_critical_activities": [],
            "critical_by_wbs":          [],
            "critical_path_changes":    [],
            "constrained_critical":     [],
            "critical_with_lags":       [],
            "near_critical_threshold_days": round(NEAR_CRITICAL_HRS / hpd, 1),
            "summary": {},
        }
        source = self.data_store.get_latest()
        if not source or "tasks" not in source.get("df", {}):
            self._cache[key] = out
            return out

        tasks = source["df"]["tasks"].copy()
        work  = tasks[tasks["task_type"] != "TT_LOE"].copy() if "task_type" in tasks.columns else tasks.copy()

        work["_float_hrs"]  = _num(work, "total_float_hr_cnt")
        work["_float_days"] = work["_float_hrs"] / hpd
        work["_dur_hrs"]    = _num(work, "target_drtn_hr_cnt")
        work["_dur_days"]   = work["_dur_hrs"] / hpd
        work["_complete_pct"] = _num(work, "phys_complete_pct")

        critical  = work[work["_float_hrs"] <= 0].copy()
        near_crit = work[
            (work["_float_hrs"] > 0) & (work["_float_hrs"] <= NEAR_CRITICAL_HRS)
        ].copy()

        out["critical_activities"] = _safe_records(
            critical.sort_values("_float_hrs"),
            ["task_id","task_code","task_name","task_type","status_code",
             "_float_days","_dur_days","_complete_pct",
             "target_start_date","target_end_date","wbs_id","cstr_type"],
        )
        out["near_critical_activities"] = _safe_records(
            near_crit.sort_values("_float_hrs"),
            ["task_id","task_code","task_name","status_code",
             "_float_days","_dur_days","target_start_date","target_end_date"],
        )
        out["summary"] = {
            "critical_count":              len(critical),
            "near_critical_count":         len(near_crit),
            "critical_pct":                round(len(critical) / max(len(work), 1) * 100, 1),
            "total_critical_duration_days": round(float(critical["_dur_days"].sum()), 1),
            "avg_critical_float_days":     round(float(critical["_float_hrs"].mean()) / hpd, 2)
                                           if not critical.empty else 0,
        }

        if "wbs_id" in critical.columns and not critical.empty:
            wbs_g = critical.groupby("wbs_id", as_index=False).agg(
                critical_count=("task_id", "count"),
                total_duration_days=("_dur_days", "sum"),
                avg_float_days=("_float_days", "mean"),
            )
            wbs_g["total_duration_days"] = wbs_g["total_duration_days"].round(1)
            wbs_g["avg_float_days"]      = wbs_g["avg_float_days"].round(2)
            for wbs_key in ("projwbs", "wbs"):
                if wbs_key in source["df"]:
                    wdf = source["df"][wbs_key]
                    if "wbs_id" in wdf.columns and "wbs_name" in wdf.columns:
                        wbs_g = wbs_g.merge(
                            wdf[["wbs_id","wbs_name"]].drop_duplicates("wbs_id"),
                            on="wbs_id", how="left",
                        )
                        break
            out["critical_by_wbs"] = wbs_g.nlargest(20, "critical_count").to_dict("records")

        if "cstr_type" in critical.columns:
            constr = critical[
                _col(critical, "cstr_type").notna() & (_col(critical, "cstr_type") != "")
            ]
            out["constrained_critical"] = _safe_records(
                constr, ["task_id","task_code","task_name","cstr_type","cstr_date","_float_days"],
            )

        if "taskpred" in source["df"] and "task_id" in critical.columns:
            pred = source["df"]["taskpred"].copy()
            pred["_lag_hrs"]  = _num(pred, "lag_hr_cnt")
            pred["_lag_days"] = pred["_lag_hrs"] / hpd
            crit_ids  = set(critical["task_id"].tolist())
            lag_ties  = pred[pred["task_id"].isin(crit_ids) & (pred["_lag_hrs"] != 0)].copy()
            out["critical_with_lags"] = _safe_records(
                lag_ties.head(30), ["task_id","pred_task_id","pred_type","_lag_days"],
            )

        baseline = self.data_store.get_baseline()
        if baseline and "tasks" in baseline.get("df", {}):
            bl_tasks = baseline["df"]["tasks"].copy()
            bl_tasks["_bl_float"] = _num(bl_tasks, "total_float_hr_cnt")
            bl_crit_ids = set(
                bl_tasks.loc[bl_tasks["_bl_float"] <= 0, "task_id"].dropna().tolist()
            ) if "task_id" in bl_tasks.columns else set()
            cu_crit_ids = set(critical["task_id"].dropna().tolist()) if "task_id" in critical.columns else set()
            newly = cu_crit_ids - bl_crit_ids
            left  = bl_crit_ids - cu_crit_ids
            changes = []
            if "task_id" in work.columns:
                for tid, label in (
                    [(tid, "became_critical") for tid in list(newly)[:20]]
                    + [(tid, "left_critical")  for tid in list(left)[:20]]
                ):
                    row = work[work["task_id"] == tid]
                    if not row.empty:
                        r = row.iloc[0]
                        changes.append({
                            "task_id":            tid,
                            "task_code":          str(r.get("task_code", "")),
                            "task_name":          str(r.get("task_name", "")),
                            "change":             label,
                            "current_float_days": round(float(r.get("_float_days", 0)), 2),
                        })
            out["critical_path_changes"] = changes

        self._cache[key] = out
        return out

    def get_resource_insights(self) -> Dict:
        key = "resources"
        if key in self._cache:
            return self._cache[key]
        hpd = self.data_store.hours_per_day
        out = {
            "cost_summary":            {},
            "unresourced_count":       0,
            "unresourced_activities":  [],
            "top_resources_by_cost":   [],
            "overbudget_activities":   [],
            "resource_type_breakdown": {},
            "cost_by_wbs":             [],
        }
        source = self.data_store.get_latest()
        if not source:
            self._cache[key] = out
            return out

        tasks_df    = source["df"].get("tasks")
        rsrc_assign = source["df"].get("taskrsrc")
        rsrc_def    = source["df"].get("rsrc")

        if tasks_df is None:
            self._cache[key] = out
            return out

        work_tasks = (
            tasks_df[~tasks_df["task_type"].isin({"TT_LOE","TT_Mile","TT_FinMile"})].copy()
            if "task_type" in tasks_df.columns else tasks_df.copy()
        )

        if rsrc_assign is not None and "task_id" in rsrc_assign.columns and "task_id" in work_tasks.columns:
            resourced_ids = set(_col(rsrc_assign, "task_id").dropna().tolist())
            unres = work_tasks[~work_tasks["task_id"].isin(resourced_ids)].copy()
            out["unresourced_count"] = len(unres)
            if not unres.empty:
                unres["_dur_hrs"]  = _num(unres, "target_drtn_hr_cnt")
                unres["_dur_days"] = unres["_dur_hrs"] / hpd
                unres["_float_hrs"]= _num(unres, "total_float_hr_cnt")
                out["unresourced_activities"] = _safe_records(
                    unres.nlargest(20, "_dur_hrs"),
                    ["task_id","task_code","task_name","status_code",
                     "_dur_days","_float_hrs","target_start_date"],
                )

        if rsrc_assign is None:
            self._cache[key] = out
            return out

        ra = rsrc_assign.copy()
        ra["_budget"] = _num(ra, "target_cost")
        ra["_actual"] = _num(ra, "act_reg_cost")
        ra["_remain"] = _num(ra, "remain_cost")

        b = float(ra["_budget"].sum())
        a = float(ra["_actual"].sum())
        r = float(ra["_remain"].sum())
        out["cost_summary"] = {
            "total_budgeted_cost":    round(b, 2),
            "total_actual_cost":      round(a, 2),
            "total_remaining_cost":   round(r, 2),
            "estimate_at_completion": round(a + r, 2),
            "cost_variance":          round(a - b, 2),
            "cost_performance_index": round(b / a, 3) if a > 0 else None,
            "budget_spent_pct":       round(a / b * 100, 1) if b > 0 else None,
        }

        if "rsrc_id" in ra.columns:
            rg = ra.groupby("rsrc_id", as_index=False).agg(
                total_budget=("_budget", "sum"),
                total_actual=("_actual", "sum"),
                total_remain=("_remain", "sum"),
                assignments =("task_id", "count"),
            )
            if rsrc_def is not None and "rsrc_id" in rsrc_def.columns:
                keep = ["rsrc_id"] + [c for c in ("rsrc_name","rsrc_short_name","rsrc_type") if c in rsrc_def.columns]
                rg = rg.merge(rsrc_def[keep].drop_duplicates("rsrc_id"), on="rsrc_id", how="left")
            out["top_resources_by_cost"] = rg.nlargest(20, "total_budget").to_dict("records")

        self._cache[key] = out
        return out


# ---------------------------------------------------------------------------
# 6. Code executor
# ---------------------------------------------------------------------------

class XERQueryExecutor:
    def __init__(self, data_store: XERDataStore, insights_layer=None):
        self.data_store      = data_store
        self._insights_layer = insights_layer

    def set_insights(self, layer):
        self._insights_layer = layer

    def execute(self, code: str) -> Dict[str, Any]:
        try:
            insights_snapshot = {}
            if self._insights_layer:
                try:
                    insights_snapshot = self._insights_layer.get_all_insights()
                except Exception:
                    pass

            ctx = {
                "pd":                   pd,
                "datetime":             datetime,
                "json":                 json,
                "baseline":             self.data_store.get_baseline(),
                "updates":              self.data_store.updates,
                "latest":               self.data_store.get_latest(),
                "get_latest":           self.data_store.get_latest,
                "get_baseline":         self.data_store.get_baseline,
                "get_update_by_date":   self.data_store.get_update_by_date,
                "get_update_by_month":  self.data_store.get_update_by_month,
                "hours_per_day":        self.data_store.hours_per_day,
                "insights":             insights_snapshot,
                "result":               None,
            }
            exec(code, ctx)
            result = ctx.get("result")
            if isinstance(result, pd.DataFrame):
                result = result.head(50).to_dict("records")
            return {"success": True, "result": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 7. Code generation templates
# ---------------------------------------------------------------------------

INTENT_CODE_TEMPLATES: Dict[str, str] = {
    "critical_path": """
tasks = latest['df']['tasks'].copy()
work  = tasks[~tasks['task_type'].isin(['TT_LOE','TT_Mile','TT_FinMile'])].copy()
work['float_hrs'] = pd.to_numeric(work['total_float_hr_cnt'], errors='coerce').fillna(0)
work['float_days']= work['float_hrs'] / hours_per_day
work['dur_hrs']   = pd.to_numeric(work['target_drtn_hr_cnt'], errors='coerce').fillna(0)
work['dur_days']  = work['dur_hrs'] / hours_per_day
critical = work[work['float_hrs'] <= 0].sort_values(['float_hrs','dur_hrs'], ascending=[True,False])
result = critical.head(50)[[c for c in ['task_code','task_name','task_type','status_code',
    'float_days','dur_days','target_start_date','target_end_date','wbs_id','cstr_type']
    if c in critical.columns]].fillna('').to_dict('records')
""",
    "duration": """
tasks = latest['df']['tasks'].copy()
work  = tasks[~tasks['task_type'].isin(['TT_LOE','TT_Mile','TT_FinMile'])].copy()
work['dur_hrs']   = pd.to_numeric(work['target_drtn_hr_cnt'], errors='coerce').fillna(0)
work['dur_days']  = work['dur_hrs'] / hours_per_day
work['float_hrs'] = pd.to_numeric(work['total_float_hr_cnt'], errors='coerce').fillna(0)
long = work[work['dur_days'] > 20].sort_values('dur_days', ascending=False).head(50)
result = long[[c for c in ['task_code','task_name','dur_days','float_hrs',
    'status_code','target_start_date','target_end_date','wbs_id']
    if c in long.columns]].fillna('').to_dict('records')
""",
    "quality": """
tasks   = latest['df']['tasks'].copy()
work    = tasks[~tasks['task_type'].isin(['TT_LOE','TT_Mile','TT_FinMile'])].copy()
preds   = latest['df'].get('taskpred', pd.DataFrame())
work['float_hrs'] = pd.to_numeric(work.get('total_float_hr_cnt', 0), errors='coerce').fillna(0)
work['dur_days']  = pd.to_numeric(work.get('target_drtn_hr_cnt',  0), errors='coerce').fillna(0) / hours_per_day
total = len(work)
work_ids  = set(work['task_id'].tolist()) if 'task_id' in work.columns else set()
has_succ  = set(preds['pred_task_id'].tolist()) if not preds.empty and 'pred_task_id' in preds.columns else set()
has_pred  = set(preds['task_id'].tolist())      if not preds.empty and 'task_id'      in preds.columns else set()
lag_col   = pd.to_numeric(preds.get('lag_hr_cnt', pd.Series([])), errors='coerce').fillna(0) if not preds.empty else pd.Series([])
result = {
    'total_work_activities': total,
    'open_ended':            len(work_ids - has_succ),
    'open_ended_pct':        round(len(work_ids - has_succ) / max(total,1)*100, 2),
    'dangling':              len(work_ids - has_pred),
    'dangling_pct':          round(len(work_ids - has_pred) / max(total,1)*100, 2),
    'long_duration_gt20d':   int((work['dur_days'] > 20).sum()),
    'long_dur_pct':          round((work['dur_days'] > 20).sum() / max(total,1)*100, 2),
    'negative_float':        int((work['float_hrs'] < 0).sum()),
    'total_relationships':   len(preds),
    'negative_lags':         int((lag_col < 0).sum()),
    'positive_lags':         int((lag_col > 0).sum()),
    'lag_pct':               round((lag_col != 0).sum() / max(len(preds),1)*100, 2),
}
""",
    "milestones": """
tasks = latest['df']['tasks'].copy()
ms    = tasks[tasks['task_type'].isin(['TT_Mile','TT_FinMile'])].copy() if 'task_type' in tasks.columns else pd.DataFrame()
if not ms.empty:
    ms['float_days'] = pd.to_numeric(ms['total_float_hr_cnt'], errors='coerce').fillna(0) / hours_per_day
    ms = ms.sort_values('target_end_date')
    result = ms[[c for c in ['task_code','task_name','task_type','status_code',
        'float_days','target_end_date','wbs_id','cstr_type'] if c in ms.columns]].fillna('').to_dict('records')
else:
    result = []
""",
    "resources": """
ra = latest['df'].get('taskrsrc', pd.DataFrame()).copy()
if not ra.empty:
    for col in ['target_cost','act_reg_cost','remain_cost']:
        ra[col] = pd.to_numeric(ra.get(col, 0), errors='coerce').fillna(0)
    tasks = latest['df']['tasks'].copy()
    work  = tasks[~tasks['task_type'].isin(['TT_LOE','TT_Mile','TT_FinMile'])] if 'task_type' in tasks.columns else tasks
    resourced = set(ra.get('task_id', pd.Series()).tolist())
    unresourced = work[~work['task_id'].isin(resourced)] if 'task_id' in work.columns else pd.DataFrame()
    result = {
        'total_budget':        round(float(ra['target_cost'].sum()),  2),
        'total_actual':        round(float(ra['act_reg_cost'].sum()), 2),
        'total_remain':        round(float(ra['remain_cost'].sum()),  2),
        'resource_count':      ra.get('rsrc_id', pd.Series()).nunique(),
        'task_count':          ra.get('task_id', pd.Series()).nunique(),
        'unresourced_count':   len(unresourced),
        'resource_loaded_pct': round(len(resourced) / max(len(work),1)*100, 1),
    }
else:
    result = {'message': 'No resource assignment data found'}
""",
}


# ---------------------------------------------------------------------------
# 8. Main XERAnalyzer
# ---------------------------------------------------------------------------

class XERAnalyzer:
    """Main analyzer — improved LLM pipeline with intent routing."""

    def __init__(self):
        self.data_store = XERDataStore()
        self.insights   = InsightsDataLayer(self.data_store)
        self.executor   = XERQueryExecutor(self.data_store, self.insights)
        self._context_builder = ContextBuilder()

        # XERContextBuilder — load with explicit warning on failure
        try:
            from xer_context_builder import XERContextBuilder
            self._ctx_builder = XERContextBuilder(
                self.data_store,
                hours_per_day=self.data_store.hours_per_day,
            )
            print("[XERAnalyzer] XERContextBuilder loaded — real data context active.")
        except ImportError as exc:
            warnings.warn(
                f"[XERAnalyzer] xer_context_builder.py import failed: {exc}\n"
                "Falling back to aggregate-stats context. "
                "LLM responses WILL hallucinate activity codes. "
                "Ensure xer_context_builder.py is in the same directory as xer_analyzer.py.",
                stacklevel=2,
            )
            self._ctx_builder = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_baseline(self, data: Dict, name: str = None, data_date: str = None):
        if name is None:
            name = data.get("project", {}).get("project_name", "Baseline")
        if data_date is None:
            raw = data.get("project", {}).get("data_date", "")
            data_date = str(raw)[:10] if raw else ""
        self.data_store.load_baseline(data, name, data_date)
        self.insights.invalidate()

    def add_update(self, data: Dict, name: str = None, data_date: str = None):
        if name is None:
            name = data.get("project", {}).get("project_name", "Update")
        if data_date is None:
            raw = data.get("project", {}).get("data_date", "")
            data_date = str(raw)[:10] if raw else ""
        self.data_store.add_update(data, name, data_date)
        self.insights.invalidate()

    def remove_update(self, index: int):
        self.data_store.remove_update(index)
        self.insights.invalidate()

    # ------------------------------------------------------------------
    # Stats / insights shortcuts
    # ------------------------------------------------------------------

    def get_basic_stats(self)          -> Dict: return self.data_store.compute_basic_stats()
    def get_all_insights(self)         -> Dict: return self.insights.get_all_insights()
    def get_delay_insights(self)       -> Dict: return self.insights.get_delay_insights()
    def get_critical_path_insights(self) -> Dict: return self.insights.get_critical_path_insights()
    def get_resource_insights(self)    -> Dict: return self.insights.get_resource_insights()
    def execute_code(self, code: str)  -> Dict: return self.executor.execute(code)

    # ------------------------------------------------------------------
    # LLM prompt builders
    # ------------------------------------------------------------------

    def get_system_prompt(self, intent: str = "general") -> str:
        """Return the strict anti-hallucination system prompt for the given intent."""
        return SystemPromptLibrary.get(intent)

    def get_code_generation_prompt(self, user_query: str, basic_stats: Dict) -> str:
        source      = self.data_store.get_latest()
        primary     = IntentClassifier.primary_intent(user_query)
        template    = INTENT_CODE_TEMPLATES.get(primary, "")
        columns_info = ""
        if source and "df" in source:
            for table_name, df in list(source["df"].items())[:10]:
                cols = [c for c in df.columns if not c.startswith("_")][:20]
                columns_info += f"\n  {table_name.upper()}: {', '.join(cols)}"
        hint_block = (
            f"\nRECOMMENDED PATTERN FOR {primary.upper()}:\n{template}\n"
            "(Adapt as needed.)"
        ) if template else ""
        return f"""Generate Python code to answer this schedule question.

USER QUESTION: {user_query}
DETECTED INTENT: {primary}

AVAILABLE TABLES AND COLUMNS:{columns_info}

AVAILABLE VARIABLES:
- latest['df']['tasks']       — all activities (DataFrame)
- latest['df']['taskpred']    — predecessor relationships
- latest['df']['taskrsrc']    — resource assignments
- latest['df']['projwbs']     — WBS structure
- baseline                    — baseline data dict
- updates                     — list of update dicts sorted by data_date
- hours_per_day               — {self.data_store.hours_per_day}
- insights                    — pre-computed insights dict

KEY FIELDS:
  task_code, task_name, task_type (TT_Task/TT_Mile/TT_FinMile/TT_LOE)
  status_code (TK_NotStart/TK_Active/TK_Complete)
  total_float_hr_cnt   (HOURS — divide by hours_per_day for days)
  target_drtn_hr_cnt   (HOURS)
  target_start_date, target_end_date, act_start_date, act_end_date
  wbs_id, cstr_type, cstr_date, lag_hr_cnt, pred_type

RULES:
1. pd.to_numeric(df['col'], errors='coerce').fillna(0) before ALL arithmetic
2. .copy() before modifying any DataFrame
3. Exclude TT_LOE from quality/float/duration analyses
4. Final line MUST be: result = <dict|list|scalar>  (must be JSON-serialisable)
5. if 'col' in df.columns before accessing any column
6. Limit lists to 50 items max
{hint_block}

Return ONLY valid Python. No markdown fences. No explanations."""

    def get_response_prompt(
        self,
        user_query:   str,
        basic_stats:  Dict,
        code_result:  Any,
        code_success: bool,
        code_error:   Optional[str] = None,
    ) -> str:
        """
        Build the user-turn prompt for the response LLM.

        Priority:
          1. XERContextBuilder  — actual DataFrame rows (prevents hallucination)
          2. Legacy ContextBuilder — aggregate stats (hallucination risk, flagged)
        """
        # ── Attempt real-data context ──────────────────────────────────────────
        context = None
        if self._ctx_builder is not None:
            try:
                context = self._ctx_builder.build_context(user_query)
                # Sanity-check: context must be substantive
                if not context or len(context.strip()) < 300:
                    warnings.warn(
                        "[XERAnalyzer] Context builder returned very short context "
                        f"({len(context.strip() if context else '')} chars). "
                        "Falling back to legacy context."
                    )
                    context = None
            except Exception as exc:
                warnings.warn(f"[XERAnalyzer] Context builder raised: {exc}")
                context = None

        # ── Fallback: legacy aggregate context ─────────────────────────────────
        if context is None:
            intents  = IntentClassifier.classify(user_query)
            insights = self.insights.get_all_insights()
            context  = self._context_builder.build(
                query=user_query,
                intents=intents,
                basic_stats=basic_stats,
                insights=insights,
                code_result=code_result if code_success else None,
                code_error=code_error if not code_success else None,
            )
            # Alert the LLM that it has no actual rows
            context = (
                "⚠️ CONTEXT WARNING: This section contains aggregate statistics only — "
                "NO individual activity rows are available. "
                "Do NOT reference specific task codes in your answer; state counts only.\n\n"
                + context
            )

        # ── Append code execution result ───────────────────────────────────────
        code_block = ""
        if code_success and code_result is not None:
            try:
                result_text = json.dumps(code_result, indent=2, default=str)
            except Exception:
                result_text = str(code_result)
            code_block = (
                "\n\n---\n## COMPUTED QUERY RESULT\n"
                f"```json\n{result_text[:4000]}\n```\n"
                "Use these values in your answer. "
                "Do not invent additional rows beyond what is shown here."
            )
        elif code_error:
            code_block = (
                f"\n\n⚠️ Dynamic computation failed: {code_error}. "
                "Answer from the data tables above only."
            )

        return (
            f"{context}"
            f"{code_block}"
            "\n\n---\n"
            f"**QUESTION:** {user_query}\n\n"
            "Answer using ONLY the data above. "
            "If specific activity codes are requested but only counts are available, "
            "state the count and note that the full list was not included in the context."
        )

    def _build_targeted_context(self, query: str) -> str:
        """Emergency fallback context — used in hard error path."""
        if self._ctx_builder is not None:
            try:
                return self._ctx_builder.build_context(query)
            except Exception as exc:
                return f"Context unavailable: {exc}"
        return f"Basic stats: {self.get_basic_stats()}"

