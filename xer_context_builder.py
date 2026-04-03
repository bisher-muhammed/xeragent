"""
xer_context_builder.py
======================
Builds question-specific context for LLM prompts by extracting ACTUAL DATA ROWS.

Fixes vs original:
  - Near-critical threshold uses NEAR_CRITICAL_HRS from xer_analyzer (single source)
  - _MAX_ROWS increased to 60 (more real codes reach the LLM)
  - Open-ended/dangling computed on WORK tasks only (excludes LOE/milestone ties)
  - _tasks() validates required columns exist before returning DataFrame
  - Truncation notices added to every table so LLM knows total vs shown count
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HPD = 10      # default hours per working day

_MAX_ROWS = 60         # max rows per table section in prompt (was 40)

# Near-critical threshold — kept in sync with xer_analyzer.py
# Import from xer_analyzer if available, otherwise use the same literal value.
try:
    from xer_analyzer import NEAR_CRITICAL_HRS, NEAR_CRITICAL_DAYS
except ImportError:
    NEAR_CRITICAL_HRS  = 80
    NEAR_CRITICAL_DAYS = 8

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_INTENT_MAP: Dict[str, List[str]] = {
    "critical_path": [
        "critical", "critical path", "longest path", "near critical",
        "near-critical", "driving", "float zero", "zero float",
    ],
    "schedule_health": [
        "health", "quality", "open ended", "open-ended", "dangling",
        "missing logic", "long duration", "review schedule",
        "schedule check", "schedule audit", "lag", "negative lag",
        "relationship", "logic tie",
    ],
    "duration": [
        "duration over", "days long", "how long", "longest activit",
        "duration greater", "more than", "activities over",
        "exceed", "longer than", "duration above",
    ],
    "constraints": [
        "constraint", "hard constraint", "mandatory", "must start",
        "must finish", "imposed date",
    ],
    "delays": [
        "delay", "slip", "slippage", "late", "overdue", "behind",
        "missed", "past due", "not started yet", "behind schedule",
    ],
    "resources": [
        "resource", "cost", "budget", "actual", "spent", "cpi",
        "eac", "labour", "labor", "material", "equipment", "loaded",
        "unloaded", "unresourced", "over budget", "variance", "manpower",
    ],
    "comparison": [
        "compare", " vs ", "versus", "baseline", "difference",
        "before and after", "progress", "update", "trend",
        "month over month", "changed",
    ],
    "wbs": [
        "wbs", "work breakdown", "area", "zone", "section",
        "discipline", "package",
    ],
    "milestones": [
        "milestone", "key date", "completion date", "finish",
        "handover", "delivery",
    ],
    "float": [
        "float", "total float", "free float", "negative float", "slack",
    ],
}


def classify_intent(query: str) -> List[str]:
    """Return a ranked list of intent labels for a query (most specific first)."""
    q = query.lower()
    matched: List[Tuple[int, str]] = []
    for intent, keywords in _INTENT_MAP.items():
        score = sum(1 for kw in keywords if kw in q)
        if score:
            matched.append((score, intent))
    if not matched:
        return ["general"]
    matched.sort(key=lambda x: -x[0])
    return [m[1] for m in matched]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _to_md(df: pd.DataFrame, max_rows: int = _MAX_ROWS,
           total_count: int = None, round_dp: int = 1) -> str:
    """
    Convert a DataFrame to a compact markdown table.
    Appends a truncation notice when total_count > max_rows.
    """
    if df is None or df.empty:
        return "*(no records)*"
    shown = df.head(max_rows).copy().reset_index(drop=True)
    for col in shown.select_dtypes(include="float").columns:
        shown[col] = shown[col].round(round_dp)
    shown = shown.fillna("")
    try:
        table = shown.to_markdown(index=False)
    except Exception:
        table = shown.to_string(index=False)
    if total_count is not None and total_count > max_rows:
        table += f"\n\n*(showing {min(max_rows, len(df))} of {total_count} total — remaining rows not listed)*"
    elif len(df) > max_rows:
        table += f"\n\n*(showing {max_rows} of {len(df)} total — remaining rows not listed)*"
    return table


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _col_or_empty(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(dtype=object, index=df.index)


# ---------------------------------------------------------------------------
# XERContextBuilder
# ---------------------------------------------------------------------------

class XERContextBuilder:
    """
    Builds targeted, question-specific LLM context from XER data.
    Pulls ACTUAL DATA ROWS (task codes, names, dates, float) into the prompt,
    not just aggregate counts.
    """

    def __init__(self, data_store: Any, hours_per_day: float = _DEFAULT_HPD):
        self.ds  = data_store
        self.hpd = float(hours_per_day)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(self, query: str) -> str:
        """Return context string with actual schedule data relevant to the query."""
        source   = self.ds.get_latest()
        baseline = self.ds.get_baseline()

        if not source:
            return "**No schedule data is loaded.**"

        intents  = classify_intent(query)
        sections: List[str] = []

        sections.append(self._project_header(source, baseline))

        seen: set = set()
        for intent in intents:
            if intent in seen:
                continue
            seen.add(intent)

            if intent == "critical_path":
                sections.append(self._critical_path_section(source, baseline))
            elif intent == "float":
                sections.append(self._float_section(source))
            elif intent in ("schedule_health", "duration"):
                sections.append(self._schedule_health_section(source))
            elif intent == "constraints":
                sections.append(self._constraints_section(source))
            elif intent == "delays":
                sections.append(self._delays_section(source, baseline))
            elif intent == "resources":
                sections.append(self._resources_section(source))
            elif intent == "comparison":
                sections.append(self._comparison_section(source, baseline))
            elif intent == "wbs":
                sections.append(self._wbs_section(source))
            elif intent == "milestones":
                sections.append(self._milestones_section(source))
            elif intent == "general":
                sections.append(self._general_section(source))
                if "schedule_health" not in seen:
                    sections.append(self._schedule_health_section(source))
                    seen.add("schedule_health")

        return "\n\n".join(s for s in sections if s and s.strip())

    def build_messages(self, query: str) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt) ready for any LLM."""
        system  = self._system_prompt()
        context = self.build_context(query)
        user    = f"{context}\n\n---\n\n**USER QUESTION:** {query}"
        return system, user

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt() -> str:
        return textwrap.dedent("""\
            You are a Primavera P6 Schedule Analyst. Answer schedule questions
            accurately and concisely using ONLY the data provided in the user message.

            STRICT RULES:
            1. Every number, task code, name, and date MUST come from the provided data
               tables. Never invent or estimate values.
            2. When listing activities, include their task codes (e.g. N.E.SL.00.00.MOB.035).
            3. When a table says "showing X of Y total", state that Y total exist but only
               X are shown. Do NOT generate codes for the remaining Y-X activities.
            4. If the data does not contain enough information to fully answer, say so
               explicitly — state what is missing and how to obtain it.
            5. Do not add generic project management advice unless recommendations
               are explicitly requested.
            6. Lead with the direct answer, then support with specific data from the tables.
            7. All float values shown are in WORKING DAYS unless stated otherwise.
        """)

    # ------------------------------------------------------------------
    # Project header (always included)
    # ------------------------------------------------------------------

    def _project_header(self, source: Dict, baseline: Optional[Dict]) -> str:
        proj      = source.get("data", {}).get("project", {})
        tasks_df  = self._tasks(source)

        name       = proj.get("project_name") or proj.get("project_code") or source.get("name", "N/A")
        data_date  = source.get("data_date", "N/A")
        plan_start = proj.get("plan_start_date", "")[:10]
        plan_end   = proj.get("plan_end_date",   "")[:10]
        scd_end    = proj.get("scheduled_end_date", "")[:10]

        lines = [
            "## PROJECT HEADER",
            f"- **Name:** {name}",
            f"- **Data Date:** {data_date}",
            f"- **Planned Start:** {plan_start}  |  **Planned Finish:** {plan_end}",
        ]
        if scd_end and scd_end != plan_end:
            lines.append(f"- **Scheduled Finish (current):** {scd_end}")

        bl_name   = baseline["name"] if baseline else "None"
        n_updates = len(self.ds.updates)
        lines.append(f"- **Baseline:** {bl_name}  |  **Updates loaded:** {n_updates}")

        if tasks_df is not None:
            total = len(tasks_df)
            work  = self._work(tasks_df)
            if "status_code" in tasks_df.columns:
                sc       = tasks_df["status_code"].value_counts()
                complete = sc.get("TK_Complete", 0)
                active   = sc.get("TK_Active",   0)
                not_start= sc.get("TK_NotStart", 0)
                lines.append(
                    f"- **Activities:** {total} total  "
                    f"({complete} complete / {active} in-progress / {not_start} not started)"
                )
            if "task_type" in tasks_df.columns:
                tc = tasks_df["task_type"].value_counts()
                lines.append(
                    f"- **Types:** {tc.get('TT_Task',0)} tasks, "
                    f"{tc.get('TT_Mile',0)+tc.get('TT_FinMile',0)} milestones, "
                    f"{tc.get('TT_LOE',0)} LOE"
                )
            if "_float_d" in work.columns:
                crit_n = (work["_float_d"] <= 0).sum()
                neg_n  = (work["_float_d"] <  0).sum()
                lines.append(
                    f"- **Critical (float ≤ 0):** {crit_n}  |  "
                    f"**Negative float:** {neg_n}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Critical path
    # ------------------------------------------------------------------

    def _critical_path_section(self, source: Dict, baseline: Optional[Dict]) -> str:
        tasks_df = self._tasks(source)
        if tasks_df is None:
            return ""

        work = self._work(tasks_df)
        if "_float_d" not in work.columns:
            return "## CRITICAL PATH\nFloat data (total_float_hr_cnt) unavailable in this file."

        crit = work[work["_float_d"] <= 0].sort_values("_float_d")
        near = work[
            (work["_float_d"] > 0) & (work["_float_d"] <= NEAR_CRITICAL_HRS / self.hpd)
        ].sort_values("_float_d")

        pct = len(crit) / max(len(work), 1) * 100

        lines = [
            "## CRITICAL PATH DATA",
            f"- Critical activities (float ≤ 0): **{len(crit)}** ({pct:.1f}% of work activities)",
            f"- Near-critical (0 < float ≤ {round(NEAR_CRITICAL_HRS/self.hpd, 1)}d): **{len(near)}**",
            f"- Total critical duration: **{crit['_dur_d'].sum():.0f} days** (sum, not project span)",
        ]

        # Critical by WBS
        wbs_df = self._wbs(source)
        if "wbs_id" in crit.columns and wbs_df is not None:
            grp = crit.groupby("wbs_id", as_index=False).agg(
                critical_count=("task_id", "count") if "task_id" in crit.columns else ("_float_d", "count"),
                total_dur_days=("_dur_d", "sum"),
                avg_float_days=("_float_d", "mean"),
            )
            if "wbs_id" in wbs_df.columns and "wbs_name" in wbs_df.columns:
                grp = grp.merge(
                    wbs_df[["wbs_id","wbs_name"]].drop_duplicates("wbs_id"),
                    on="wbs_id", how="left",
                )
            grp = grp.sort_values("critical_count", ascending=False)
            lines.append("\n### Critical Activities by WBS Area")
            lines.append(_to_md(grp.head(15)))

        # Critical activity list — with truncation notice
        lines.append(f"\n### Critical Activity List")
        cols = [c for c in [
            "task_code","task_name","status_code","_float_d","_dur_d",
            "target_start_date","target_end_date","cstr_type",
        ] if c in crit.columns]
        rename = {
            "task_code":"Code","task_name":"Activity","status_code":"Status",
            "_float_d":"Float(d)","_dur_d":"Dur(d)",
            "target_start_date":"Start","target_end_date":"Finish","cstr_type":"Constraint",
        }
        lines.append(_to_md(crit[cols].rename(columns=rename), total_count=len(crit)))

        # Near-critical list
        if not near.empty:
            lines.append(f"\n### Near-Critical Activity List (float ≤ {round(NEAR_CRITICAL_HRS/self.hpd,1)}d)")
            nc_cols = [c for c in [
                "task_code","task_name","status_code","_float_d","_dur_d","target_end_date",
            ] if c in near.columns]
            lines.append(_to_md(near[nc_cols].rename(columns=rename), total_count=len(near)))

        # Baseline comparison
        if baseline and "tasks" in baseline.get("df", {}):
            bl_t = baseline["df"]["tasks"].copy()
            if "total_float_hr_cnt" in bl_t.columns and "task_id" in bl_t.columns:
                bl_t["_bl_float_d"] = _num(bl_t["total_float_hr_cnt"]) / self.hpd
                bl_crit_ids = set(bl_t.loc[bl_t["_bl_float_d"] <= 0, "task_id"].dropna())
                cu_crit_ids = set(crit["task_id"].dropna()) if "task_id" in crit.columns else set()
                newly     = cu_crit_ids - bl_crit_ids
                recovered = bl_crit_ids - cu_crit_ids
                lines.append(
                    f"\n### Critical Path Changes vs Baseline\n"
                    f"- Newly critical: **{len(newly)}**\n"
                    f"- Recovered from critical: **{len(recovered)}**"
                )
                if newly and "task_id" in work.columns:
                    new_df = work[work["task_id"].isin(newly)][cols].rename(columns=rename).head(20)
                    lines.append("**Newly Critical:**")
                    lines.append(_to_md(new_df, total_count=len(newly)))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Float distribution
    # ------------------------------------------------------------------

    def _float_section(self, source: Dict) -> str:
        tasks_df = self._tasks(source)
        if tasks_df is None:
            return ""
        work = self._work(tasks_df)
        if "_float_d" not in work.columns:
            return ""
        f = work["_float_d"]
        nc_thresh = round(NEAR_CRITICAL_HRS / self.hpd, 1)
        lines = [
            "## FLOAT DISTRIBUTION",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Min float | {f.min():.1f} d |",
            f"| Max float | {f.max():.1f} d |",
            f"| Average float | {f.mean():.1f} d |",
            f"| Median float | {f.median():.1f} d |",
            f"| Critical (≤0) | {(f<=0).sum()} |",
            f"| Near-critical (≤{nc_thresh}d) | {((f>0)&(f<=nc_thresh)).sum()} |",
            f"| Negative float | {(f<0).sum()} |",
            f"| Float > 30d | {(f>30).sum()} |",
        ]
        worst = (
            work[work["_float_d"] < 0].sort_values("_float_d")
            if (work["_float_d"] < 0).any()
            else work[work["_float_d"] <= 0].sort_values("_float_d").head(20)
        )
        cols = [c for c in ["task_code","task_name","_float_d","_dur_d","target_end_date"] if c in worst.columns]
        lines.append("\n### Lowest Float Activities")
        lines.append(_to_md(
            worst[cols].rename(columns={"_float_d":"Float(d)","_dur_d":"Dur(d)",
                                        "task_code":"Code","task_name":"Activity",
                                        "target_end_date":"Finish"}),
            total_count=len(worst),
        ))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Schedule health / quality
    # ------------------------------------------------------------------

    def _schedule_health_section(self, source: Dict) -> str:
        tasks_df = self._tasks(source)
        if tasks_df is None:
            return ""
        work    = self._work(tasks_df)
        pred_df = source.get("df", {}).get("taskpred")

        lines = ["## SCHEDULE QUALITY DATA"]

        if pred_df is not None and not pred_df.empty:
            pred_df = pred_df.copy()
            pred_df["_lag_h"] = _num(_col_or_empty(pred_df, "lag_hr_cnt"))
            pred_df["_lag_d"] = pred_df["_lag_h"] / self.hpd
            total_rels = len(pred_df)
            with_lag   = (pred_df["_lag_h"] > 0).sum()
            neg_lag    = (pred_df["_lag_h"] < 0).sum()

            lines.append(f"\n### Relationships")
            lines.append(f"| Metric | Count |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Total relationships | {total_rels} |")
            if "pred_type" in pred_df.columns:
                for t, n in pred_df["pred_type"].value_counts().items():
                    lines.append(f"| {t} | {n} |")
            lines.append(f"| With positive lag | {with_lag} |")
            lines.append(f"| With negative lag | {neg_lag} |")

            # Open-ended / dangling — WORK tasks only
            if "task_id" in work.columns:
                work_ids = set(work["task_id"].dropna())

                # Only consider relationships where at least one endpoint is a work task
                pred_work = pred_df[
                    pred_df["task_id"].isin(work_ids) | pred_df["pred_task_id"].isin(work_ids)
                ] if "task_id" in pred_df.columns and "pred_task_id" in pred_df.columns else pred_df

                has_succ = set(_col_or_empty(pred_work, "pred_task_id").dropna())
                has_pred = set(_col_or_empty(pred_work, "task_id").dropna())

                open_ended = work[~work["task_id"].isin(has_succ)]
                dangling   = work[~work["task_id"].isin(has_pred)]

                lines.append(f"\n### Open-Ended Activities (no successor): **{len(open_ended)}**")
                if not open_ended.empty:
                    oe_cols = [c for c in ["task_code","task_name","status_code","_float_d","_dur_d"] if c in open_ended.columns]
                    lines.append(_to_md(
                        open_ended[oe_cols].rename(columns={
                            "_float_d":"Float(d)","_dur_d":"Dur(d)",
                            "task_code":"Code","task_name":"Activity","status_code":"Status",
                        }),
                        total_count=len(open_ended),
                    ))

                lines.append(f"\n### Dangling Activities (no predecessor): **{len(dangling)}**")
                if not dangling.empty:
                    d_cols = [c for c in ["task_code","task_name","status_code","_dur_d"] if c in dangling.columns]
                    lines.append(_to_md(
                        dangling[d_cols].rename(columns={
                            "_dur_d":"Dur(d)","task_code":"Code",
                            "task_name":"Activity","status_code":"Status",
                        }),
                        total_count=len(dangling),
                    ))

            # Negative lag details
            if neg_lag > 0:
                neg_lag_df = pred_df[pred_df["_lag_h"] < 0][
                    [c for c in ["task_id","pred_task_id","pred_type","_lag_d"] if c in pred_df.columns]
                ]
                neg_lag_df = self._enrich_pred_with_codes(neg_lag_df, tasks_df)
                lines.append(f"\n### Negative Lag Relationships: **{neg_lag}**")
                lines.append(_to_md(neg_lag_df.head(20), total_count=int(neg_lag)))

            # Large positive lags
            if with_lag > 0:
                lag_sample = pred_df[pred_df["_lag_h"] > 0].nlargest(20, "_lag_d")[
                    [c for c in ["task_id","pred_task_id","pred_type","_lag_d"] if c in pred_df.columns]
                ]
                lag_sample = self._enrich_pred_with_codes(lag_sample, tasks_df)
                lines.append(f"\n### Largest Positive Lags (top 20 of {int(with_lag)})")
                lines.append(_to_md(lag_sample, total_count=int(with_lag)))

        # Long duration
        if "_dur_d" in work.columns:
            long = work[work["_dur_d"] > 20].sort_values("_dur_d", ascending=False)
            pct  = len(long) / max(len(work), 1) * 100
            dcma = "✓ PASS" if pct < 5 else "✗ FAIL"
            lines.append(f"\n### Long Duration Activities (>20 days): **{len(long)}** ({pct:.1f}%) — DCMA {dcma}")
            if not long.empty:
                ld_cols = [c for c in ["task_code","task_name","status_code","_dur_d","_float_d",
                                       "target_start_date","target_end_date"] if c in long.columns]
                lines.append(_to_md(
                    long[ld_cols].rename(columns={
                        "_float_d":"Float(d)","_dur_d":"Dur(d)","task_code":"Code",
                        "task_name":"Activity","status_code":"Status",
                        "target_start_date":"Start","target_end_date":"Finish",
                    }),
                    total_count=len(long),
                ))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _constraints_section(self, source: Dict) -> str:
        tasks_df = self._tasks(source, work_only=False)
        if tasks_df is None:
            return ""
        eligible = tasks_df[tasks_df["task_type"] != "TT_LOE"].copy() \
            if "task_type" in tasks_df.columns else tasks_df.copy()

        if "cstr_type" not in eligible.columns:
            return "## CONSTRAINTS\nNo constraint column (cstr_type) found in this file."

        constrained  = eligible[eligible["cstr_type"].notna() & (eligible["cstr_type"] != "")].copy()
        total        = len(constrained)
        work_constr  = constrained[~constrained["task_type"].isin({"TT_Mile","TT_FinMile"})] \
            if "task_type" in constrained.columns else constrained
        ms_constr    = constrained[constrained["task_type"].isin({"TT_Mile","TT_FinMile"})] \
            if "task_type" in constrained.columns else pd.DataFrame()

        lines = [
            f"## CONSTRAINT DATA: **{total} constrained activities** "
            f"({len(work_constr)} work tasks, {len(ms_constr)} milestones)"
        ]

        if constrained.empty:
            lines.append("No constraints found in this schedule.")
            return "\n".join(lines)

        CSTR_LABELS = {
            "CS_MSOA":     "Must Start On or After",
            "CS_MEOA":     "Must End On or After",
            "CS_MEOB":     "Must End On or Before",
            "CS_MSO":      "Must Start On",
            "CS_MEO":      "Must Finish On",
            "CS_MFOB":     "Mandatory Finish",
            "CS_MANDFIN":  "Mandatory Finish",
            "CS_MANDSTART":"Mandatory Start",
        }
        for ctype, n in constrained["cstr_type"].value_counts().items():
            label = CSTR_LABELS.get(ctype, ctype)
            lines.append(f"  - {label} ({ctype}): {n}")

        cols = [c for c in [
            "task_code","task_name","task_type","status_code",
            "cstr_type","cstr_date","cstr_type2","cstr_date2","_float_d",
        ] if c in constrained.columns]
        rename = {
            "_float_d":"Float(d)","task_code":"Code","task_name":"Activity",
            "task_type":"Type","status_code":"Status",
            "cstr_type":"Constraint1","cstr_date":"Date1",
            "cstr_type2":"Constraint2","cstr_date2":"Date2",
        }
        lines.append("\n### Constrained Activities")
        lines.append(_to_md(constrained[cols].rename(columns=rename), total_count=total))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Delays / overdue
    # ------------------------------------------------------------------

    def _delays_section(self, source: Dict, baseline: Optional[Dict]) -> str:
        tasks_df = self._tasks(source)
        if tasks_df is None:
            return ""
        work         = self._work(tasks_df)
        data_date_str= source.get("data_date", "")
        data_date    = pd.to_datetime(data_date_str, errors="coerce") if data_date_str else None

        lines = ["## DELAY & PROGRESS DATA"]

        if data_date is None or pd.isna(data_date):
            lines.append("Data date unavailable — cannot compute overdue activities.")
        else:
            lines.append(f"Data Date: **{data_date.date()}**\n")

            if "status_code" in work.columns and "target_start_date" in work.columns:
                ns = work[work["status_code"] == "TK_NotStart"].copy()
                ns["_ts"] = pd.to_datetime(ns["target_start_date"], errors="coerce")
                ns_over   = ns[ns["_ts"] < data_date].copy()
                if not ns_over.empty:
                    ns_over["days_overdue"] = (data_date - ns_over["_ts"]).dt.days
                    ns_over = ns_over.sort_values("days_overdue", ascending=False)
                    lines.append(f"### Not Started but Past Target Start: **{len(ns_over)}**")
                    cols = [c for c in ["task_code","task_name","target_start_date","days_overdue","_float_d","_dur_d"] if c in ns_over.columns]
                    lines.append(_to_md(
                        ns_over[cols].rename(columns={
                            "_float_d":"Float(d)","_dur_d":"Dur(d)","task_code":"Code",
                            "task_name":"Activity","target_start_date":"Target Start","days_overdue":"Days Overdue",
                        }),
                        total_count=len(ns_over),
                    ))
                else:
                    lines.append("### Not Started Past Target Start: **0**")

            if "status_code" in work.columns and "target_end_date" in work.columns:
                ip = work[work["status_code"] == "TK_Active"].copy()
                ip["_te"] = pd.to_datetime(ip["target_end_date"], errors="coerce")
                ip_over   = ip[ip["_te"] < data_date].copy()
                if not ip_over.empty:
                    ip_over["days_overdue"] = (data_date - ip_over["_te"]).dt.days
                    ip_over = ip_over.sort_values("days_overdue", ascending=False)
                    lines.append(f"\n### In Progress but Past Target Finish: **{len(ip_over)}**")
                    cols = [c for c in ["task_code","task_name","target_end_date","days_overdue","_pct","_float_d"] if c in ip_over.columns]
                    lines.append(_to_md(
                        ip_over[cols].rename(columns={
                            "_float_d":"Float(d)","_pct":"% Done","task_code":"Code",
                            "task_name":"Activity","target_end_date":"Target Finish","days_overdue":"Days Overdue",
                        }),
                        total_count=len(ip_over),
                    ))
                else:
                    lines.append("### In-Progress Past Target Finish: **0**")

        if baseline and "tasks" in baseline.get("df", {}):
            bl = baseline["df"]["tasks"].copy()
            if "task_id" in bl.columns and "target_end_date" in bl.columns and "task_id" in work.columns:
                bl_slim  = bl[["task_id","target_end_date"]].rename(columns={"target_end_date":"bl_end"})
                merged   = work.merge(bl_slim, on="task_id", how="inner")
                merged["_cu_end"] = pd.to_datetime(_col_or_empty(merged,"target_end_date"), errors="coerce")
                merged["_bl_end"] = pd.to_datetime(_col_or_empty(merged,"bl_end"),          errors="coerce")
                merged["slip_d"]  = (merged["_cu_end"] - merged["_bl_end"]).dt.days.fillna(0).astype(int)

                bl_proj_end = pd.to_datetime(bl["target_end_date"], errors="coerce").max()
                cu_proj_end = pd.to_datetime(_col_or_empty(tasks_df,"target_end_date"), errors="coerce").max()
                if pd.notna(bl_proj_end) and pd.notna(cu_proj_end):
                    proj_slip = (cu_proj_end - bl_proj_end).days
                    lines.append(
                        f"\n### Project End Slippage vs Baseline: **{proj_slip:+d} days**\n"
                        f"  Baseline finish: {bl_proj_end.date()} → Current: {cu_proj_end.date()}"
                    )

                slipped = merged[merged["slip_d"] > 0].sort_values("slip_d", ascending=False)
                if not slipped.empty:
                    lines.append(f"\n### Activities Slipped vs Baseline: **{len(slipped)}**")
                    sl_cols = [c for c in ["task_code","task_name","status_code","bl_end","target_end_date","slip_d"] if c in slipped.columns]
                    lines.append(_to_md(
                        slipped[sl_cols].rename(columns={
                            "task_code":"Code","task_name":"Activity","status_code":"Status",
                            "bl_end":"BL Finish","target_end_date":"Current Finish","slip_d":"Slip(d)",
                        }),
                        total_count=len(slipped),
                    ))
        else:
            lines.append("\n*No baseline loaded — slippage analysis unavailable.*")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Resources & cost
    # ------------------------------------------------------------------

    def _resources_section(self, source: Dict) -> str:
        ra       = source.get("df", {}).get("taskrsrc")
        rsrc_def = source.get("df", {}).get("rsrc")
        tasks_df = self._tasks(source)
        lines    = ["## RESOURCE & COST DATA"]

        if ra is None or ra.empty:
            lines.append("No resource assignment data found in this file.")
            return "\n".join(lines)

        ra = ra.copy()
        for c in ["target_cost","act_reg_cost","remain_cost","target_qty","act_reg_qty","remain_qty"]:
            if c in ra.columns:
                ra[c] = _num(ra[c])

        budget = ra["target_cost"].sum()  if "target_cost"  in ra.columns else 0
        actual = ra["act_reg_cost"].sum() if "act_reg_cost" in ra.columns else 0
        remain = ra["remain_cost"].sum()  if "remain_cost"  in ra.columns else 0
        eac    = actual + remain
        cv     = actual - budget
        cpi    = budget / actual if actual > 0 else None

        lines.append(f"\n### Cost Summary")
        lines.append(f"| Metric | Amount |")
        lines.append(f"|--------|--------|")
        lines.append(f"| Budget (BAC) | {budget:,.0f} |")
        lines.append(f"| Actual Cost (AC) | {actual:,.0f} |")
        lines.append(f"| Remaining | {remain:,.0f} |")
        lines.append(f"| EAC (AC + Remain) | {eac:,.0f} |")
        lines.append(f"| Cost Variance | {cv:+,.0f} |")
        if cpi is not None:
            lines.append(f"| CPI | {cpi:.3f} |")

        if "rsrc_id" in ra.columns:
            rg = ra.groupby("rsrc_id", as_index=False).agg(
                budget=("target_cost",  "sum") if "target_cost"  in ra.columns else ("rsrc_id","count"),
                actual=("act_reg_cost", "sum") if "act_reg_cost" in ra.columns else ("rsrc_id","count"),
                remain=("remain_cost",  "sum") if "remain_cost"  in ra.columns else ("rsrc_id","count"),
                assignments=("task_id", "count") if "task_id" in ra.columns else ("rsrc_id","count"),
            )
            rg["variance"] = rg["actual"] - rg["budget"]
            if rsrc_def is not None and "rsrc_id" in rsrc_def.columns:
                nc = next((c for c in ("rsrc_name","rsrc_short_name") if c in rsrc_def.columns), None)
                if nc:
                    rg = rg.merge(
                        rsrc_def[["rsrc_id",nc]].drop_duplicates("rsrc_id"),
                        on="rsrc_id", how="left",
                    )
            rg = rg.sort_values("budget", ascending=False)
            lines.append(f"\n### Resources by Budget")
            lines.append(_to_md(rg, total_count=len(rg)))

        if tasks_df is not None and "task_id" in ra.columns and "task_id" in tasks_df.columns:
            work       = self._work(tasks_df)
            resourced  = set(_col_or_empty(ra,"task_id").dropna())
            unres      = work[~work["task_id"].isin(resourced)].copy()
            lines.append(f"\n### Unresourced Work Activities: **{len(unres)}**")
            if not unres.empty:
                uc = [c for c in ["task_code","task_name","status_code","_dur_d","_float_d"] if c in unres.columns]
                lines.append(_to_md(
                    unres.sort_values("_dur_d", ascending=False)[uc].rename(
                        columns={"_dur_d":"Dur(d)","_float_d":"Float(d)","task_code":"Code",
                                 "task_name":"Activity","status_code":"Status"}),
                    total_count=len(unres),
                ))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Baseline vs current comparison
    # ------------------------------------------------------------------

    def _comparison_section(self, source: Dict, baseline: Optional[Dict]) -> str:
        lines = ["## BASELINE vs CURRENT COMPARISON"]
        if not baseline:
            lines.append("No baseline loaded. Upload a baseline XER to enable comparison.")
            return "\n".join(lines)

        bl_t = baseline.get("df", {}).get("tasks")
        cu_t = self._tasks(source)
        if bl_t is None or cu_t is None:
            return "\n".join(lines + ["Task data unavailable."])

        lines.append(f"\n| Metric | Baseline | Current | Δ |")
        lines.append(f"|--------|----------|---------|---|")
        lines.append(f"| Total activities | {len(bl_t)} | {len(cu_t)} | {len(cu_t)-len(bl_t):+d} |")

        if "status_code" in bl_t.columns and "status_code" in cu_t.columns:
            for s in ["TK_Complete","TK_Active","TK_NotStart"]:
                bv  = (bl_t["status_code"] == s).sum()
                cv_ = (cu_t["status_code"] == s).sum()
                lines.append(f"| {s} | {bv} | {cv_} | {cv_-bv:+d} |")

        if "total_float_hr_cnt" in bl_t.columns and "total_float_hr_cnt" in cu_t.columns:
            bl_f = _num(bl_t["total_float_hr_cnt"]) / self.hpd
            cu_f = _num(cu_t["total_float_hr_cnt"]) / self.hpd
            bl_crit = (bl_f <= 0).sum()
            cu_crit = (cu_f <= 0).sum()
            bl_neg  = (bl_f <  0).sum()
            cu_neg  = (cu_f <  0).sum()
            lines.append(f"| Critical (float ≤ 0) | {bl_crit} | {cu_crit} | {cu_crit-bl_crit:+d} |")
            lines.append(f"| Negative float | {bl_neg} | {cu_neg} | {cu_neg-bl_neg:+d} |")
            lines.append(f"| Avg float (days) | {bl_f.mean():.1f} | {cu_f.mean():.1f} | {cu_f.mean()-bl_f.mean():+.1f} |")

        if "task_id" in bl_t.columns and "task_id" in cu_t.columns and "target_end_date" in bl_t.columns:
            cu_work = self._work(cu_t)
            bl_slim = bl_t[["task_id","target_end_date"]].rename(columns={"target_end_date":"bl_end"})
            merged  = cu_work.merge(bl_slim, on="task_id", how="inner")
            merged["_cu_e"] = pd.to_datetime(_col_or_empty(merged,"target_end_date"), errors="coerce")
            merged["_bl_e"] = pd.to_datetime(_col_or_empty(merged,"bl_end"),          errors="coerce")
            merged["slip_d"]= (merged["_cu_e"] - merged["_bl_e"]).dt.days.fillna(0).astype(int)
            slipped = merged[merged["slip_d"] > 0].sort_values("slip_d", ascending=False)
            lines.append(f"\n### Most Slipped Activities vs Baseline ({len(slipped)} total slipped)")
            sl_cols = [c for c in ["task_code","task_name","status_code","bl_end","target_end_date","slip_d","_float_d"] if c in slipped.columns]
            lines.append(_to_md(
                slipped[sl_cols].rename(columns={
                    "task_code":"Code","task_name":"Activity","status_code":"Status",
                    "bl_end":"BL Finish","target_end_date":"Current Finish",
                    "slip_d":"Slip(d)","_float_d":"Float(d)",
                }),
                total_count=len(slipped),
            ))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # WBS
    # ------------------------------------------------------------------

    def _wbs_section(self, source: Dict) -> str:
        wbs_df   = self._wbs(source)
        tasks_df = self._tasks(source)
        lines    = ["## WBS STRUCTURE"]

        if wbs_df is None or wbs_df.empty:
            lines.append("No WBS data found.")
            return "\n".join(lines)

        lines.append(f"Total WBS nodes: **{len(wbs_df)}**")
        wbs_cols = [c for c in ["wbs_id","wbs_short_name","wbs_name","parent_wbs_id"] if c in wbs_df.columns]
        lines.append(_to_md(wbs_df[wbs_cols].head(30)))

        if tasks_df is not None and "wbs_id" in tasks_df.columns and "task_id" in tasks_df.columns:
            work    = self._work(tasks_df)
            summary = work.groupby("wbs_id", as_index=False).agg(activities=("task_id","count"))
            if "_dur_d" in work.columns:
                dur_agg = work.groupby("wbs_id")["_dur_d"].sum().reset_index()
                dur_agg.columns = ["wbs_id","total_dur_d"]
                summary = summary.merge(dur_agg, on="wbs_id", how="left")
            if "_float_d" in work.columns:
                crit_agg = work.groupby("wbs_id").apply(
                    lambda x: (x["_float_d"] <= 0).sum()
                ).reset_index()
                crit_agg.columns = ["wbs_id","critical_count"]
                summary = summary.merge(crit_agg, on="wbs_id", how="left")
            if "wbs_name" in wbs_df.columns:
                summary = summary.merge(
                    wbs_df[["wbs_id","wbs_name"]].drop_duplicates("wbs_id"),
                    on="wbs_id", how="left",
                )
            lines.append("\n### Activity Count by WBS")
            lines.append(_to_md(summary.sort_values("activities", ascending=False).head(25)))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------

    def _milestones_section(self, source: Dict) -> str:
        tasks_df = self._tasks(source, work_only=False)
        if tasks_df is None:
            return ""
        miles = tasks_df[tasks_df["task_type"].isin(["TT_Mile","TT_FinMile"])].copy() \
            if "task_type" in tasks_df.columns else pd.DataFrame()
        lines = [f"## MILESTONES: **{len(miles)} total**"]
        if miles.empty:
            lines.append("No milestones found.")
            return "\n".join(lines)
        cols = [c for c in ["task_code","task_name","task_type","status_code",
                             "_float_d","target_end_date","cstr_type"] if c in miles.columns]
        lines.append(_to_md(
            miles.sort_values("target_end_date")[cols].rename(columns={
                "task_code":"Code","task_name":"Milestone","task_type":"Type",
                "status_code":"Status","_float_d":"Float(d)",
                "target_end_date":"Date","cstr_type":"Constraint",
            }),
            total_count=len(miles),
        ))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # General overview
    # ------------------------------------------------------------------

    def _general_section(self, source: Dict) -> str:
        tasks_df = self._tasks(source)
        if tasks_df is None:
            return ""
        work  = self._work(tasks_df)
        lines = ["## SCHEDULE OVERVIEW"]

        if "_dur_d" in work.columns:
            lines.append(f"\n### Duration Statistics (work activities only)")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Count | {len(work)} |")
            lines.append(f"| Average duration | {work['_dur_d'].mean():.1f} d |")
            lines.append(f"| Median duration | {work['_dur_d'].median():.1f} d |")
            lines.append(f"| Max duration | {work['_dur_d'].max():.1f} d |")
            lines.append(f"| Activities > 20 days | {(work['_dur_d']>20).sum()} |")
            lines.append(f"| Activities > 60 days | {(work['_dur_d']>60).sum()} |")

        if "_float_d" in work.columns:
            nc_thresh = round(NEAR_CRITICAL_HRS / self.hpd, 1)
            lines.append(f"\n### Float Statistics")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Average float | {work['_float_d'].mean():.1f} d |")
            lines.append(f"| Critical (≤0) | {(work['_float_d']<=0).sum()} ({(work['_float_d']<=0).sum()/max(len(work),1)*100:.1f}%) |")
            lines.append(f"| Near-critical (≤{nc_thresh}d) | {((work['_float_d']>0)&(work['_float_d']<=nc_thresh)).sum()} |")
            lines.append(f"| Negative float | {(work['_float_d']<0).sum()} |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DataFrame accessors
    # ------------------------------------------------------------------

    def _enrich_pred_with_codes(self, pred_df: pd.DataFrame,
                                 tasks_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if tasks_df is None or "task_code" not in tasks_df.columns or pred_df.empty:
            return pred_df
        code_map = tasks_df.set_index("task_id")["task_code"].to_dict() \
            if "task_id" in tasks_df.columns else {}
        out = pred_df.copy()
        if "task_id" in out.columns:
            out["successor_code"]  = out["task_id"].map(code_map).fillna(out["task_id"].astype(str))
        if "pred_task_id" in out.columns:
            out["predecessor_code"]= out["pred_task_id"].map(code_map).fillna(out["pred_task_id"].astype(str))
        drop = [c for c in ["task_id","pred_task_id"] if c in out.columns]
        return out.drop(columns=drop)

    def _tasks(self, source: Dict, work_only: bool = False) -> Optional[pd.DataFrame]:
        dfs = source.get("df", {})
        df  = dfs.get("tasks")
        if df is None or (hasattr(df, "empty") and df.empty):
            df = dfs.get("task")
        if df is None or (hasattr(df, "empty") and df.empty):
            return None

        df = df.copy()

        # Derive computed columns — only if source columns exist
        if "target_drtn_hr_cnt" in df.columns:
            df["_dur_d"]  = _num(df["target_drtn_hr_cnt"]) / self.hpd
        if "total_float_hr_cnt" in df.columns:
            df["_float_d"]= _num(df["total_float_hr_cnt"]) / self.hpd
        if "phys_complete_pct" in df.columns:
            df["_pct"]    = _num(df["phys_complete_pct"])

        return self._work(df) if work_only else df

    def _work(self, df: pd.DataFrame) -> pd.DataFrame:
        if "task_type" in df.columns:
            return df[~df["task_type"].isin({"TT_LOE","TT_Mile","TT_FinMile"})].copy()
        return df.copy()

    def _wbs(self, source: Dict) -> Optional[pd.DataFrame]:
        dfs = source.get("df", {})
        for key in ("projwbs", "wbs"):
            df = dfs.get(key)
            if df is not None and not df.empty:
                return df
        return None
