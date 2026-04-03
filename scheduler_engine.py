"""
Scheduler Engine — CPM Forward / Backward Pass
Accepts EITHER raw TASKPRED rows or processed TASK_PREDECESSORS dicts.
All field-name differences are resolved at construction time via _normalize_rel().
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Field normalisation — single internal schema
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _normalize_rel(rel: Dict) -> Dict:
    """
    Convert ANY relationship dict to a single internal schema:
        _pred_id   : str   — predecessor task id
        _succ_id   : str   — successor task id  (the row's task_id)
        _rel_type  : str   — PR_FS | PR_SS | PR_FF | PR_SF
        _lag_hrs   : float — lag in hours (may be negative)

    Handles:
      • Raw TASKPRED       → pred_task_id, pred_type, lag_hr_cnt
      • Processed TASK_PRED → predecessor_task_id, relationship_type, lag_hours
    """
    if "pred_task_id" in rel:                          # raw TASKPRED
        return {
            "_pred_id":  str(rel.get("pred_task_id", "")),
            "_succ_id":  str(rel.get("task_id", "")),
            "_rel_type": str(rel.get("pred_type", "PR_FS")),
            "_lag_hrs":  _safe_float(rel.get("lag_hr_cnt", 0)),
        }
    if "predecessor_task_id" in rel:                   # processed
        return {
            "_pred_id":  str(rel.get("predecessor_task_id", "")),
            "_succ_id":  str(rel.get("task_id", "")),
            "_rel_type": str(rel.get("relationship_type", "PR_FS")),
            "_lag_hrs":  _safe_float(rel.get("lag_hours", 0)),
        }
    # unknown — safe fallback
    return {
        "_pred_id":  "",
        "_succ_id":  str(rel.get("task_id", "")),
        "_rel_type": "PR_FS",
        "_lag_hrs":  0.0,
    }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """
    Critical Path Method scheduler.

    Parameters
    ----------
    tasks : dict {task_id_str: task_dict}  OR  list of task dicts
        Each task dict must have  task_id  and  target_drtn_hr_cnt.
    relationships : list of relationship dicts (raw or processed — both accepted)
    hours_per_day : float  — calendar hours per working day (default 8)
    """

    def __init__(
        self,
        tasks: Any,
        relationships: List[Dict],
        hours_per_day: float = 8.0,
    ):
        # Normalise tasks → {str_id: dict}
        if isinstance(tasks, list):
            self.tasks: Dict[str, Dict] = {
                str(t.get("task_id", "")): t for t in tasks
            }
        else:
            self.tasks = {str(k): v for k, v in tasks.items()}

        # Normalise ALL relationships to the internal schema up-front
        self._rels: List[Dict] = [_normalize_rel(r) for r in relationships]

        self.hours_per_day = float(hours_per_day)

        # Adjacency lists — populated by build_graph()
        self.successors:   Dict[str, List[Dict]] = defaultdict(list)
        self.predecessors: Dict[str, List[Dict]] = defaultdict(list)

        # Results
        self.early_dates: Dict[str, Dict[str, float]] = {}
        self.late_dates:  Dict[str, Dict[str, float]] = {}
        self.float:       Dict[str, float] = {}
        self.topo_order:  List[str] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> None:
        """Populate successor/predecessor dicts from normalised relationships."""
        self.successors.clear()
        self.predecessors.clear()

        for rel in self._rels:
            pred = rel["_pred_id"]
            succ = rel["_succ_id"]
            # Skip dangling refs to unknown tasks
            if not pred or pred not in self.tasks:
                continue
            if not succ or succ not in self.tasks:
                continue
            self.successors[pred].append(rel)
            self.predecessors[succ].append(rel)

    # ------------------------------------------------------------------
    # Topological sort — Kahn's algorithm
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        in_degree = {t: 0 for t in self.tasks}
        for succ, rels in self.predecessors.items():
            in_degree[succ] = in_degree.get(succ, 0) + len(rels)

        queue = deque(t for t in self.tasks if in_degree[t] == 0)
        topo: List[str] = []

        while queue:
            node = queue.popleft()
            topo.append(node)
            for rel in self.successors.get(node, []):
                nxt = rel["_succ_id"]
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(topo) != len(self.tasks):
            missing = set(self.tasks) - set(topo)
            raise ValueError(
                f"Cycle detected in schedule logic — "
                f"{len(missing)} tasks not reachable. Sample: {list(missing)[:5]}"
            )

        self.topo_order = topo
        return topo

    # ------------------------------------------------------------------
    # Duration helper
    # ------------------------------------------------------------------

    def _dur(self, task_id: str) -> float:
        """Return task duration in DAYS."""
        hrs = _safe_float(
            self.tasks.get(task_id, {}).get("target_drtn_hr_cnt", 0)
        )
        return hrs / self.hours_per_day

    # ------------------------------------------------------------------
    # Forward pass  →  Early Start / Early Finish
    # ------------------------------------------------------------------

    def run_forward_pass(self) -> Dict[str, Dict[str, float]]:
        if not self.topo_order:
            self.topological_sort()

        for task_id in self.topo_order:
            dur  = self._dur(task_id)
            preds = self.predecessors.get(task_id, [])

            if not preds:
                ES = 0.0
            else:
                candidates = []
                for rel in preds:
                    p_early = self.early_dates.get(rel["_pred_id"], {"ES": 0.0, "EF": 0.0})
                    pES = p_early["ES"]
                    pEF = p_early["EF"]
                    lag  = rel["_lag_hrs"] / self.hours_per_day
                    rt   = rel["_rel_type"]

                    if rt == "PR_FS":
                        candidates.append(pEF + lag)
                    elif rt == "PR_SS":
                        candidates.append(pES + lag)
                    elif rt == "PR_FF":
                        # EF of succ = pEF + lag  →  ES = EF - dur
                        candidates.append(pEF + lag - dur)
                    elif rt == "PR_SF":
                        candidates.append(pES + lag - dur)
                    else:
                        candidates.append(pEF + lag)   # treat unknown as FS

                ES = max(candidates)

            ES = max(ES, 0.0)   # never negative
            self.early_dates[task_id] = {
                "ES": round(ES,       4),
                "EF": round(ES + dur, 4),
            }

        return self.early_dates

    # ------------------------------------------------------------------
    # Backward pass  →  Late Start / Late Finish
    # ------------------------------------------------------------------

    def run_backward_pass(self) -> Dict[str, Dict[str, float]]:
        if not self.topo_order:
            self.topological_sort()
        if not self.early_dates:
            self.run_forward_pass()

        pf = max(v["EF"] for v in self.early_dates.values())  # project finish

        for task_id in reversed(self.topo_order):
            dur   = self._dur(task_id)
            succs = self.successors.get(task_id, [])

            if not succs:
                LF = pf
            else:
                candidates = []
                for rel in succs:
                    s_late = self.late_dates.get(rel["_succ_id"], {"LS": pf, "LF": pf})
                    sLS = s_late["LS"]
                    sLF = s_late["LF"]
                    lag  = rel["_lag_hrs"] / self.hours_per_day
                    rt   = rel["_rel_type"]

                    if rt == "PR_FS":
                        candidates.append(sLS - lag)
                    elif rt == "PR_SS":
                        # LS of pred = sLS - lag
                        # LF of pred = LS + dur
                        candidates.append(sLS - lag + dur)
                    elif rt == "PR_FF":
                        candidates.append(sLF - lag)
                    elif rt == "PR_SF":
                        candidates.append(sLF - lag + dur)
                    else:
                        candidates.append(sLS - lag)

                LF = min(candidates)

            LS = LF - dur
            self.late_dates[task_id] = {
                "LS": round(LS, 4),
                "LF": round(LF, 4),
            }

        return self.late_dates

    # ------------------------------------------------------------------
    # Total float
    # ------------------------------------------------------------------

    def compute_float(self) -> Dict[str, float]:
        if not self.late_dates:
            self.run_backward_pass()

        for task_id in self.tasks:
            es = self.early_dates.get(task_id, {}).get("ES", 0.0)
            ls = self.late_dates.get(task_id,  {}).get("LS", 0.0)
            self.float[task_id] = round(ls - es, 4)

        return self.float

    # ------------------------------------------------------------------
    # One-shot convenience
    # ------------------------------------------------------------------

    def run_all(self) -> "Scheduler":
        """Build graph → sort → forward → backward → float."""
        self.build_graph()
        self.topological_sort()
        self.run_forward_pass()
        self.run_backward_pass()
        self.compute_float()
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        issues: Dict[str, Any] = {
            "start_nodes":           len([t for t in self.tasks if t not in self.predecessors]),
            "end_nodes":             len([t for t in self.tasks if t not in self.successors]),
            "negative_float_count":  sum(1 for v in self.float.values() if v < 0),
            "negative_es_count":     sum(1 for v in self.early_dates.values() if v["ES"] < -0.01),
        }
        neg_sample = [t for t, v in self.float.items() if v < -0.01][:10]
        if neg_sample:
            issues["negative_float_sample"] = neg_sample
        return issues

    # ------------------------------------------------------------------
    # Legacy alias — keeps callers that use compute_early_dates() working
    # ------------------------------------------------------------------

    def compute_early_dates(self) -> Dict:
        self.build_graph()
        self.topological_sort()
        return self.run_forward_pass()

