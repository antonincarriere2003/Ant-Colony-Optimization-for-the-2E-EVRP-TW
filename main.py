from __future__ import annotations

import io
import json
import math
import random
import shutil
import statistics
import time
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from dataclasses import asdict, is_dataclass

from src.instance import Instance2E, load_instance
from src.network import attach_distance_matrix
from src.aco import ACSParams
from src.acs_penalty import run_acs_penalty
from src.checks import (
    capacity_violation_lvl1,
    capacity_violation_lvl2,
    check_clients_served_and_assignment_hard,
    check_routes1_and_interechelon_hard,
    check_routes2_hard,
)
from src.solution import Solution2E

BUCKETS = ("Customer_5", "Customer_10", "Customer_15", "Customer_100")
RESULTS_ROOT = Path("results/batch_allsizes")
SEED = 0
LAMBDA_SYNC = 1000.0

BASE_PARAMS = ACSParams(
    m=12,
    Imax=8,
    alpha=1.8,
    beta=4.0,
    q0=0.80,
    rho=0.20,
    xi=0.08,
    tau0=1e-4,
    eps_dist=1e-9,
    heuristic_w_dist=2.5,
    heuristic_w_late=0.75,
    heuristic_w_energy=1.5,
    heuristic_w_risk=2.0,
    lambda_unserved=2e5,
    penalty_repair_fail=8e4,
    max_repair_attempts=4,
    max_ant_rebuild_attempts=50,
    seed_pheromone_boost=4.0,
    max_route_station_moves=40,
    max_exact_labels=30000,
    max_global_dfs_states=80000,
    global_update_rule="iter_best",
    ls_max_passes=5,
    ls_exact_threshold=8,
    ls_exact_small_threshold=5,
    ls_max_neighbors=100,
    lns_period=3,
    lns_stagnation_trigger=2,
    lns_destroy_fraction=0.20,
    lns_min_destroy=2,
    lns_max_destroy=12,
    lns_repair_passes=2,
    sp_period=4,
    sp_stagnation_trigger=3,
    sp_max_pool_routes=160,
    sp_elite_size=8,
    assign_alpha=1.5,
    assign_beta=2.0,
    assign_q0=0.80,
    assign_rho=0.18,
    assign_xi=0.08,
)


SMALL_INSTANCE_PARAMS = BASE_PARAMS


LARGE_INSTANCE_PARAMS = ACSParams(
    m=20,
    Imax=20,
    alpha=1.8,
    beta=4.0,
    q0=0.85,
    rho=0.15,
    xi=0.05,
    tau0=1e-4,
    eps_dist=1e-9,
    heuristic_w_dist=2.5,
    heuristic_w_late=0.75,
    heuristic_w_energy=1.5,
    heuristic_w_risk=2.0,
    lambda_unserved=2e5,
    penalty_repair_fail=8e4,
    max_repair_attempts=6,
    max_ant_rebuild_attempts=80,
    seed_pheromone_boost=4.0,
    max_route_station_moves=80,
    max_exact_labels=120000,
    max_global_dfs_states=250000,
    global_update_rule="iter_best",
    ls_max_passes=8,
    ls_exact_threshold=10,
    ls_exact_small_threshold=6,
    ls_max_neighbors=250,
    lns_period=3,
    lns_stagnation_trigger=2,
    lns_destroy_fraction=0.20,
    lns_min_destroy=2,
    lns_max_destroy=12,
    lns_repair_passes=2,
    sp_period=4,
    sp_stagnation_trigger=3,
    sp_max_pool_routes=160,
    sp_elite_size=8,
    assign_alpha=1.5,
    assign_beta=2.0,
    assign_q0=0.80,
    assign_rho=0.18,
    assign_xi=0.08,
)


def params_for_bucket(bucket: str) -> ACSParams:
    """
    Returns the ACS parameter set associated with the instance size bucket.
    """
    if bucket in {"Customer_5", "Customer_10", "Customer_15"}:
        return SMALL_INSTANCE_PARAMS
    if bucket == "Customer_100":
        return LARGE_INSTANCE_PARAMS
    return BASE_PARAMS


def set_global_seed(seed: int) -> None:
    """
    Sets the global random seed for Python and NumPy to ensure reproducible runs.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Safely converts a value to float and returns a default value if conversion fails.
    """
    try:
        return float(x)
    except Exception:
        return float(default)


def infer_customer_bucket(path: Path) -> str:
    """
    Infers the instance bucket name from the file path.
    """
    parts = {p.lower() for p in path.parts}
    if "customer_5" in parts:
        return "Customer_5"
    if "customer_10" in parts:
        return "Customer_10"
    if "customer_15" in parts:
        return "Customer_15"
    if "customer_100" in parts:
        return "Customer_100"
    return "Unknown"


def route_to_str(nodes: List[int]) -> str:
    """
    Converts a route represented by node ids into a readable string.
    """
    return " -> ".join(str(x) for x in nodes)


class TeeIO(io.TextIOBase):
    """
    Small utility stream used to duplicate standard output into several targets at once.

    This class is used to capture the raw solver log while still displaying it on screen.
    In practice, every message written to this object is forwarded to all registered
    streams, which allows the code to both print the solver progress to the console
    and store the same content in an in-memory buffer for later export into result files.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def discover_instance_files(data_root: Path, buckets: Sequence[str]) -> List[Path]:
    """
    Collects all instance files contained in the selected bucket folders.
    """
    files: List[Path] = []
    for bucket in buckets:
        bucket_dir = data_root / bucket
        if not bucket_dir.exists():
            raise FileNotFoundError(f"Missing folder: {bucket_dir}")
        files.extend(sorted(fp for fp in bucket_dir.glob("*.txt") if fp.is_file()))
    return files


def mean_or_zero(values: Iterable[float]) -> float:
    """
    Computes the mean of finite numeric values and returns 0.0 if none are valid.
    """
    vals = []
    for v in values:
        try:
            x = float(v)
            if math.isfinite(x):
                vals.append(x)
        except Exception:
            pass
    return float(sum(vals) / len(vals)) if vals else 0.0


def pstdev_or_zero(values: Iterable[float]) -> float:
    """
    Computes the population standard deviation of finite numeric values, or 0.0 if unavailable.
    """
    vals = []
    for v in values:
        try:
            x = float(v)
            if math.isfinite(x):
                vals.append(x)
        except Exception:
            pass
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def acsparams_to_dict(params: Any) -> Dict[str, Any]:
    """Convert ACSParams dataclass (or similar) to dict."""
    if is_dataclass(params):
        return asdict(params)
    if hasattr(params, "__dict__"):
        return dict(params.__dict__)
    return {"params_repr": repr(params)}


def summarize_solution(
    inst: Instance2E,
    sol: Solution2E,
    returned_score: float,
    cpu_s: float,
    source_path: str,
    params: ACSParams,
) -> Dict[str, Any]:
    """
    Builds a complete result dictionary from the final solution returned by the solver.

    This function centralizes all information that will later be written to reports,
    summaries, and the batch manifest. It extracts the main instance metadata,
    computation time, objective values, route structures, assignment decisions,
    penalties, debug information, and parameter values. Its role is to transform
    the internal solution object into a serializable and report-friendly structure
    used by the rest of the batch pipeline.
    """
    return {
        "instance_name": getattr(inst, "name", Path(source_path).stem),
        "source_path": source_path,
        "bucket": infer_customer_bucket(Path(source_path)),
        "seed": sol.meta.get("seed"),
        "acs_params": acsparams_to_dict(params),
        "returned_score": float(returned_score),
        "cpu_s": float(cpu_s),
        "dist_lvl1": sol.meta.get("dist_lvl1"),
        "dist_lvl2": sol.meta.get("dist_lvl2"),
        "distance_total": safe_float(sol.meta.get("dist_lvl1")) + safe_float(sol.meta.get("dist_lvl2")),
        "penalty_tw": sol.meta.get("penalty_tw"),
        "penalty_misc": sol.meta.get("penalty_misc"),
        "total_hybrid": sol.meta.get("total_hybrid"),
        "acs_history": list(sol.meta.get("acs_history", [])),
        "acs_best_before_repair": dict(sol.meta.get("acs_best_before_repair", {})),
        "acs_best_after_repair": dict(sol.meta.get("acs_best_after_repair", {})),
        "post_acs_debug": list(sol.meta.get("post_acs_debug", [])),
        "lvl1_debug": list(sol.meta.get("lvl1_debug", [])),
        "lvl2_absolute_debug": list(sol.meta.get("lvl2_absolute_debug", [])),
        "n_clients": len(inst.clients),
        "n_satellites": len(inst.satellites),
        "n_stations": len(inst.stations),
        "nv1": int(inst.nv1),
        "nv2": int(inst.nv2),
        "routes_lvl1": [r.nodes for r in sol.routes_lvl1],
        "routes_lvl2": {str(s): [r.nodes for r in rs] for s, rs in sol.routes_lvl2.items()},
        "assignment": dict(sol.assignment),
        "unserved_clients": list(sol.meta.get("unserved_clients", [])),
        "debug_msgs": list(sol.meta.get("debug_msgs", [])),
        "hard_checks": {},
        "soft_checks": {},
    }


def compute_checks(inst: Instance2E, sol: Solution2E) -> Dict[str, Any]:
    """
    Evaluates the final solution through hard feasibility checks and soft diagnostic indicators.

    The function runs the consistency tests defined in the checking module in order to verify:
    - second-echelon route validity,
    - customer service and assignment consistency,
    - first-echelon and inter-echelon coherence.
    It also computes softer indicators such as vehicle usage, capacity violations,
    penalties, and number of unserved customers. 
    """
    r2 = check_routes2_hard(inst, sol)
    rc = check_clients_served_and_assignment_hard(inst, sol)
    r1 = check_routes1_and_interechelon_hard(inst, sol)
    return {
        "hard": {
            "lvl2_routes_ok": bool(r2.hard_ok),
            "clients_assignment_ok": bool(rc.hard_ok),
            "lvl1_interechelon_ok": bool(r1.hard_ok),
            "all_ok": bool(r2.hard_ok and rc.hard_ok and r1.hard_ok),
            "lvl2_violations": list(r2.hard_violations),
            "clients_assignment_violations": list(rc.hard_violations),
            "lvl1_interechelon_violations": list(r1.hard_violations),
        },
        "soft": {
            "nv1_used": len(sol.routes_lvl1),
            "nv1_max": int(inst.nv1),
            "nv1_ok": len(sol.routes_lvl1) <= int(inst.nv1),
            "nv2_used": sum(len(v) for v in sol.routes_lvl2.values()),
            "nv2_max": int(inst.nv2),
            "nv2_ok": sum(len(v) for v in sol.routes_lvl2.values()) <= int(inst.nv2),
            "cap_violation_lvl1": float(capacity_violation_lvl1(inst, sol)),
            "cap_violation_lvl2": float(capacity_violation_lvl2(inst, sol)),
            "penalty_tw": float(sol.meta.get("penalty_tw", 0.0)),
            "penalty_misc": float(sol.meta.get("penalty_misc", 0.0)),
            "unserved_count": len(sol.meta.get("unserved_clients", [])),
        },
    }


def classify_result(result: Dict[str, Any]) -> str:
    """
    Classifies a result as OK, WARN, or FAIL based on feasibility and violation indicators.
    """
    hc = result["hard_checks"]
    sc = result["soft_checks"]
    if (
        hc["all_ok"]
        and sc["nv1_ok"]
        and sc["nv2_ok"]
        and sc["cap_violation_lvl1"] <= 1e-9
        and sc["cap_violation_lvl2"] <= 1e-9
        and sc["unserved_count"] == 0
    ):
        return "OK"
    if hc["lvl2_routes_ok"] and hc["clients_assignment_ok"]:
        return "WARN"
    return "FAIL"


def solve_one_instance(instance_path: str, seed: int, lambda_sync: float, params: ACSParams) -> Dict[str, Any]:
    """
    Solves a single instance from loading to final evaluation.
    """
    set_global_seed(seed)
    inst = load_instance(instance_path)
    attach_distance_matrix(inst)
    inst.source_path = instance_path
    if not getattr(inst, "name", None):
        inst.name = Path(instance_path).stem

    t0 = time.perf_counter()
    sol, best_score = run_acs_penalty(inst, params, lambda_sync=float(lambda_sync), seed=int(seed))
    cpu_s = time.perf_counter() - t0

    summary = summarize_solution(inst, sol, best_score, cpu_s, instance_path, params)
    checks = compute_checks(inst, sol)
    summary["hard_checks"] = checks["hard"]
    summary["soft_checks"] = checks["soft"]
    summary["status"] = classify_result(summary)
    return summary


def write_instance_report(folder: Path, result: Dict[str, Any], raw_log: str) -> None:
    """
    Writes a detailed text report for a single solved instance. It is useful both for
    result analysis and debugging.
    """
    soft = result.get("soft_checks", {})
    hard = result.get("hard_checks", {})
    lines: List[str] = []
    lines.append(f"Instance: {result.get('instance_name')}")
    lines.append(f"Bucket: {result.get('bucket')}")
    lines.append(f"Status: {result.get('status')}")
    lines.append(f"CPU (s): {safe_float(result.get('cpu_s')):.6f}")
    lines.append(f"Returned score: {safe_float(result.get('returned_score')):.6f}")
    lines.append(f"Distance level 1: {safe_float(result.get('dist_lvl1')):.6f}")
    lines.append(f"Distance level 2: {safe_float(result.get('dist_lvl2')):.6f}")
    lines.append(f"Distance total: {safe_float(result.get('distance_total')):.6f}")
    lines.append(f"Penalty TW: {safe_float(result.get('penalty_tw')):.6f}")
    lines.append(f"Penalty misc: {safe_float(result.get('penalty_misc')):.6f}")
    lines.append(f"Unserved clients: {soft.get('unserved_count', 0)}")
    lines.append(f"nv1 used / max: {soft.get('nv1_used', 0)} / {soft.get('nv1_max', 0)}")
    lines.append(f"nv2 used / max: {soft.get('nv2_used', 0)} / {soft.get('nv2_max', 0)}")
    lines.append("")
    lines.append(f"Hard checks all OK: {hard.get('all_ok', False)}")
    lines.append(f"Level 2 routes OK: {hard.get('lvl2_routes_ok', False)}")
    lines.append(f"Clients/assignment OK: {hard.get('clients_assignment_ok', False)}")
    lines.append(f"Level 1/inter-echelon OK: {hard.get('lvl1_interechelon_ok', False)}")
    lines.append(f"Capacity violation lvl1: {safe_float(soft.get('cap_violation_lvl1')):.6f}")
    lines.append(f"Capacity violation lvl2: {safe_float(soft.get('cap_violation_lvl2')):.6f}")
    lines.append("")

    lines.append("=== Routes level 1 ===")
    if result.get("routes_lvl1"):
        for idx, route in enumerate(result.get("routes_lvl1", []), start=1):
            lines.append(f"R1_{idx}: {route_to_str(route)}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("=== Routes level 2 ===")
    lvl2 = result.get("routes_lvl2", {})
    if lvl2:
        for sat_key in sorted(lvl2.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            lines.append(f"Satellite {sat_key}:")
            sat_routes = lvl2.get(sat_key, [])
            if sat_routes:
                for idx, route in enumerate(sat_routes, start=1):
                    lines.append(f"  R2_{idx}: {route_to_str(route)}")
            else:
                lines.append("  (none)")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("=== Assignment client -> satellite ===")
    assignment = result.get("assignment", {})
    if assignment:
        for client_id in sorted(assignment.keys(), key=lambda x: int(x)):
            lines.append(f"{client_id} -> {assignment[client_id]}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("=== Violations / debug ===")
    for label, values in (
        ("lvl2_violations", hard.get("lvl2_violations", [])),
        ("clients_assignment_violations", hard.get("clients_assignment_violations", [])),
        ("lvl1_interechelon_violations", hard.get("lvl1_interechelon_violations", [])),
        ("unserved_clients", result.get("unserved_clients", [])),
        ("debug_msgs", result.get("debug_msgs", [])),
    ):
        lines.append(f"{label}: {values}")

    lines.append("")
    lines.append("=== Parameters ===")
    lines.append(json.dumps(result.get("acs_params", {}), indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("=== Raw solver log ===")
    lines.append(raw_log.rstrip())
    (folder / f"{result.get('instance_name')}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bucket_summary(folder: Path, bucket: str, rows: List[Dict[str, Any]]) -> None:
    """
    Writes the aggregate summary file for one bucket of instances.
    
    This function computes the main statistics over all instances of a given size bucket,
    such as:
    - number of instances,
    - distribution of statuses,
    - average and standard deviation of total distance,
    - average CPU time.
    It also appends a detailed per-instance table so that each bucket has a compact
    summary file useful for performance comparison and experimental analysis.
    """
    ok_count = sum(1 for r in rows if r.get("status") == "OK")
    warn_count = sum(1 for r in rows if r.get("status") == "WARN")
    fail_count = sum(1 for r in rows if r.get("status") == "FAIL")
    error_count = sum(1 for r in rows if r.get("status") == "ERROR")
    mean_total = mean_or_zero(r.get("distance_total") for r in rows)
    std_total = pstdev_or_zero(r.get("distance_total") for r in rows)
    mean_cpu = mean_or_zero(r.get("cpu_s") for r in rows)

    lines: List[str] = []
    lines.append(f"Summary {bucket}")
    lines.append(f"Number of instances: {len(rows)}")
    lines.append(f"Status OK/WARN/FAIL/ERROR: {ok_count}/{warn_count}/{fail_count}/{error_count}")
    lines.append(f"Average total distance: {mean_total:.6f}")
    lines.append(f"Std total distance: {std_total:.6f}")
    lines.append(f"Average CPU (s): {mean_cpu:.6f}")
    lines.append("")
    lines.append("=== Details by instance ===")
    for row in sorted(rows, key=lambda r: str(r.get("instance_name", ""))):
        lines.append(
            f"{row.get('instance_name')} | status={row.get('status')} | "
            f"dist_total={safe_float(row.get('distance_total')):.6f} | "
            f"dist1={safe_float(row.get('dist_lvl1')):.6f} | "
            f"dist2={safe_float(row.get('dist_lvl2')):.6f} | "
            f"cpu_s={safe_float(row.get('cpu_s')):.6f}"
        )
    (folder / f"summary_{bucket}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """
    Runs the complete batch experiment over all selected instance buckets.

    This function is the global entry point of the experimental pipeline. It first
    discovers all instance files from the configured data folders, recreates the output
    directory, and initializes the result containers. Then, for each instance, it:
    - determines the corresponding bucket,
    - selects the appropriate parameter set,
    - captures the solver output while still displaying it,
    - solves the instance,
    - handles possible errors gracefully,
    - writes the detailed report file.

    Once all instances have been processed, it generates one summary file per bucket,
    exports a global JSON manifest containing the whole batch configuration and results,
    and prints the final output directory. In short, this function orchestrates the full
    benchmark campaign from input discovery to final report generation.
    """
    data_root = Path("data")
    files = discover_instance_files(data_root, BUCKETS)

    if RESULTS_ROOT.exists():
        shutil.rmtree(RESULTS_ROOT)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    bucket_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_runs = len(files)

    print("\n=== BATCH RUN : Customer_100 ===")
    print(f"Instances selected: {total_runs}")
    for fp in files:
        print(f"- {infer_customer_bucket(fp)} :: {fp.name}")

    for run_idx, fp in enumerate(files, start=1):
        bucket = infer_customer_bucket(fp)
        bucket_folder = RESULTS_ROOT / bucket
        bucket_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[{run_idx}/{total_runs}] {bucket} :: {fp.name}")
        buffer = io.StringIO()
        tee = TeeIO(buffer, __import__("sys").stdout)

        try:
            with redirect_stdout(tee):
                params = params_for_bucket(bucket)
                result = solve_one_instance(str(fp), seed=SEED, lambda_sync=LAMBDA_SYNC, params=params)
            print(
                f"  status={result['status']} distance_total={safe_float(result['distance_total']):.3f} "
                f"cpu={safe_float(result['cpu_s']):.3f}s"
            )
        except Exception as e:
            result = {
                "instance_name": fp.stem,
                "source_path": str(fp),
                "bucket": bucket,
                "seed": SEED,
                "acs_params": acsparams_to_dict(params),
                "returned_score": math.nan,
                "cpu_s": math.nan,
                "dist_lvl1": math.nan,
                "dist_lvl2": math.nan,
                "distance_total": math.nan,
                "penalty_tw": math.nan,
                "penalty_misc": math.nan,
                "total_hybrid": math.nan,
                "routes_lvl1": [],
                "routes_lvl2": {},
                "assignment": {},
                "unserved_clients": [],
                "debug_msgs": [],
                "hard_checks": {},
                "soft_checks": {},
                "status": "ERROR",
                "error_message": str(e),
            }
            print(f"  ERROR: {type(e).__name__}: {e}")
            buffer.write(f"\nERROR: {type(e).__name__}: {e}\n")

        bucket_rows[bucket].append(result)
        write_instance_report(bucket_folder, result, buffer.getvalue())

    for bucket in BUCKETS:
        write_bucket_summary(RESULTS_ROOT, bucket, bucket_rows.get(bucket, []))

    manifest = {
        "seed": SEED,
        "lambda_sync": LAMBDA_SYNC,
        "base_params": acsparams_to_dict(BASE_PARAMS),
        "buckets": list(BUCKETS),
        "results": bucket_rows,
    }
    (RESULTS_ROOT / "batch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== END BATCH RUN ===")
    print(f"Output folder: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
