from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import random

from src.instance import Instance2E
from src.solution import Solution2E, Route, VehicleIdGenerator
from src.aco import (
    ACSParams,
    Pheromones,
    HeuristicMemory,
    RetryParams,
    START_NODE_ID,
    _build_one_attempt,
    construct_seed_solution_lvl2,
    seed_pheromone_from_solution,
    strict_reinsert_remaining,
)
from src.lvl1_cw import (
    build_satellite_demands,
    clarke_wright_lvl1,
    simulate_truck_arrivals,
    distance_of_route,
    repair_merge_lvl1_to_fleet,
)
from src.sync import repair_and_replay_lvl2_absolute, _refresh_solution_consistency
from src.checks import capacity_violation_lvl2, capacity_violation_lvl1
from src.local_search import intensify_lvl2_solution, intensify_absolute_with_fixed_lvl1, periodic_destroy_repair_lns, optimize_solution_recharges, recombine_route_pool


@dataclass
class AntPenaltyScore:
    """Stores the penalized evaluation of an ant solution together with its main components."""
    total: float
    dist_lvl2: float
    penalty_unserved: float
    penalty_route_excess: float
    penalty_cap_lvl2: float
    n_unserved: int
    route_count: int


def _dist(inst: Instance2E, i: int, j: int) -> float:
    """Returns the distance between two nodes in the instance distance matrix."""
    return float(inst.require_dist()[inst.idx(i)][inst.idx(j)])


def _lvl2_distance(inst: Instance2E, sol: Solution2E) -> float:
    """Computes the total traveled distance over all second-level routes of a solution."""
    return sum(distance_of_route(inst, r.nodes) for rs in sol.routes_lvl2.values() for r in rs)


def _latest_sat_arrival(inst: Instance2E, assignment: Dict[int, int], rel_from_sat: Dict[int, float]) -> Dict[int, float]:
    """Computes the latest feasible arrival time at each satellite induced by assigned client schedules."""
    out: Dict[int, float] = {}
    for s in inst.satellite_ids:
        vals = [float(inst.tw_late(c)) - float(rel_from_sat[c]) for c, ss in assignment.items() if ss == s and c in rel_from_sat]
        out[s] = min(vals) if vals else float('inf')
    return out


def _elite_key(sol: Solution2E) -> Tuple[Tuple[int, Tuple[Tuple[int, ...], ...]], ...]:
    """Builds a canonical signature of the second-level solution structure for elite deduplication."""
    sig: List[Tuple[int, Tuple[Tuple[int, ...], ...]]] = []
    for sat in sorted(sol.routes_lvl2.keys()):
        routes = tuple(tuple(int(n) for n in r.nodes if n in sol.assignment) for r in sol.routes_lvl2[sat])
        sig.append((int(sat), routes))
    return tuple(sig)


def _push_elite(elites: List[Solution2E], cand: Optional[Solution2E], inst: Instance2E, limit: int) -> List[Solution2E]:
    """Inserts a candidate into the elite pool while removing duplicates and keeping only the best solutions."""
    if cand is None:
        return elites
    sig = _elite_key(cand)
    kept: List[Solution2E] = []
    seen = set()
    
    # Merge the current elites with the candidate and rank them by second-level distance
    all_sols = list(elites) + [cand]
    all_sols.sort(key=lambda s: _lvl2_distance(inst, s))
    
    # Keep only distinct elite structures up to the requested pool size
    for sol in all_sols:
        s = _elite_key(sol)
        if s in seen:
            continue
        kept.append(sol)
        seen.add(s)
        if len(kept) >= max(1, int(limit)):
            break
    return kept


def evaluate_ant_penalty(inst: Instance2E, sol: Solution2E, params: ACSParams) -> AntPenaltyScore:
    """Evaluates an ant solution with penalties for unserved clients, route excess, and second-level capacity violations."""
    # Compute the base second-level travel distance and the main infeasibility indicators
    dist2 = _lvl2_distance(inst, sol)
    n_unserved = len(sol.meta.get('unserved_clients', []))
    route_count = sum(len(v) for v in sol.routes_lvl2.values())
    
    # Build the penalized score used to compare partially feasible ant solutions
    penalty_unserved = float(params.lambda_unserved) * float(n_unserved)
    penalty_route_excess = 2.5e4 * max(0, route_count - int(inst.nv2))
    penalty_cap = 2.0e4 * float(capacity_violation_lvl2(inst, sol))
    total = dist2 + penalty_unserved + penalty_route_excess + penalty_cap
    return AntPenaltyScore(
        total=float(total),
        dist_lvl2=float(dist2),
        penalty_unserved=float(penalty_unserved),
        penalty_route_excess=float(penalty_route_excess),
        penalty_cap_lvl2=float(penalty_cap),
        n_unserved=int(n_unserved),
        route_count=int(route_count),
    )


def _clone_routes(routes_by_sat: Dict[int, List[Route]]) -> Dict[int, List[Route]]:
    """Creates a deep-enough copy of the route dictionary by duplicating route nodes and vehicle identifiers."""
    return {int(s): [Route(nodes=list(r.nodes), vehicle_id=int(r.vehicle_id)) for r in rs] for s, rs in routes_by_sat.items() if rs}


def _construct_penalized_ant(inst: Instance2E, ph: Pheromones, hm: HeuristicMemory, params: ACSParams, rng: random.Random, gen: VehicleIdGenerator) -> Solution2E:
    """
    Builds one ant solution and wraps it into a penalized second-level solution
    object with construction metadata.
    """
    # Construct one hierarchical ant attempt and recover its routing data
    routes_by_sat, assignment, t_rel_arr, rel_from_sat, unserved, debug_msgs, used_arcs = _build_one_attempt(inst, ph, hm, params, rng, gen)
    
    # Derive the latest feasible satellite arrival information from the relative client timings
    latest = _latest_sat_arrival(inst, assignment, rel_from_sat)
    return Solution2E(
        assignment=dict(assignment),
        routes_lvl2=_clone_routes(routes_by_sat),
        routes_lvl1=[],
        meta={
            't_rel_arr': {int(k): float(v) for k, v in t_rel_arr.items()},
            'rel_from_sat': {int(k): float(v) for k, v in rel_from_sat.items()},
            'latest_sat_arrival': {int(k): float(v) for k, v in latest.items()},
            'unserved_clients': sorted(int(x) for x in unserved),
            'debug_msgs': list(debug_msgs),
            'used_arcs': [(int(i), int(j)) for i, j in used_arcs],
            'ant_accept': True,
            'infeasible': bool(unserved),
        },
    )


def _repair_best_lvl2(inst: Instance2E, base_sol: Solution2E, params: ACSParams) -> Tuple[Solution2E, List[str], bool]:
    """
    Repairs a second-level solution by strictly reinserting unserved clients
    into existing routes when possible.
    """
    # Copy the current second-level structures and recover the timing metadata
    routes_by_sat = _clone_routes(base_sol.routes_lvl2)
    assignment = dict(base_sol.assignment)
    t_rel_arr = {int(k): float(v) for k, v in base_sol.meta.get('t_rel_arr', {}).items()}
    rel_from_sat = {int(k): float(v) for k, v in base_sol.meta.get('rel_from_sat', {}).items()}
    remaining: Set[int] = set(int(x) for x in base_sol.meta.get('unserved_clients', []))
    debug_msgs: List[str] = list(base_sol.meta.get('debug_msgs', []))
    
    # Attempt a strict reinsertion only when some clients are still unserved
    ok = True
    if remaining:
        ok, _ = strict_reinsert_remaining(inst, routes_by_sat, assignment, t_rel_arr, rel_from_sat, remaining, debug_msgs)
        
    # Build the repaired solution and refresh the main metadata fields
    repaired = Solution2E(
        assignment=dict(assignment),
        routes_lvl2=_clone_routes(routes_by_sat),
        routes_lvl1=[],
        meta=dict(base_sol.meta),
    )
    repaired.meta['t_rel_arr'] = t_rel_arr
    repaired.meta['rel_from_sat'] = rel_from_sat
    repaired.meta['latest_sat_arrival'] = _latest_sat_arrival(inst, assignment, rel_from_sat)
    repaired.meta['unserved_clients'] = sorted(int(x) for x in remaining)
    repaired.meta['post_acs_repair_ok'] = bool(ok and not remaining)
    repaired.meta['post_acs_debug'] = list(debug_msgs)
    repaired.meta['infeasible'] = bool(remaining)
    return repaired, debug_msgs, bool(ok and not remaining)


def _finalize_hybrid(inst: Instance2E, lvl2_sol: Solution2E, params: ACSParams, lambda_sync: float) -> Tuple[Solution2E, float]:
    """
    Finalizes the hybrid solution by reconstructing level 1, replaying level 2
    in absolute time, and computing the global penalized objective.
    """
    # Rebuild the first level from satellite demands and repair it if the truck fleet limit is exceeded
    latest = lvl2_sol.meta.get('latest_sat_arrival', {}) or {}
    D = build_satellite_demands(inst, lvl2_sol.assignment)
    routes_lvl1, dbg1 = clarke_wright_lvl1(inst, D, latest)
    if len(routes_lvl1) > int(inst.nv1):
        routes_lvl1, dbg1b, ok_merge = repair_merge_lvl1_to_fleet(inst, routes_lvl1, D, int(inst.nv1), latest)
        dbg1 = list(dbg1) + list(dbg1b)
    else:
        ok_merge = True
        
    # Combine the rebuilt first level with the current second level and replay the EV routes in absolute time
    sol = Solution2E(
        assignment=dict(lvl2_sol.assignment),
        routes_lvl2=_clone_routes(lvl2_sol.routes_lvl2),
        routes_lvl1=list(routes_lvl1),
        meta=dict(lvl2_sol.meta),
    )
    A_s = simulate_truck_arrivals(inst, sol.routes_lvl1)
    sol_abs, tw_pen, _, abs_arr, dbg2, ok_abs = repair_and_replay_lvl2_absolute(inst, sol, A_s, max_attempts=int(params.max_repair_attempts))
    sol_abs = _refresh_solution_consistency(inst, sol_abs)
    
    # Evaluate the final hybrid solution with synchronization, fleet, and capacity penalties
    dist1 = sum(distance_of_route(inst, r.nodes) for r in sol_abs.routes_lvl1)
    dist2 = _lvl2_distance(inst, sol_abs)
    unserved = len(sol_abs.meta.get('unserved_clients', []))

    nv1_excess = max(0, len(sol_abs.routes_lvl1) - int(inst.nv1))
    nv2_excess = max(0, sum(len(v) for v in sol_abs.routes_lvl2.values()) - int(inst.nv2))
    cap1_pen = 2.0e4 * float(capacity_violation_lvl1(inst, sol_abs))
    cap2_pen = 2.0e4 * float(capacity_violation_lvl2(inst, sol_abs))
    nv1_pen = 5.0e3 * nv1_excess
    nv2_pen = 5.0e3 * nv2_excess

    penalty_misc = nv1_pen + nv2_pen + cap1_pen + cap2_pen
    total = dist1 + dist2 + float(lambda_sync) * float(tw_pen) + float(params.lambda_unserved) * float(unserved) + penalty_misc
    
    # Emit diagnostic messages for any remaining structural infeasibilities
    if nv1_excess > 0:
        msg = f"nv1_excess={nv1_excess} penalty={nv1_pen:.2f} used={len(sol_abs.routes_lvl1)} max={int(inst.nv1)}"
        dbg1.append(msg)
        print(f"[FINAL][PENALTY] {msg}")
    if nv2_excess > 0:
        msg = f"nv2_excess={nv2_excess} penalty={nv2_pen:.2f}"
        dbg2.append(msg)
        print(f"[FINAL][PENALTY] {msg}")
    if cap1_pen > 0.0:
        print(f"[FINAL][PENALTY] lvl1_capacity penalty={cap1_pen:.2f}")
    if cap2_pen > 0.0:
        print(f"[FINAL][PENALTY] lvl2_capacity penalty={cap2_pen:.2f}")
        
    # Store the final evaluation and feasibility indicators in the solution metadata
    sol_abs.meta.update({
        'dist_lvl1': float(dist1),
        'dist_lvl2': float(dist2),
        'penalty_tw': float(tw_pen),
        'penalty_misc': float(penalty_misc),
        'total_hybrid': float(total),
        'lvl1_debug': list(dbg1),
        'lvl2_absolute_debug': list(dbg2),
        'lvl1_merge_ok': bool(ok_merge),
        'lvl2_absolute_ok': bool(ok_abs),
        'lambda_sync': float(lambda_sync),
        'nv1_excess': int(nv1_excess),
        'nv1_excess_penalty': float(nv1_pen),
        'nv2_excess': int(nv2_excess),
        'nv2_excess_penalty': float(nv2_pen),
        'infeasible': bool(unserved or nv1_excess or nv2_excess or cap1_pen > 0.0 or cap2_pen > 0.0),
    })
    return sol_abs, float(total)


def _prepare_acs_params_for_instance(inst: Instance2E, params: ACSParams) -> ACSParams:
    """Adapts ACS parameters to the instance size in order to give larger cases a deeper search budget."""
    p = ACSParams(**params.__dict__)
    n_clients = len(inst.client_ids)

    if n_clients >= 15:
        p.m = max(int(p.m), 18)
        p.Imax = max(int(p.Imax), 14)
        p.lns_period = max(2, int(p.lns_period))
        p.sp_period = max(3, int(p.sp_period))
    elif n_clients >= 10:
        p.m = max(int(p.m), 16)
        p.Imax = max(int(p.Imax), 12)
        p.lns_period = max(2, int(p.lns_period))
        p.sp_period = max(3, int(p.sp_period))

    return p


def _initialize_acs_state(
    inst: Instance2E,
    params: ACSParams,
    seed: int,
) -> Tuple[random.Random, Pheromones, HeuristicMemory, RetryParams, VehicleIdGenerator]:
    """Initializes the random generator and the main ACS data structures."""
    rng = random.Random(seed)
    ph = Pheromones(inst, params.tau0)
    hm = HeuristicMemory(inst, 1.0)
    retry = RetryParams()
    gen = VehicleIdGenerator()
    return rng, ph, hm, retry, gen


def _try_initialize_seed_solution(
    inst: Instance2E,
    params: ACSParams,
    gen: VehicleIdGenerator,
    ph: Pheromones,
    elite_solutions: List[Solution2E],
    acs_history: List[Dict[str, float]],
) -> Tuple[
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    List[Solution2E],
]:
    """
    Builds the initial seed solution, evaluates it, and injects it into the ACS
    state if successful.
    """
    best_any_sol: Optional[Solution2E] = None
    best_any_score: Optional[AntPenaltyScore] = None
    best_sol: Optional[Solution2E] = None
    best_score: Optional[AntPenaltyScore] = None

    try:
        seed_sol = construct_seed_solution_lvl2(inst, params, gen)
        seed_score = evaluate_ant_penalty(inst, seed_sol, params)

        best_any_sol, best_any_score = seed_sol, seed_score
        best_sol, best_score = seed_sol, seed_score
        elite_solutions = _push_elite(
            elite_solutions,
            seed_sol,
            inst,
            int(getattr(params, 'sp_elite_size', 8)),
        )

        acs_history.append({
            'iter': 0,
            'ant': 0,
            'best_total': seed_score.total,
            'best_unserved': seed_score.n_unserved,
            'best_dist_lvl2': seed_score.dist_lvl2,
        })

        seed_pheromone_from_solution(ph, seed_sol, params)
    except Exception as e:
        print(f"[SEED] failed: {type(e).__name__}: {e}")

    return best_any_sol, best_any_score, best_sol, best_score, elite_solutions


def _run_one_acs_iteration(
    inst: Instance2E,
    ph: Pheromones,
    hm: HeuristicMemory,
    params: ACSParams,
    rng: random.Random,
    gen: VehicleIdGenerator,
    it: int,
    best_any_sol: Optional[Solution2E],
    best_any_score: Optional[AntPenaltyScore],
    best_sol: Optional[Solution2E],
    best_score: Optional[AntPenaltyScore],
    acs_history: List[Dict[str, float]],
) -> Tuple[
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    bool,
]:
    """Runs one ACS iteration over all ants and updates the best solutions found so far."""
    iter_best_sol: Optional[Solution2E] = None
    iter_best_score: Optional[AntPenaltyScore] = None
    improved_this_iter = False

    # Evaluate all ants and update iteration-best and global-best references
    for ant in range(1, int(params.m) + 1):
        sol = _construct_penalized_ant(inst, ph, hm, params, rng, gen)
        sc = evaluate_ant_penalty(inst, sol, params)

        if iter_best_score is None or sc.total < iter_best_score.total:
            iter_best_sol, iter_best_score = sol, sc

        if best_any_score is None or sc.total < best_any_score.total:
            best_any_sol, best_any_score = sol, sc
            acs_history.append({
                'iter': it,
                'ant': ant,
                'best_total': sc.total,
                'best_unserved': sc.n_unserved,
                'best_dist_lvl2': sc.dist_lvl2,
            })
            improved_this_iter = True

        if best_score is None or sc.total < best_score.total:
            best_sol, best_score = sol, sc

    return (
        iter_best_sol,
        iter_best_score,
        best_any_sol,
        best_any_score,
        best_sol,
        best_score,
        improved_this_iter,
    )


def _select_pheromone_update_target(
    params: ACSParams,
    best_sol: Optional[Solution2E],
    best_score: Optional[AntPenaltyScore],
    iter_best_sol: Optional[Solution2E],
    iter_best_score: Optional[AntPenaltyScore],
) -> Tuple[Optional[Solution2E], Optional[AntPenaltyScore]]:
    """Selects the solution used for the global pheromone update according to the configured rule."""
    if params.global_update_rule == "global_best":
        return best_sol, best_score
    return iter_best_sol, iter_best_score


def _maybe_run_periodic_lns(
    inst: Instance2E,
    params: ACSParams,
    rng: random.Random,
    it: int,
    stagnation_iters: int,
    best_sol: Optional[Solution2E],
    best_score: Optional[AntPenaltyScore],
    update_sol: Optional[Solution2E],
    update_score: Optional[AntPenaltyScore],
    best_any_sol: Optional[Solution2E],
    best_any_score: Optional[AntPenaltyScore],
    elite_solutions: List[Solution2E],
    acs_history: List[Dict[str, float]],
    lns_debug: List[str],
    improved_this_iter: bool,
) -> Tuple[
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    List[Solution2E],
    bool,
]:
    """Runs the periodic destroy-repair LNS when its triggering conditions are met."""
    trigger_periodic = int(getattr(params, 'lns_period', 0)) > 0 and (
        it % max(1, int(getattr(params, 'lns_period', 1))) == 0
    )
    trigger_stagnation = int(getattr(params, 'lns_stagnation_trigger', 0)) > 0 and (
        stagnation_iters >= int(getattr(params, 'lns_stagnation_trigger', 0))
    )

    if update_sol is None or update_score is None or not (trigger_periodic or trigger_stagnation):
        return (
            best_sol,
            best_score,
            best_any_sol,
            best_any_score,
            update_sol,
            update_score,
            elite_solutions,
            improved_this_iter,
        )

    # Improve the current target solution through a periodic destroy-repair phase
    lns_target_sol = best_sol if best_sol is not None else update_sol
    lns_target_score = best_score if best_score is not None else update_score
    lns_res = periodic_destroy_repair_lns(
        inst,
        lns_target_sol,
        rng,
        destroy_fraction=float(getattr(params, 'lns_destroy_fraction', 0.20)),
        min_destroy=max(1, int(getattr(params, 'lns_min_destroy', 2))),
        max_destroy=max(1, int(getattr(params, 'lns_max_destroy', 12))),
        repair_passes=max(1, int(getattr(params, 'lns_repair_passes', 2))),
        exact_threshold=max(3, int(getattr(params, 'ls_exact_threshold', 10))),
        exact_small_threshold=max(3, int(getattr(params, 'ls_exact_small_threshold', 6))),
        max_neighbors=max(20, int(getattr(params, 'ls_max_neighbors', 120))),
    )
    lns_debug.extend(list(lns_res.debug))

    if not lns_res.improved:
        return (
            best_sol,
            best_score,
            best_any_sol,
            best_any_score,
            update_sol,
            update_score,
            elite_solutions,
            improved_this_iter,
        )

    lns_sol = lns_res.solution
    lns_score = evaluate_ant_penalty(inst, lns_sol, params)

    if best_sol is None or lns_score.total < best_score.total:
        best_sol, best_score = lns_sol, lns_score

    elite_solutions = _push_elite(
        elite_solutions,
        lns_sol,
        inst,
        int(getattr(params, 'sp_elite_size', 8)),
    )

    if best_any_sol is None or lns_score.total < best_any_score.total:
        best_any_sol, best_any_score = lns_sol, lns_score
        acs_history.append({
            'iter': it,
            'ant': -1,
            'best_total': lns_score.total,
            'best_unserved': lns_score.n_unserved,
            'best_dist_lvl2': lns_score.dist_lvl2,
        })
        improved_this_iter = True

    if params.global_update_rule == 'global_best':
        update_sol, update_score = best_sol, best_score
    elif lns_score.total < update_score.total:
        update_sol, update_score = lns_sol, lns_score

    return (
        best_sol,
        best_score,
        best_any_sol,
        best_any_score,
        update_sol,
        update_score,
        elite_solutions,
        improved_this_iter,
    )


def _maybe_run_route_pool_recombination(
    inst: Instance2E,
    params: ACSParams,
    rng: random.Random,
    it: int,
    stagnation_iters: int,
    best_sol: Optional[Solution2E],
    best_score: Optional[AntPenaltyScore],
    update_sol: Optional[Solution2E],
    update_score: Optional[AntPenaltyScore],
    best_any_sol: Optional[Solution2E],
    best_any_score: Optional[AntPenaltyScore],
    elite_solutions: List[Solution2E],
    acs_history: List[Dict[str, float]],
    sp_debug: List[str],
    improved_this_iter: bool,
) -> Tuple[
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    Optional[Solution2E],
    Optional[AntPenaltyScore],
    List[Solution2E],
    bool,
]:
    """Runs elite route-pool recombination when its periodic or stagnation trigger fires."""
    trigger_sp_periodic = int(getattr(params, 'sp_period', 0)) > 0 and (
        it % max(1, int(getattr(params, 'sp_period', 1))) == 0
    )
    trigger_sp_stagnation = int(getattr(params, 'sp_stagnation_trigger', 0)) > 0 and (
        stagnation_iters >= int(getattr(params, 'sp_stagnation_trigger', 0))
    )

    if (
        update_sol is None
        or update_score is None
        or not elite_solutions
        or not (trigger_sp_periodic or trigger_sp_stagnation)
    ):
        return (
            best_sol,
            best_score,
            best_any_sol,
            best_any_score,
            update_sol,
            update_score,
            elite_solutions,
            improved_this_iter,
        )

    # Recombine the current target with elite solutions stored in the route pool
    sp_target_sol = best_sol if best_sol is not None else update_sol
    sp_res = recombine_route_pool(
        inst,
        sp_target_sol,
        elite_solutions,
        exact_threshold=max(3, int(getattr(params, 'ls_exact_threshold', 10))),
        exact_small_threshold=max(3, int(getattr(params, 'ls_exact_small_threshold', 6))),
        max_pool_routes=max(20, int(getattr(params, 'sp_max_pool_routes', 160))),
        max_routes=int(inst.nv2),
        rng=rng,
    )
    sp_debug.extend(list(sp_res.debug))

    if not sp_res.improved:
        return (
            best_sol,
            best_score,
            best_any_sol,
            best_any_score,
            update_sol,
            update_score,
            elite_solutions,
            improved_this_iter,
        )

    sp_sol = sp_res.solution
    sp_score = evaluate_ant_penalty(inst, sp_sol, params)

    elite_solutions = _push_elite(
        elite_solutions,
        sp_sol,
        inst,
        int(getattr(params, 'sp_elite_size', 8)),
    )

    if best_sol is None or sp_score.total < best_score.total:
        best_sol, best_score = sp_sol, sp_score

    if best_any_sol is None or sp_score.total < best_any_score.total:
        best_any_sol, best_any_score = sp_sol, sp_score
        acs_history.append({
            'iter': it,
            'ant': -2,
            'best_total': sp_score.total,
            'best_unserved': sp_score.n_unserved,
            'best_dist_lvl2': sp_score.dist_lvl2,
        })
        improved_this_iter = True

    if params.global_update_rule == 'global_best':
        update_sol, update_score = best_sol, best_score
    elif sp_score.total < update_score.total:
        update_sol, update_score = sp_sol, sp_score

    return (
        best_sol,
        best_score,
        best_any_sol,
        best_any_score,
        update_sol,
        update_score,
        elite_solutions,
        improved_this_iter,
    )


def _apply_global_pheromone_update(
    inst: Instance2E,
    ph: Pheromones,
    params: ACSParams,
    update_sol: Optional[Solution2E],
    update_score: Optional[AntPenaltyScore],
) -> None:
    """Applies the global pheromone reinforcement on route arcs, satellite starts, and assignment arcs."""
    if update_sol is None or update_score is None:
        return

    delta = 1.0 / max(1.0, update_score.total)

    # Reinforce the route starts, second-level arcs, and assignment decisions of the update solution
    for s in inst.satellite_ids:
        if update_sol.routes_lvl2.get(s):
            ph.global_reinforce(START_NODE_ID, s, delta)

    for i, j in update_sol.all_arcs_lvl2():
        ph.global_reinforce(i, j, delta)

    for c, s in update_sol.assignment.items():
        ph.global_reinforce_assign(int(c), int(s), delta * 1.75)


def _post_acs_time_budgets(inst: Instance2E) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[int], Optional[int]]:
    """Returns the post-ACS time budgets and repair limits according to the instance size."""
    n_clients_total = len(inst.clients)
    heavy_postacs = n_clients_total >= 80

    post_ls_time_budget = 6.0 if heavy_postacs else None
    post_sp_time_budget = 4.0 if heavy_postacs else None
    post_abs_time_budget = 12.0 if heavy_postacs else None
    post_abs_sat_budget = 2.5 if heavy_postacs else None
    post_abs_max_clients = 34 if heavy_postacs else None
    post_abs_max_routes = 5 if heavy_postacs else None

    return (
        post_ls_time_budget,
        post_sp_time_budget,
        post_abs_time_budget,
        post_abs_sat_budget,
        post_abs_max_clients,
        post_abs_max_routes,
    )


def _post_acs_improve_lvl2(
    inst: Instance2E,
    params: ACSParams,
    rng: random.Random,
    repaired_lvl2: Solution2E,
    repaired_score: AntPenaltyScore,
    elite_solutions: List[Solution2E],
    sp_debug: List[str],
    post_ls_time_budget: Optional[float],
    post_sp_time_budget: Optional[float],
) -> Tuple[Solution2E, AntPenaltyScore, List[Solution2E], object]:
    """
    Improves the repaired second-level solution through recharge optimization, 
    local search, and final route-pool recombination.
    """
    pre_ls_recharge = optimize_solution_recharges(inst, repaired_lvl2)
    if pre_ls_recharge.improved:
        repaired_lvl2 = pre_ls_recharge.solution
        repaired_score = evaluate_ant_penalty(inst, repaired_lvl2, params)

    # Intensify the repaired second level with local search
    ls_result = intensify_lvl2_solution(
        inst,
        repaired_lvl2,
        max_passes=max(0, int(getattr(params, "ls_max_passes", 0))),
        exact_threshold=max(3, int(getattr(params, "ls_exact_threshold", 10))),
        exact_small_threshold=max(3, int(getattr(params, "ls_exact_small_threshold", 6))),
        max_neighbors=max(20, int(getattr(params, "ls_max_neighbors", 120))),
        rng=rng,
        recharge_every=1,
        time_budget_s=post_ls_time_budget,
    )
    improved_lvl2 = ls_result.solution if ls_result.improved else repaired_lvl2

    elite_solutions = _push_elite(
        elite_solutions,
        improved_lvl2,
        inst,
        int(getattr(params, 'sp_elite_size', 8)),
    )

    # Recombine the improved solution with the elite pool one last time before hybrid finalization
    sp_final_res = recombine_route_pool(
        inst,
        improved_lvl2,
        elite_solutions,
        exact_threshold=max(3, int(getattr(params, 'ls_exact_threshold', 10))),
        exact_small_threshold=max(3, int(getattr(params, 'ls_exact_small_threshold', 6))),
        max_pool_routes=max(20, int(getattr(params, 'sp_max_pool_routes', 160))),
        max_routes=int(inst.nv2),
        rng=rng,
        time_budget_s=post_sp_time_budget,
    )
    sp_debug.extend(list(sp_final_res.debug))
    if sp_final_res.improved:
        improved_lvl2 = sp_final_res.solution

    improved_score = evaluate_ant_penalty(inst, improved_lvl2, params)
    return improved_lvl2, improved_score, elite_solutions, ls_result


def _finalize_and_select_hybrid_base(
    inst: Instance2E,
    params: ACSParams,
    lambda_sync: float,
    repaired_lvl2: Solution2E,
    improved_lvl2: Solution2E,
) -> Tuple[Solution2E, float, str, float, float]:
    """Finalizes the repaired and improved level-2 solutions and selects the better hybrid base solution."""
    final_from_repaired, total_from_repaired = _finalize_hybrid(inst, repaired_lvl2, params, lambda_sync)
    recharge_repaired = optimize_solution_recharges(inst, final_from_repaired)
    if recharge_repaired.improved:
        final_from_repaired = recharge_repaired.solution
        final_from_repaired, total_from_repaired = _finalize_hybrid(
            inst, final_from_repaired, params, lambda_sync
        )

    final_from_improved, total_from_improved = _finalize_hybrid(inst, improved_lvl2, params, lambda_sync)
    recharge_improved = optimize_solution_recharges(inst, final_from_improved)
    if recharge_improved.improved:
        final_from_improved = recharge_improved.solution
        final_from_improved, total_from_improved = _finalize_hybrid(
            inst, final_from_improved, params, lambda_sync
        )

    # Keep the better hybrid solution between the post-repair and post-LS branches
    if total_from_improved + 1e-9 < total_from_repaired:
        final_base = final_from_improved
        final_total = total_from_improved
        final_choice = "after_ls"
    else:
        final_base = final_from_repaired
        final_total = total_from_repaired
        final_choice = "after_repair"

    return final_base, final_total, final_choice, total_from_repaired, total_from_improved


def _maybe_run_absolute_ls(
    inst: Instance2E,
    params: ACSParams,
    lambda_sync: float,
    final_base: Solution2E,
    final_total: float,
    post_abs_time_budget: Optional[float],
    post_abs_sat_budget: Optional[float],
    post_abs_max_clients: Optional[int],
    post_abs_max_routes: Optional[int],
) -> Tuple[Solution2E, float]:
    """
    Runs the absolute-time local search with fixed level 1 and accepts it only 
    if it improves the incumbent.
    """
    abs_ls_result = intensify_absolute_with_fixed_lvl1(
        inst,
        final_base,
        max_attempts=max(1, int(params.max_repair_attempts)),
        time_budget_s=post_abs_time_budget,
        per_satellite_budget_s=post_abs_sat_budget,
        max_clients_per_satellite=post_abs_max_clients,
        max_routes_per_satellite=post_abs_max_routes,
    )

    if not abs_ls_result.improved:
        return final_base, final_total

    cand = abs_ls_result.solution
    cand.meta['dist_lvl1'] = float(cand.meta.get('dist_lvl1', 0.0)) or float(
        sum(distance_of_route(inst, r.nodes) for r in cand.routes_lvl1)
    )
    cand.meta['dist_lvl2'] = float(_lvl2_distance(inst, cand))
    cand.meta['total_hybrid'] = float(
        cand.meta['dist_lvl1']
        + cand.meta['dist_lvl2']
        + float(lambda_sync) * float(cand.meta.get('penalty_tw', 0.0))
        + float(params.lambda_unserved) * len(cand.meta.get('unserved_clients', []))
        + float(cand.meta.get('penalty_misc', 0.0))
    )

    if float(cand.meta['total_hybrid']) + 1e-9 < final_total:
        return cand, float(cand.meta['total_hybrid'])

    return final_base, final_total


def _maybe_run_final_recharge(
    inst: Instance2E,
    params: ACSParams,
    lambda_sync: float,
    final_sol: Solution2E,
    final_total: float,
) -> Tuple[Solution2E, float]:
    """Runs a last recharge optimization pass and accepts it only if it improves the final objective."""
    final_recharge = optimize_solution_recharges(inst, final_sol)
    if not final_recharge.improved:
        return final_sol, final_total

    cand = final_recharge.solution
    cand.meta['dist_lvl1'] = float(cand.meta.get('dist_lvl1', 0.0)) or float(
        sum(distance_of_route(inst, r.nodes) for r in cand.routes_lvl1)
    )
    cand.meta['dist_lvl2'] = float(_lvl2_distance(inst, cand))
    cand.meta['total_hybrid'] = float(
        cand.meta['dist_lvl1']
        + cand.meta['dist_lvl2']
        + float(lambda_sync) * float(cand.meta.get('penalty_tw', 0.0))
        + float(params.lambda_unserved) * len(cand.meta.get('unserved_clients', []))
        + float(cand.meta.get('penalty_misc', 0.0))
    )

    if float(cand.meta['total_hybrid']) + 1e-9 < final_total:
        return cand, float(cand.meta['total_hybrid'])

    return final_sol, final_total


def _store_run_acs_metadata(
    final_sol: Solution2E,
    seed: int,
    best_any_score: AntPenaltyScore,
    repaired_score: AntPenaltyScore,
    improved_score: AntPenaltyScore,
    final_choice: str,
    total_from_repaired: float,
    total_from_improved: float,
    acs_history: List[Dict[str, float]],
    repair_debug: List[str],
    ls_result,
    lns_debug: List[str],
    sp_debug: List[str],
) -> None:
    """Stores the full ACS, repair, and post-processing summary in the final solution metadata."""
    final_sol.meta.update({
        'seed': int(seed),
        'acs_history': acs_history,
        'acs_best_before_repair': {
            'total': float(best_any_score.total),
            'dist_lvl2': float(best_any_score.dist_lvl2),
            'n_unserved': int(best_any_score.n_unserved),
            'route_count': int(best_any_score.route_count),
        },
        'acs_best_after_repair': {
            'total': float(repaired_score.total),
            'dist_lvl2': float(repaired_score.dist_lvl2),
            'n_unserved': int(repaired_score.n_unserved),
            'route_count': int(repaired_score.route_count),
        },
        'acs_best_after_ls': {
            'total': float(improved_score.total),
            'dist_lvl2': float(improved_score.dist_lvl2),
            'n_unserved': int(improved_score.n_unserved),
            'route_count': int(improved_score.route_count),
        },
        'final_choice': final_choice,
        'final_total_after_repair': float(total_from_repaired),
        'final_total_after_ls': float(total_from_improved),
        'post_acs_debug': list(repair_debug) + list(ls_result.debug) + list(lns_debug) + list(sp_debug),
        'bilevel_assignment_pheromones': True,
    })


def run_acs_penalty(inst: Instance2E, params: ACSParams, lambda_sync: float = 1000.0, seed: int = 0) -> Tuple[Solution2E, float]:
    """
    Runs the full penalized ACS workflow, including seed construction, adaptive
    intensification, repair, and hybrid finalization.
    """
    params = _prepare_acs_params_for_instance(inst, params)
    rng, ph, hm, retry, gen = _initialize_acs_state(inst, params, seed)

    best_any_sol: Optional[Solution2E] = None
    best_any_score: Optional[AntPenaltyScore] = None
    best_sol: Optional[Solution2E] = None
    best_score: Optional[AntPenaltyScore] = None
    acs_history: List[Dict[str, float]] = []
    lns_debug: List[str] = []
    sp_debug: List[str] = []
    stagnation_iters = 0
    elite_solutions: List[Solution2E] = []

    # Start from a fast seed solution when possible and use it to initialize the search state
    best_any_sol, best_any_score, best_sol, best_score, elite_solutions = _try_initialize_seed_solution(
        inst, params, gen, ph, elite_solutions, acs_history
    )

    # Main ACS loop with optional periodic LNS and elite-pool recombination
    for it in range(1, int(params.Imax) + 1):
        (
            iter_best_sol,
            iter_best_score,
            best_any_sol,
            best_any_score,
            best_sol,
            best_score,
            improved_this_iter,
        ) = _run_one_acs_iteration(
            inst, ph, hm, params, rng, gen, it,
            best_any_sol, best_any_score, best_sol, best_score, acs_history
        )

        elite_solutions = _push_elite(
            elite_solutions,
            iter_best_sol,
            inst,
            int(getattr(params, 'sp_elite_size', 8)),
        )

        update_sol, update_score = _select_pheromone_update_target(
            params, best_sol, best_score, iter_best_sol, iter_best_score
        )

        (
            best_sol,
            best_score,
            best_any_sol,
            best_any_score,
            update_sol,
            update_score,
            elite_solutions,
            improved_this_iter,
        ) = _maybe_run_periodic_lns(
            inst, params, rng, it, stagnation_iters,
            best_sol, best_score,
            update_sol, update_score,
            best_any_sol, best_any_score,
            elite_solutions, acs_history, lns_debug, improved_this_iter
        )

        (
            best_sol,
            best_score,
            best_any_sol,
            best_any_score,
            update_sol,
            update_score,
            elite_solutions,
            improved_this_iter,
        ) = _maybe_run_route_pool_recombination(
            inst, params, rng, it, stagnation_iters,
            best_sol, best_score,
            update_sol, update_score,
            best_any_sol, best_any_score,
            elite_solutions, acs_history, sp_debug, improved_this_iter
        )

        _apply_global_pheromone_update(inst, ph, params, update_sol, update_score)
        stagnation_iters = 0 if improved_this_iter else (stagnation_iters + 1)

    assert best_any_sol is not None and best_any_score is not None

    # Repair the best penalized second-level solution before launching heavier post-processing
    repaired_lvl2, repair_debug, repair_ok = _repair_best_lvl2(inst, best_any_sol, params)
    repaired_score = evaluate_ant_penalty(inst, repaired_lvl2, params)

    (
        post_ls_time_budget,
        post_sp_time_budget,
        post_abs_time_budget,
        post_abs_sat_budget,
        post_abs_max_clients,
        post_abs_max_routes,
    ) = _post_acs_time_budgets(inst)

    improved_lvl2, improved_score, elite_solutions, ls_result = _post_acs_improve_lvl2(
        inst,
        params,
        rng,
        repaired_lvl2,
        repaired_score,
        elite_solutions,
        sp_debug,
        post_ls_time_budget,
        post_sp_time_budget,
    )

    # Finalize the hybrid solution from both branches and keep the best base candidate
    final_base, final_total, final_choice, total_from_repaired, total_from_improved = _finalize_and_select_hybrid_base(
        inst, params, lambda_sync, repaired_lvl2, improved_lvl2
    )

    # Apply the last absolute-time and recharge improvements on top of the selected hybrid base
    final_sol, final_total = _maybe_run_absolute_ls(
        inst,
        params,
        lambda_sync,
        final_base,
        final_total,
        post_abs_time_budget,
        post_abs_sat_budget,
        post_abs_max_clients,
        post_abs_max_routes,
    )
    final_sol, final_total = _maybe_run_final_recharge(
        inst, params, lambda_sync, final_sol, final_total
    )

    # Store the global run summary and return the final solution
    _store_run_acs_metadata(
        final_sol,
        seed,
        best_any_score,
        repaired_score,
        improved_score,
        final_choice,
        total_from_repaired,
        total_from_improved,
        acs_history,
        repair_debug,
        ls_result,
        lns_debug,
        sp_debug,
    )

    print(
        f"[FINAL] total={final_total:.2f} "
        f"dist1={final_sol.meta.get('dist_lvl1', 0.0):.2f} "
        f"dist2={final_sol.meta.get('dist_lvl2', 0.0):.2f} "
        f"tw_pen={final_sol.meta.get('penalty_tw', 0.0):.2f}"
    )
    return final_sol, float(final_total)