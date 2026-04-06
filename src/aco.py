from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
import random
import heapq

from .instance import Instance2E
from .solution import Solution2E, Route, VehicleIdGenerator

RETURN_TOKEN = -10_000_000
START_NODE_ID = 10_000_000_007
BIG_PENALTY = 1e12
ACS_DEBUG = True


@dataclass
class ACSParams:
    """
    Stores all parameters controlling the ACS algorithm and its extensions
    (assignment, local search, LNS, set partitioning, etc.).
    This acts as the global configuration of the solver.
    """

    m: int = 20
    Imax: int = 80
    alpha: float = 2.0
    beta: float = 2.0
    q0: float = 0.85
    rho: float = 0.20
    xi: float = 0.10
    tau0: float = 1e-4
    eps_dist: float = 1e-9
    heuristic_w_dist: float = 1.0
    heuristic_w_late: float = 2.0
    heuristic_w_energy: float = 1.0
    heuristic_w_risk: float = 2.5
    lambda_unserved: float = 5e4
    penalty_repair_fail: float = 2e4
    max_repair_attempts: int = 6
    max_ant_rebuild_attempts: int = 4
    seed_pheromone_boost: float = 12.0
    max_route_station_moves: int = 40
    max_exact_labels: int = 50000
    max_global_dfs_states: int = 120000
    global_update_rule: str = "iter_best"
    ls_max_passes: int = 12
    ls_exact_threshold: int = 10
    ls_exact_small_threshold: int = 6
    ls_max_neighbors: int = 120
    lns_period: int = 3
    lns_stagnation_trigger: int = 2
    lns_destroy_fraction: float = 0.20
    lns_min_destroy: int = 2
    lns_max_destroy: int = 12
    lns_repair_passes: int = 2
    sp_period: int = 4
    sp_stagnation_trigger: int = 3
    sp_max_pool_routes: int = 160
    sp_elite_size: int = 8
    assign_alpha: float = 1.5
    assign_beta: float = 2.0
    assign_q0: float = 0.80
    assign_rho: float = 0.18
    assign_xi: float = 0.08
    assign_candidate_sats: int = 3


@dataclass
class RetryParams:
    """
    Defines the retry strategy when an ant fails to build a feasible solution.
    Used to guide diversification and penalize bad structures.
    """
    max_cluster_attempts: int = 8
    heuristic_evap_on_fail: float = 0.25
    min_heur: float = 1e-6


class Pheromones:
    # Function summary:   init  .
    def __init__(self, inst: Instance2E, tau0: float):
        """
        Initializes pheromone matrices for routing arcs and client-to-satellite assignments.
        """
        self.inst = inst
        self.n = inst.n_nodes
        self.start_idx = self.n
        
        # initialize arc pheromones
        size = self.n + 1
        self.tau = [[float(tau0) for _ in range(size)] for __ in range(size)]
        self.tau0 = float(tau0)
        
        # build mapping for assignment pheromones
        sats = [int(s) for s in sorted(inst.satellite_ids)]
        clients = [int(c) for c in sorted(inst.client_ids)]
        self.sat_to_pos = {s:i for i,s in enumerate(sats)}
        self.client_to_pos = {c:i for i,c in enumerate(clients)}
        
        # initialize assignment pheromones
        self.assign_tau = [[float(tau0) for _ in range(len(sats))] for __ in range(len(clients))]

    def _to_idx(self, node_id: int) -> int:
        if node_id == self.start_node_id():
            return self.start_idx
        return self.inst.idx(node_id)

    def start_node_id(self) -> int:
        return START_NODE_ID

    def get(self, i: int, j: int) -> float:
        return float(self.tau[self._to_idx(i)][self._to_idx(j)])

    def set(self, i: int, j: int, val: float) -> None:
        self.tau[self._to_idx(i)][self._to_idx(j)] = float(val)

    def local_update(self, i: int, j: int, xi: float) -> None:
        """
        Applies ACS local pheromone update on arc (i → j).
        Encourages exploration by pushing pheromone toward tau0.
        """
        cur = self.get(i, j)
        self.set(i, j, (1.0 - xi) * cur + xi * self.tau0)

    def global_evap(self, i: int, j: int, rho: float) -> None:
        cur = self.get(i, j)
        self.set(i, j, max(self.tau0, (1.0 - rho) * cur))

    def global_reinforce(self, i: int, j: int, delta_tau: float) -> None:
        """
        Reinforces pheromone on arc (i → j) based on solution quality.
        """
        cur = self.get(i, j)
        self.set(i, j, cur + float(delta_tau))

    def get_assign(self, client_id: int, sat_id: int) -> float:
        return float(self.assign_tau[self.client_to_pos[int(client_id)]][self.sat_to_pos[int(sat_id)]])

    def set_assign(self, client_id: int, sat_id: int, val: float) -> None:
        self.assign_tau[self.client_to_pos[int(client_id)]][self.sat_to_pos[int(sat_id)]] = float(val)

    def local_update_assign(self, client_id: int, sat_id: int, xi: float) -> None:
        cur = self.get_assign(client_id, sat_id)
        self.set_assign(client_id, sat_id, (1.0 - float(xi)) * cur + float(xi) * self.tau0)

    def global_evap_assign(self, client_id: int, sat_id: int, rho: float) -> None:
        cur = self.get_assign(client_id, sat_id)
        self.set_assign(client_id, sat_id, max(self.tau0, (1.0 - float(rho)) * cur))

    def global_reinforce_assign(self, client_id: int, sat_id: int, delta_tau: float) -> None:
        cur = self.get_assign(client_id, sat_id)
        self.set_assign(client_id, sat_id, cur + float(delta_tau))


class HeuristicMemory:
    def __init__(self, inst: Instance2E, h0: float = 1.0):
        self.inst = inst
        self.n = inst.n_nodes
        self.start_idx = self.n
        size = self.n + 1
        self.hmem = [[float(h0) for _ in range(size)] for __ in range(size)]
        self.h0 = float(h0)

    def _to_idx(self, node_id: int) -> int:
        if node_id == START_NODE_ID:
            return self.start_idx
        return self.inst.idx(node_id)

    def get(self, i: int, j: int) -> float:
        return float(self.hmem[self._to_idx(i)][self._to_idx(j)])


def _dist(inst: Instance2E, i: int, j: int) -> float:
    """
    Returns the distance between two nodes using the distance matrix.
    """
    return float(inst.require_dist()[inst.idx(i)][inst.idx(j)])


def _sat_start(inst: Instance2E, sat_id: int) -> float:
    """
    Returns the proxy start time from depot to satellite.
    """
    return _dist(inst, inst.depot_id, sat_id)


def _assignment_visibility(inst: Instance2E, client_id: int, sat_id: int) -> float:
    """
    Computes the heuristic visibility of assigning a client to a given satellite.
    A higher value means a more promising client-to-satellite assignment.
    """
    d_sc = _dist(inst, sat_id, client_id)
    tw_center = 0.5 * (float(inst.tw_early(client_id)) + float(inst.tw_late(client_id)))
    dep_shift = abs(tw_center - _sat_start(inst, sat_id))
    return 1.0 / (1.0 + d_sc + 0.15 * dep_shift)


def _assignment_desirability(inst: Instance2E, ph: Pheromones, p: ACSParams, client_id: int, sat_id: int) -> float:
    """
    Combines assignment pheromone and assignment visibility into a single desirability score.
    This is the ACS value used to evaluate how attractive it is to assign a client
    to a specific satellite.
    """
    tau = ph.get_assign(client_id, sat_id) ** float(getattr(p, 'assign_alpha', 1.0))
    eta = _assignment_visibility(inst, client_id, sat_id) ** float(getattr(p, 'assign_beta', 1.0))
    return tau * eta


def _rank_satellites_for_customer(inst: Instance2E, ph: Pheromones, p: ACSParams, client_id: int) -> List[int]:
    """
    Ranks satellites for one client according to assignment desirability.
    Satellites are sorted primarily by ACS desirability, then by direct distance,
    and finally by id for deterministic tie-breaking. Only the best candidate
    satellites are kept.
    """
    ranked = sorted(
        [int(s) for s in inst.satellite_ids],
        key=lambda s: (-_assignment_desirability(inst, ph, p, client_id, s), _dist(inst, s, client_id), s),
    )
    limit = max(1, min(len(ranked), int(getattr(p, 'assign_candidate_sats', 3))))
    return ranked[:limit]


def _cluster_centroid(inst: Instance2E, clients: List[int]) -> Tuple[float, float]:
    """
    Computes the geometric centroid of a client set.
    This is used to estimate how compact a satellite cluster is.
    """
    if not clients:
        return 0.0, 0.0
    xs = [float(inst.node_by_id[c].x) for c in clients]
    ys = [float(inst.node_by_id[c].y) for c in clients]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _assignment_proxy_cost(
    inst: Instance2E,
    partial_assign: Dict[int, int],
    sat_clients: Dict[int, List[int]],
    sat_loads: Dict[int, float],
    client_id: int,
    sat_id: int,
) -> float:
    """
    Estimates the structural cost of assigning a client to a satellite.

    This proxy cost is not the exact routing cost. Instead, it approximates how good
    or bad the assignment is from several perspectives:
    - direct satellite-to-client distance,
    - consistency with the client's time window,
    - compactness of the existing satellite cluster,
    - load pressure on the satellite,
    - risk of creating too many second-echelon routes,
    - and a small first-echelon distance penalty.

    Lower values indicate more promising assignments.
    """
    # Direct distance between the satellite and the client.
    d_sc = _dist(inst, sat_id, client_id)
    
    # Temporal mismatch between the satellite start and the client's preferred service period.
    tw_mid = 0.5 * (float(inst.tw_early(client_id)) + float(inst.tw_late(client_id)))
    depot_shift = abs(tw_mid - _sat_start(inst, sat_id))
    
    # Current clients already assigned to this satellite.
    existing = sat_clients.get(int(sat_id), [])
    compactness = 0.0
    
    # If the cluster already exists, estimate how far the client is from its centroid.
    if existing:
        cx, cy = _cluster_centroid(inst, existing)
        compactness = ((float(inst.node_by_id[client_id].x) - cx) ** 2 + (float(inst.node_by_id[client_id].y) - cy) ** 2) ** 0.5
    
    # Current load already assigned to the satellite.
    sat_load = float(sat_loads.get(int(sat_id), 0.0))
    
    # Computing of pressure
    load_pressure = max(0.0, sat_load + float(inst.node_by_id[client_id].demand) - float(inst.Q2))
    route_pressure = max(0.0, (sat_load + float(inst.node_by_id[client_id].demand)) / max(1.0, float(inst.Q2)) - 0.85)
    
    # Small proxy for first-echelon cost: depot to satellite distance.
    lvl1_proxy = _dist(inst, inst.depot_id, sat_id)
    
    # Weighted sum used as structural assignment proxy cost.
    return d_sc + 0.15 * depot_shift + 0.30 * compactness + 3.0 * load_pressure + 6.0 * route_pressure + 0.08 * lvl1_proxy


def _choose_satellite_for_client(
    inst: Instance2E,
    ph: Pheromones,
    params: ACSParams,
    rng: random.Random,
    client_id: int,
    partial_assign: Dict[int, int],
    sat_clients: Dict[int, List[int]],
    sat_loads: Dict[int, float],
) -> int:
    """
    Selects a satellite for a client using ACS decision rule (greedy + probabilistic).
    """
    # get candidate satellites
    candidates = _rank_satellites_for_customer(inst, ph, params, client_id)
    
    scored: List[Tuple[int, float]] = []
    for sat in candidates:
        # pheromone + heuristic desirability
        desir = _assignment_desirability(inst, ph, params, client_id, sat)
        # proxy cost (capacity, compactness, time)
        proxy = _assignment_proxy_cost(inst, partial_assign, sat_clients, sat_loads, client_id, sat)
        val = max(1e-12, desir / max(1e-9, proxy))
        scored.append((int(sat), float(val)))
    
    # fallback: nearest satellite
    if not scored:
        return int(min(inst.satellite_ids, key=lambda s: _dist(inst, s, client_id)))
    
    # exploitation vs exploration
    q0_assign = float(getattr(params, 'assign_q0', params.q0))
    if rng.random() <= q0_assign:
        return max(scored, key=lambda kv: kv[1])[0]
    
    # roulette selection
    total = sum(v for _, v in scored)
    if total <= 0.0:
        return scored[0][0]
    pick = rng.random() * total
    acc = 0.0
    for sat, val in scored:
        acc += val
        if acc >= pick:
            return sat
    return scored[-1][0]


def _client_assignment_difficulty(
    inst: Instance2E,
    ph: Pheromones,
    params: ACSParams,
    client_id: int,
) -> Tuple[float, float, float, int]:
    """
    Returns a priority key used to sort clients before the assignment phase.

    Clients with tighter time windows, larger demand, and fewer good satellite options
    are processed first so that difficult decisions are taken earlier.
    """
    ranked = _rank_satellites_for_customer(inst, ph, params, client_id)
    best_d = min(
        (_dist(inst, s, client_id) for s in ranked),
        default=min(_dist(inst, s, client_id) for s in inst.satellite_ids),
    )
    slack = float(inst.tw_late(client_id)) - float(inst.tw_early(client_id))
    return (slack, -best_d, -float(inst.node_by_id[client_id].demand), int(client_id))


def _ordered_clients_for_assignment(
    inst: Instance2E,
    ph: Pheromones,
    params: ACSParams,
) -> List[int]:
    """
    Builds the ordered client list used by the assignment phase.
    """
    return sorted(
        inst.client_ids,
        key=lambda c: _client_assignment_difficulty(inst, ph, params, int(c)),
    )


def _initialize_assignment_state(
    inst: Instance2E,
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, float], List[str]]:
    """
    Initializes the data structures used during the client-to-satellite assignment phase.
    """
    assignment: Dict[int, int] = {}
    sat_clients: Dict[int, List[int]] = {int(s): [] for s in inst.satellite_ids}
    sat_loads: Dict[int, float] = {int(s): 0.0 for s in inst.satellite_ids}
    debug_msgs: List[str] = []
    return assignment, sat_clients, sat_loads, debug_msgs


def _assign_one_client_to_satellite(
    inst: Instance2E,
    ph: Pheromones,
    params: ACSParams,
    rng: random.Random,
    client_id: int,
    assignment: Dict[int, int],
    sat_clients: Dict[int, List[int]],
    sat_loads: Dict[int, float],
) -> None:
    """
    Assigns one client to one satellite, updates the assignment structures,
    and applies the local pheromone update on the chosen assignment.
    """
    sat = _choose_satellite_for_client(
        inst,
        ph,
        params,
        rng,
        int(client_id),
        assignment,
        sat_clients,
        sat_loads,
    )

    assignment[int(client_id)] = int(sat)
    sat_clients[int(sat)].append(int(client_id))
    sat_loads[int(sat)] += float(inst.node_by_id[client_id].demand)

    ph.local_update_assign(
        int(client_id),
        int(sat),
        float(getattr(params, "assign_xi", params.xi)),
    )


def _sort_clients_within_satellites(
    inst: Instance2E,
    sat_clients: Dict[int, List[int]],
) -> None:
    """
    Sorts assigned clients inside each satellite cluster so that urgent and nearby clients
    appear earlier in the local ordering.
    """
    for sat in sat_clients:
        sat_clients[sat].sort(
            key=lambda c: (
                float(inst.tw_late(c)),
                float(inst.tw_early(c)),
                _dist(inst, sat, c),
                c,
            )
        )


def _finalize_assignment_debug(
    inst: Instance2E,
    assignment: Dict[int, int],
    sat_clients: Dict[int, List[int]],
    debug_msgs: List[str],
) -> None:
    """
    Appends the final summary message for the assignment phase.
    """
    debug_msgs.append(
        f"phase_assignment_done served={len(assignment)}/{len(inst.client_ids)} "
        f"nonempty_sats={sum(1 for s in sat_clients.values() if s)}"
    )


def _construct_assignment_one_ant(
    inst: Instance2E,
    ph: Pheromones,
    params: ACSParams,
    rng: random.Random,
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, float], List[str]]:
    """
    Builds the client-to-satellite assignment for one ant.

    The procedure follows three main steps:
    1. initialize the assignment structures,
    2. sort clients by assignment difficulty and assign them one by one,
    3. reorder clients inside each satellite cluster and record summary debug information.
    """
    # Initialize assignment containers.
    assignment, sat_clients, sat_loads, debug_msgs = _initialize_assignment_state(inst)

    # Process the most difficult clients first.
    ordered_clients = _ordered_clients_for_assignment(inst, ph, params)
    for client_id in ordered_clients:
        _assign_one_client_to_satellite(
            inst,
            ph,
            params,
            rng,
            int(client_id),
            assignment,
            sat_clients,
            sat_loads,
        )

    # Reorder each satellite cluster after the assignment phase.
    _sort_clients_within_satellites(inst, sat_clients)

    # Store a compact summary of the phase.
    _finalize_assignment_debug(inst, assignment, sat_clients, debug_msgs)

    return assignment, sat_clients, sat_loads, debug_msgs


def _choose_next_assigned_customer(
    inst: Instance2E,
    ph: Pheromones,
    hm: HeuristicMemory,
    params: ACSParams,
    rng: random.Random,
    sat_id: int,
    cur: int,
    battery: float,
    load: float,
    t: float,
    assigned_left: Set[int],
) -> Optional[int]:
    """
    Selects the next customer to visit among the remaining assigned customers.

    The decision follows the ACS rule:
    - first filter feasible customers (capacity, battery, time windows),
    - then rank them using pheromone, heuristic visibility, and assignment bias,
    - finally select either greedily or probabilistically (pseudo-random rule).
    """
    # Filter feasible candidates
    feasible = [c for c in assigned_left if _feasible_customer(inst, sat_id, cur, battery, load, t, c)]
    if not feasible:
        return None
    
    # Compute desirability scores
    ranked: List[Tuple[int, float]] = []
    for c in feasible:
        move_score = _score_move(inst, ph, hm, params, sat_id, cur, battery, load, t, c)
        assign_bias = _assignment_desirability(inst, ph, params, c, sat_id)
        ranked.append((int(c), max(1e-12, move_score * assign_bias)))
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    keep = max(1, min(len(ranked), max(3, int(getattr(params, 'assign_candidate_sats', 3)) * 2)))
    ranked = ranked[:keep]
    
    # ACS selection rule (greedy vs probabilistic)
    if rng.random() <= params.q0:
        return max(ranked, key=lambda kv: kv[1])[0]
    
    # Roulette-wheel selection
    total = sum(v for _, v in ranked)
    if total <= 0.0:
        return ranked[0][0]
    pick = rng.random() * total
    acc = 0.0
    for c, v in ranked:
        acc += v
        if acc >= pick:
            return c
    return ranked[-1][0]


def _build_routes_for_assignment(
    inst: Instance2E,
    ph: Pheromones,
    hm: HeuristicMemory,
    params: ACSParams,
    rng: random.Random,
    gen: VehicleIdGenerator,
    sat_clients: Dict[int, List[int]],
) -> Tuple[Dict[int, List[Route]], Dict[int, int], Dict[int, float], Dict[int, float], Set[int], List[str], List[Tuple[int, int]]]:
    """
    Builds EV routes for each satellite after assignment phase.
    """
    
    routes_by_sat: Dict[int, List[Route]] = {int(s): [] for s in inst.satellite_ids}
    assignment_final: Dict[int, int] = {}
    t_rel_arr: Dict[int, float] = {}
    rel_from_sat: Dict[int, float] = {}
    unserved: Set[int] = set()
    debug_msgs: List[str] = []
    used_arcs: List[Tuple[int, int]] = []

    route_budget_left = int(inst.nv2)
    
    # process satellites with largest demand first
    sats_order = sorted(inst.satellite_ids, key=lambda s: (-len(sat_clients.get(int(s), [])), s))
    for sat in sats_order:
        assigned = list(sat_clients.get(int(sat), []))
        if not assigned:
            continue
        remaining = set(int(c) for c in assigned)
        debug_msgs.append(f"phase_routing_sat_start sat={sat} assigned={len(remaining)} budget_left={route_budget_left}")
        while remaining and route_budget_left > 0:
            
            # initialize route
            cur = int(sat)
            t = _sat_start(inst, sat)
            battery = float(inst.BCe)
            load = 0.0
            route_nodes = [int(sat)]
            moved_arcs: List[Tuple[int, int]] = []
            consecutive_station_moves = 0
            while True:
                # try to add next customer
                nxt = _choose_next_assigned_customer(inst, ph, hm, params, rng, int(sat), cur, battery, load, t, remaining)
                if nxt is not None:
                    moved_arcs.append((cur, int(nxt)))
                    cur, battery, load, t = _append_customer(inst, route_nodes, int(sat), cur, int(nxt), battery, load, t, t_rel_arr, rel_from_sat)
                    assignment_final[int(nxt)] = int(sat)
                    remaining.remove(int(nxt))
                    consecutive_station_moves = 0
                    continue
                
                # try charging station
                helpful = _station_helpful(inst, int(sat), cur, battery, load, t, remaining)
                if helpful and (len(route_nodes) == 1 or route_nodes[-1] not in inst.station_ids) and consecutive_station_moves <= int(params.max_route_station_moves):
                    station = min(helpful, key=lambda r: _dist(inst, cur, r))
                    moved_arcs.append((cur, int(station)))
                    cur, battery, t = _append_station(inst, route_nodes, cur, int(station), battery, t)
                    consecutive_station_moves += 1
                    continue
                
                # return to satellite
                ret_path = _return_path(inst, int(sat), cur, battery)
                if ret_path is None:
                    debug_msgs.append(f"phase_routing_cannot_return sat={sat} cur={cur} rem={len(remaining)}")
                    break
                for nxt2 in ret_path:
                    moved_arcs.append((cur, int(nxt2)))
                    if nxt2 in inst.station_ids:
                        cur, battery, t = _append_station(inst, route_nodes, cur, int(nxt2), battery, t)
                    else:
                        route_nodes.append(int(nxt2))
                        t += _dist(inst, cur, int(nxt2))
                        cur = int(nxt2)
                break
            
            # store route
            if len(route_nodes) <= 2:
                break

            route = Route(nodes=route_nodes, vehicle_id=gen.new_ev_id())
            routes_by_sat[int(sat)].append(route)
            used_arcs.extend(moved_arcs)
            for i, j in moved_arcs:
                ph.local_update(i, j, params.xi)
            route_budget_left -= 1

        if remaining:
            unserved.update(remaining)
            debug_msgs.append(f"phase_routing_remaining sat={sat} left={len(remaining)}")

    return routes_by_sat, assignment_final, t_rel_arr, rel_from_sat, unserved, debug_msgs, used_arcs


def _potential_lateness(inst: Instance2E, cur: int, t: float, j: int) -> float:
    """Return the potential tardiness generated by moving to customer j."""
    arr = t + _dist(inst, cur, j)
    return max(0.0, arr - float(inst.tw_late(j)))


def _energy_cost(inst: Instance2E, cur: int, j: int) -> float:
    """Return the energy consumption associated with the move cur -> j."""
    return float(inst.h) * _dist(inst, cur, j)


def _resource_risk(inst: Instance2E, sat_id: int, cur: int, battery: float, load: float, t: float, j: int) -> float:
    """Estimate the risk of exhausting route resources after selecting j."""
    demand_j = float(inst.node_by_id[j].demand)
    dist_to_j = _dist(inst, cur, j)
    battery_after = max(0.0, battery - float(inst.h) * dist_to_j)
    arrival = t + dist_to_j
    service_start = max(arrival, float(inst.tw_early(j)))
    load_after = load + demand_j

    cap_rest = max(float(inst.Q2) - load_after, 1e-6)
    time_rest = max(float(inst.tw_late(j)) - service_start, 1e-6)

    direct_back = max(0.0, battery_after - float(inst.h) * _dist(inst, j, sat_id))
    energy_rest = max(max(direct_back, battery_after), 1e-6)
    return (1.0 / cap_rest) + (1.0 / time_rest) + (1.0 / energy_rest)


def _heuristic_visibility(inst: Instance2E, params: ACSParams, sat_id: int, cur: int, battery: float, load: float, t: float, j: int) -> float:
    """Compute the new generalized visibility defined by the project architecture."""
    generalized_cost = (
        float(params.heuristic_w_dist) * _dist(inst, cur, j)
        + float(params.heuristic_w_late) * _potential_lateness(inst, cur, t, j)
        + float(params.heuristic_w_energy) * _energy_cost(inst, cur, j)
        + float(params.heuristic_w_risk) * _resource_risk(inst, sat_id, cur, battery, load, t, j)
    )
    return 1.0 / max(float(params.eps_dist), generalized_cost)


def _reachable_after(inst: Instance2E, battery: float, node: int, sat_id: int) -> bool:
    """
    Checks whether the vehicle can still reach either the satellite or at least 
    one charging station from the current node given the remaining battery.
    """
    h = float(inst.h)
    if battery - h * _dist(inst, node, sat_id) >= -1e-9:
        return True
    for r in inst.station_ids:
        if battery - h * _dist(inst, node, r) >= -1e-9:
            return True
    return False


def _feasible_customer(inst: Instance2E, sat_id: int, cur: int, battery: float, load: float, t: float, c: int) -> bool:
    """
    Verifies whether visiting a customer is feasible with respect to capacity,
    battery consumption, time windows, and future reachability.
    """
    dem = float(inst.node_by_id[c].demand)
    if load + dem > float(inst.Q2) + 1e-9:
        return False
    leg = _dist(inst, cur, c)
    after = battery - float(inst.h) * leg
    if after < -1e-9:
        return False
    arr = t + leg
    start = max(arr, float(inst.tw_early(c)))
    if start > float(inst.tw_late(c)) + 1e-9:
        return False
    return _reachable_after(inst, after, c, sat_id)


def _station_helpful(inst: Instance2E, sat_id: int, cur: int, battery: float, load: float, t: float, unserved: Set[int]) -> List[int]:
    """
    Identifies charging stations that enable at least one currently unserved 
    customer to become feasible after recharging.
    """
    helpful: List[int] = []
    for r in inst.station_ids:
        leg = _dist(inst, cur, r)
        if r == cur and abs(battery - float(inst.BCe)) <= 1e-9:
            continue
        if leg <= 1e-12 and abs(battery - float(inst.BCe)) <= 1e-9:
            continue
        if battery - float(inst.h) * leg < -1e-9:
            continue
        batt_after = battery - float(inst.h) * leg
        tr = t + leg + float(inst.ge) * max(0.0, float(inst.BCe) - batt_after)
        for c in unserved:
            if _feasible_customer(inst, sat_id, r, float(inst.BCe), load, tr, c):
                helpful.append(r)
                break
    return helpful


def _score_move(inst: Instance2E, ph: Pheromones, hm: HeuristicMemory, p: ACSParams, sat_id: int, cur: int, battery: float, load: float, t: float, candidate: int) -> float:
    """Score a customer move using pheromones and the new weighted visibility."""
    tau = ph.get(cur, candidate) ** p.alpha
    eta = _heuristic_visibility(inst, p, sat_id, cur, battery, load, t, candidate) ** p.beta
    mem = hm.get(cur, candidate)
    return tau * eta * mem


def _return_path(inst: Instance2E, sat_id: int, cur: int, battery: float) -> Optional[List[int]]:
    """
    Computes a feasible return path to the satellite, possibly via a 
    charging station, given the remaining battery.
    """
    h = float(inst.h)
    if battery - h * _dist(inst, cur, sat_id) >= -1e-9:
        return [sat_id]
    best_station = None
    best_extra = float("inf")
    for r in inst.station_ids:
        if battery - h * _dist(inst, cur, r) < -1e-9:
            continue
        if float(inst.BCe) - h * _dist(inst, r, sat_id) < -1e-9:
            continue
        extra = _dist(inst, cur, r) + _dist(inst, r, sat_id)
        if extra < best_extra:
            best_extra = extra
            best_station = r
    if best_station is not None:
        return [best_station, sat_id]
    return None


def _append_station(inst: Instance2E, route_nodes: List[int], cur: int, station: int, battery: float, t: float) -> Tuple[int, float, float]:
    """
    Appends a charging station to the route and updates battery level and time
    after travel and recharging.
    """
    route_nodes.append(station)
    battery -= float(inst.h) * _dist(inst, cur, station)
    t += _dist(inst, cur, station)
    t += float(inst.ge) * max(0.0, float(inst.BCe) - battery)
    return station, float(inst.BCe), t


def _append_customer(inst: Instance2E, route_nodes: List[int], sat_id: int, cur: int, c: int, battery: float, load: float, t: float, t_rel_arr: Dict[int, float], rel_from_sat: Dict[int, float]) -> Tuple[int, float, float, float]:
    """
    Appends a customer to the route and updates battery, load, time, and 
    relative arrival times.
    """
    route_nodes.append(c)
    route_nodes.append(c)
    battery -= float(inst.h) * _dist(inst, cur, c)
    t += _dist(inst, cur, c)
    start = max(t, float(inst.tw_early(c)))
    t_rel_arr[c] = float(start)
    rel_from_sat[c] = float(start - _sat_start(inst, sat_id))
    t = start + float(inst.service_time(c))
    load += float(inst.node_by_id[c].demand)
    return c, battery, load, t


def _extract_clients(route: Route, inst: Instance2E) -> List[int]:
    """Extracts and returns the list of customer nodes from a route, excluding depot and satellite."""
    return [n for n in route.nodes[1:-1] if n in inst.client_ids]


def _best_station_to_target(inst: Instance2E, cur: int, battery: float, target: int, sat_id: int) -> Optional[int]:
    """
    Finds the best charging station enabling a feasible trip to a target node
    while preserving future reachability.
    """
    best = None
    best_extra = float('inf')
    for r in inst.station_ids:
        if battery - float(inst.h) * _dist(inst, cur, r) < -1e-9:
            continue
        if float(inst.BCe) - float(inst.h) * _dist(inst, r, target) < -1e-9:
            continue
        # after target, still need potential return/station
        after_target = float(inst.BCe) - float(inst.h) * _dist(inst, r, target)
        if not _reachable_after(inst, after_target, target, sat_id):
            continue
        extra = _dist(inst, cur, r) + _dist(inst, r, target)
        if extra < best_extra:
            best_extra = extra
            best = r
    return best


def _materialize_route_from_clients(inst: Instance2E, sat_id: int, clients_order: List[int]) -> Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]:
    """
    Builds a complete feasible route from an ordered list of clients, 
    handling battery, time windows, and charging decisions.
    """
    
    if not clients_order:
        return [sat_id, sat_id], {}, {}, 0.0
    
    # Check total capacity feasibility before constructing the route
    total_load = sum(float(inst.node_by_id[c].demand) for c in clients_order)
    if total_load > float(inst.Q2) + 1e-9:
        return None
    route_nodes = [sat_id]
    t_rel_arr: Dict[int, float] = {}
    rel_from_sat: Dict[int, float] = {}
    cur = sat_id
    t = _sat_start(inst, sat_id)
    battery = float(inst.BCe)
    load = 0.0
    
    # Sequentially insert clients while maintaining feasibility (battery, time windows, capacity)
    for c in clients_order:
        if not _feasible_customer(inst, sat_id, cur, battery, load, t, c):
            # Try inserting a charging station if direct insertion is infeasible
            st = _best_station_to_target(inst, cur, battery, c, sat_id)
            if st is None:
                return None
            cur, battery, t = _append_station(inst, route_nodes, cur, st, battery, t)
            if not _feasible_customer(inst, sat_id, cur, battery, load, t, c):
                return None
        cur, battery, load, t = _append_customer(inst, route_nodes, sat_id, cur, c, battery, load, t, t_rel_arr, rel_from_sat)
    
    # Construct a feasible return path to the satellite (possibly via charging station)
    ret = _return_path(inst, sat_id, cur, battery)
    if ret is None:
        return None
    for nxt in ret:
        if nxt in inst.station_ids:
            cur, battery, t = _append_station(inst, route_nodes, cur, nxt, battery, t)
        else:
            route_nodes.append(nxt)
            t += _dist(inst, cur, nxt)
            cur = nxt
    return route_nodes, t_rel_arr, rel_from_sat, distance_of_nodes(inst, route_nodes)


def distance_of_nodes(inst: Instance2E, nodes: List[int]) -> float:
    """Computes the total travel distance of a route defined by a sequence of nodes."""
    return sum(_dist(inst, i, j) for i, j in zip(nodes[:-1], nodes[1:]))


def _subset_is_trivially_infeasible(inst: Instance2E, clients: List[int]) -> bool:
    """Checks whether a client subset is immediately infeasible because it is too large or exceeds vehicle capacity."""
    if len(clients) > 10:
        return True
    total_load = sum(float(inst.node_by_id[c].demand) for c in clients)
    return total_load > float(inst.Q2) + 1e-9


def _label_is_dominated(
    existing_labels: List[Tuple[float, float, int]],
    time_val: float,
    battery_val: float,
) -> bool:
    """Returns whether a new label is dominated by an existing label with no worse time and no lower battery."""
    for t0, b0, _ in existing_labels:
        if t0 <= time_val + 1e-9 and b0 >= battery_val - 1e-9:
            return True
    return False


def _filter_dominated_labels(
    existing_labels: List[Tuple[float, float, int]],
    time_val: float,
    battery_val: float,
) -> List[Tuple[float, float, int]]:
    """Removes existing labels dominated by a new label with better time and battery values."""
    return [
        (t0, b0, lid)
        for (t0, b0, lid) in existing_labels
        if not (t0 >= time_val - 1e-9 and b0 <= battery_val + 1e-9)
    ]


def _add_exact_dp_label(
    labels: Dict[Tuple[int, int], List[Tuple[float, float, int]]],
    prev: Dict[int, Tuple[Optional[int], int, Optional[float]]],
    info: Dict[int, Tuple[int, int, float, float]],
    pq: List[Tuple[float, float, int]],
    next_id_ref: List[int],
    node: int,
    mask: int,
    time_val: float,
    battery_val: float,
    parent: Optional[int],
    raw_arrival: Optional[float],
) -> Optional[int]:
    """Adds a non-dominated DP label and updates the label sets, predecessor map, and priority queue."""
    key = (node, mask)
    current = labels.get(key, [])

    if _label_is_dominated(current, time_val, battery_val):
        return None

    lid = next_id_ref[0]
    next_id_ref[0] += 1

    filtered = _filter_dominated_labels(current, time_val, battery_val)
    filtered.append((time_val, battery_val, lid))
    labels[key] = filtered

    prev[lid] = (parent, node, raw_arrival)
    info[lid] = (node, mask, time_val, battery_val)
    heapq.heappush(pq, (time_val, -battery_val, lid))
    return lid


def _try_extend_to_client(
    inst: Instance2E,
    sat_id: int,
    clients: List[int],
    client_to_bit: Dict[int, int],
    labels: Dict[Tuple[int, int], List[Tuple[float, float, int]]],
    prev: Dict[int, Tuple[Optional[int], int, Optional[float]]],
    info: Dict[int, Tuple[int, int, float, float]],
    pq: List[Tuple[float, float, int]],
    next_id_ref: List[int],
    lid: int,
    node: int,
    mask: int,
    cur_time: float,
    battery: float,
    h: float,
) -> None:
    """Generates feasible client-visit extensions from the current DP label."""
    for c in clients:
        bit = 1 << client_to_bit[c]
        if mask & bit:
            continue

        d = _dist(inst, node, c)
        if battery - h * d < -1e-9:
            continue

        arr = cur_time + d
        start = max(arr, float(inst.tw_early(c)))
        if start > float(inst.tw_late(c)) + 1e-9:
            continue

        time2 = start + float(inst.service_time(c))
        bat2 = battery - h * d
        _add_exact_dp_label(
            labels, prev, info, pq, next_id_ref,
            c, mask | bit, time2, bat2, lid, start
        )


def _try_extend_to_station(
    inst: Instance2E,
    labels: Dict[Tuple[int, int], List[Tuple[float, float, int]]],
    prev: Dict[int, Tuple[Optional[int], int, Optional[float]]],
    info: Dict[int, Tuple[int, int, float, float]],
    pq: List[Tuple[float, float, int]],
    next_id_ref: List[int],
    lid: int,
    node: int,
    mask: int,
    cur_time: float,
    battery: float,
    h: float,
) -> None:
    """Generates feasible charging-station extensions from the current DP label."""
    for r in inst.station_ids:
        if r == node:
            continue

        d = _dist(inst, node, r)
        if battery - h * d < -1e-9:
            continue

        after = battery - h * d
        time2 = cur_time + d + float(inst.ge) * max(0.0, float(inst.BCe) - after)
        bat2 = float(inst.BCe)
        _add_exact_dp_label(
            labels, prev, info, pq, next_id_ref,
            r, mask, time2, bat2, lid, time2
        )


def _reconstruct_exact_route(
    inst: Instance2E,
    sat_id: int,
    prev: Dict[int, Tuple[Optional[int], int, Optional[float]]],
    end_lid: int,
    start_time: float,
) -> Tuple[List[int], Dict[int, float], Dict[int, float], float]:
    """Reconstructs the final route and client arrival-time dictionaries from the best terminal label."""
    seq = []
    lid = end_lid
    while lid is not None:
        parent, node, raw_arr = prev[lid]
        seq.append((node, raw_arr))
        lid = parent
    seq.reverse()

    path_nodes = [sat_id]
    t_rel_arr: Dict[int, float] = {}
    rel_from_sat: Dict[int, float] = {}

    for node, arr in seq[1:]:
        path_nodes.append(node)
        if node in inst.client_ids and arr is not None:
            t_rel_arr[node] = float(arr)
            rel_from_sat[node] = float(arr - start_time)

    path_nodes.append(sat_id)
    return path_nodes, t_rel_arr, rel_from_sat, distance_of_nodes(inst, path_nodes)


def _exact_route_for_subset(inst: Instance2E, sat_id: int, clients: List[int]) -> Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]:
    """Solves a small EVRP subproblem exactly using label-setting dynamic programming."""
    clients = sorted(set(clients))
    if not clients:
        return [sat_id, sat_id], {}, {}, 0.0

    # Reject subsets that are clearly outside the intended exact DP scope
    if _subset_is_trivially_infeasible(inst, clients):
        return None

    client_to_bit = {c: i for i, c in enumerate(clients)}
    full_mask = (1 << len(clients)) - 1
    start_time = _sat_start(inst, sat_id)
    start_battery = float(inst.BCe)
    h = float(inst.h)

    labels: Dict[Tuple[int, int], List[Tuple[float, float, int]]] = {}
    prev: Dict[int, Tuple[Optional[int], int, Optional[float]]] = {}
    info: Dict[int, Tuple[int, int, float, float]] = {}
    pq: List[Tuple[float, float, int]] = []
    next_id_ref = [0]

    # Initialize the DP from the satellite with empty mask and full battery
    _add_exact_dp_label(
        labels, prev, info, pq, next_id_ref,
        sat_id, 0, start_time, start_battery, None, None
    )

    best_end: Optional[Tuple[int, float]] = None
    expanded = 0

    # Explore non-dominated states ordered by time, while pruning oversized searches
    while pq:
        expanded += 1
        if next_id_ref[0] > 50000 or expanded > 100000:
            return None

        _, _, lid = heapq.heappop(pq)
        node, mask, cur_time, battery = info[lid]

        if mask == full_mask:
            dback = _dist(inst, node, sat_id)
            if battery - h * dback >= -1e-9:
                best_end = (lid, cur_time + dback)
                break

        _try_extend_to_client(
            inst, sat_id, clients, client_to_bit,
            labels, prev, info, pq, next_id_ref,
            lid, node, mask, cur_time, battery, h
        )
        _try_extend_to_station(
            inst,
            labels, prev, info, pq, next_id_ref,
            lid, node, mask, cur_time, battery, h
        )

    # Rebuild the exact best route together with relative arrival information
    if best_end is None:
        return None

    end_lid, _ = best_end
    return _reconstruct_exact_route(inst, sat_id, prev, end_lid, start_time)


def _exact_route_any_sat_for_subset(inst: Instance2E, clients: List[int]) -> Optional[Tuple[int, List[int], Dict[int, float], Dict[int, float], float]]:
    """
    Finds the best exact route for a client subset by testing all satellites as
    possible starting points.
    """
    best = None
    
    # Evaluate the exact route obtained from each candidate satellite
    for s in sorted(inst.satellite_ids):
        mat = _exact_route_for_subset(inst, s, clients)
        if mat is None:
            continue
        nodes2, tmap, rmap, dist2 = mat
        score = (dist2, len(nodes2), s)
        
        # Keep the best route according to distance, route size, and satellite id
        if best is None or score < best[0]:
            best = (score, s, nodes2, tmap, rmap, dist2)
    return None if best is None else (best[1], best[2], best[3], best[4], best[5])


def _client_compatibility(inst: Instance2E, a: int, b: int) -> float:
    """
    Computes a compatibility score between two clients based on spatial 
    proximity, time-window similarity, and demand similarity.
    """
    dab = _dist(inst, a, b)

    # Compute stable normalization factors for time windows and demand
    tw_span = max(
        1.0,
        max(float(inst.tw_late(i)) for i in inst.client_ids)
        - min(float(inst.tw_early(i)) for i in inst.client_ids)
    )
    q_scale = max(1.0, float(inst.Q2))

    # Measure differences in time windows and demand
    late_gap = abs(float(inst.tw_late(a)) - float(inst.tw_late(b))) / tw_span
    early_gap = abs(float(inst.tw_early(a)) - float(inst.tw_early(b))) / tw_span
    dem_gap = abs(float(inst.node_by_id[a].demand) - float(inst.node_by_id[b].demand)) / q_scale
    
    # Convert these differences into similarity components
    spatial = 1.0 / (1.0 + dab)
    tw_sim = 1.0 / (1.0 + 0.5 * late_gap + 0.5 * early_gap)
    dem_sim = 1.0 / (1.0 + dem_gap)

    # Combine the components with a stronger emphasis on space, then time windows, then demand
    return 1.00 * spatial + 0.60 * tw_sim + 0.20 * dem_sim


def _route_absorption_score(inst: Instance2E, route_clients: List[int], c: int) -> float:
    """
    Computes how promising an existing route is for absorbing a client without creating a new route.
    """
    if not route_clients:
        return -1e18
    
    # Measure how well the client matches the current route
    best_pair = max(_client_compatibility(inst, c, v) for v in route_clients)

    # Apply a light penalty to routes that are already heavily loaded
    load = sum(float(inst.node_by_id[v].demand) for v in route_clients)
    load_ratio = load / max(1.0, float(inst.Q2))

    # Apply a light penalty to routes with a wide time-window span
    span = max(float(inst.tw_late(v)) for v in route_clients) - min(float(inst.tw_early(v)) for v in route_clients)
    span_pen = span / max(
        1.0,
        max(float(inst.tw_late(i)) for i in inst.client_ids)
        - min(float(inst.tw_early(i)) for i in inst.client_ids)
    )

    return best_pair - 0.35 * load_ratio - 0.10 * span_pen


def _route_global_orders(inst: Instance2E, clients: List[int]) -> List[List[int]]:
    """
    Builds several global candidate orderings for the same client set in order
    to diversify route reconstruction.
    """
    if not clients:
        return [[]]

    # Choose reference clients to build distance-based orderings
    urgent = min(clients, key=lambda x: (float(inst.tw_late(x)), float(inst.tw_early(x))))
    tw_center = sorted(clients, key=lambda x: (float(inst.tw_early(x)) + float(inst.tw_late(x))))[0]
    
    # Generate several alternative global visit orders
    orders = []
    orders.append(list(clients))
    orders.append(sorted(clients, key=lambda x: (float(inst.tw_late(x)), float(inst.tw_early(x)))))
    orders.append(sorted(clients, key=lambda x: (float(inst.tw_early(x)), float(inst.tw_late(x)))))
    orders.append(sorted(clients, key=lambda x: (_dist(inst, urgent, x), float(inst.tw_late(x)))))
    orders.append(sorted(clients, key=lambda x: (_dist(inst, tw_center, x), float(inst.tw_late(x)))))
    
    # Remove duplicate orderings while preserving their generation order
    uniq = []
    seen = set()
    for od in orders:
        tup = tuple(od)
        if tup not in seen:
            uniq.append(od)
            seen.add(tup)
    return uniq


def _candidate_orders(inst: Instance2E, route_clients: List[int], insert_client: int) -> List[List[int]]:
    """
    Generates candidate insertion orders for adding a client into an existing 
    route using local and global heuristics.
    """
    if not route_clients:
        return [[insert_client]]

    # Select anchor clients sorted by decreasing compatibility with the inserted client
    anchors = sorted(
        route_clients,
        key=lambda v: (-_client_compatibility(inst, insert_client, v), float(inst.tw_late(v)))
    )
    
    # Build candidate insertion positions around the most compatible anchors
    anchor_positions = []
    for a in anchors[: min(3, len(anchors))]:
        p = route_clients.index(a)
        anchor_positions.extend([p, p + 1])
        
    # Combine boundary positions with anchor-based positions
    all_pos = [0, len(route_clients)] + anchor_positions
    all_pos = sorted(set(p for p in all_pos if 0 <= p <= len(route_clients)))
    orders: List[List[int]] = []

    # Generate locally guided insertions at selected positions
    for pos in all_pos:
        base = list(route_clients)
        base.insert(pos, insert_client)
        orders.append(base)

    # Add global reconstructions of the entire route for diversification
    full = list(route_clients) + [insert_client]
    orders.extend(_route_global_orders(inst, full))

    # Add a targeted insertion relative to the best anchor based on time windows
    best_anchor = anchors[0]
    p = route_clients.index(best_anchor)
    if float(inst.tw_late(insert_client)) <= float(inst.tw_late(best_anchor)):
        base = list(route_clients)
        base.insert(p, insert_client)
        orders.append(base)
    else:
        base = list(route_clients)
        base.insert(p + 1, insert_client)
        orders.append(base)

    # Remove duplicate candidate orders while preserving order
    uniq = []
    seen = set()
    for od in orders:
        tup = tuple(od)
        if tup not in seen:
            uniq.append(od)
            seen.add(tup)
    return uniq


def _strict_route_cost(inst: Instance2E, nodes: List[int]) -> float:
    """Computes the total distance of a candidate route."""
    return distance_of_nodes(inst, nodes)


def _strict_route_clients(inst: Instance2E, route: Route) -> List[int]:
    """Extracts the client sequence from an existing route."""
    return _extract_clients(route, inst)


def _strict_client_criticality(inst: Instance2E, c: int) -> Tuple[float, float]:
    """Ranks clients by urgency first and demand second for strict reinsertion."""
    slack = float(inst.tw_late(c)) - float(inst.tw_early(c))
    return (slack, -float(inst.node_by_id[c].demand))


def _strict_ranked_routes_for_client(
    inst: Instance2E,
    routes_by_sat: Dict[int, List[Route]],
    c: int,
) -> List[Tuple[float, int, int, List[int], Route]]:
    """Builds and ranks feasible candidate routes that could absorb a client without exceeding capacity."""
    ranked_routes = []

    for s, routes in routes_by_sat.items():
        for ridx, route in enumerate(routes):
            cl = _strict_route_clients(inst, route)
            if not cl:
                continue

            total_load = (
                sum(float(inst.node_by_id[x].demand) for x in cl)
                + float(inst.node_by_id[c].demand)
            )
            if total_load > float(inst.Q2) + 1e-9:
                continue

            absorb = _route_absorption_score(inst, cl, c)
            ranked_routes.append((absorb, s, ridx, cl, route))

    ranked_routes.sort(key=lambda x: x[0], reverse=True)
    return ranked_routes[: min(5, len(ranked_routes))]


def _strict_best_reinsertion_for_client(
    inst: Instance2E,
    ranked_routes: List[Tuple[float, int, int, List[int], Route]],
    c: int,
) -> Optional[Tuple[Tuple[float, int, float], int, int, List[int], List[int], Dict[int, float], Dict[int, float]]]:
    """Finds the best feasible reinsertion of a client among the most promising existing routes."""
    best = None

    for _, s, ridx, cl, old_route in ranked_routes:
        old_cost = _strict_route_cost(inst, old_route.nodes)

        for order in _candidate_orders(inst, cl, c):
            mat = _materialize_route_from_clients(inst, s, order)
            if mat is None:
                continue

            nodes2, tmap, rmap, dist2 = mat
            delta = dist2 - old_cost
            route_span = max(float(inst.tw_late(x)) for x in order) - min(float(inst.tw_early(x)) for x in order)
            score = (delta, len(nodes2), route_span)

            if best is None or score < best[0]:
                best = (score, s, ridx, order, nodes2, tmap, rmap)

    return best


def _strict_apply_reinsertion_result(
    routes_by_sat: Dict[int, List[Route]],
    assignment: Dict[int, int],
    t_rel_arr: Dict[int, float],
    rel_from_sat: Dict[int, float],
    remaining: Set[int],
    s: int,
    ridx: int,
    order: List[int],
    nodes2: List[int],
    tmap: Dict[int, float],
    rmap: Dict[int, float],
) -> List[Tuple[int, int]]:
    """Applies a successful reinsertion by replacing the route and updating solution metadata."""
    old_route = routes_by_sat[s][ridx]
    routes_by_sat[s][ridx] = Route(nodes=nodes2, vehicle_id=old_route.vehicle_id)

    for cc in order:
        assignment[cc] = s
    for cc, vv in tmap.items():
        t_rel_arr[cc] = vv
    for cc, vv in rmap.items():
        rel_from_sat[cc] = vv

    newly_updated_arcs = list(zip(nodes2[:-1], nodes2[1:]))

    inserted_client = set(order) & remaining
    for cc in list(inserted_client):
        remaining.remove(cc)

    return newly_updated_arcs


def strict_reinsert_remaining(
    inst: Instance2E,
    routes_by_sat: Dict[int, List[Route]],
    assignment: Dict[int, int],
    t_rel_arr: Dict[int, float],
    rel_from_sat: Dict[int, float],
    remaining: Set[int],
    debug_msgs: List[str]
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Strictly reinserts remaining clients into existing routes only, without creating any new route."""
    updated_arcs: List[Tuple[int, int]] = []

    # Process the most critical remaining clients first
    ordered_remaining = sorted(
        list(remaining),
        key=lambda c: _strict_client_criticality(inst, c)
    )

    for c in ordered_remaining:
        ranked_routes = _strict_ranked_routes_for_client(inst, routes_by_sat, c)

        if not ranked_routes:
            debug_msgs.append(f"strict_reinsert_failed_no_capacity:{c}")
            return False, updated_arcs

        # Search for the best feasible reconstruction among the most promising routes
        best = _strict_best_reinsertion_for_client(inst, ranked_routes, c)

        if best is None:
            ok_global, arcs_global = _rebuild_all_existing_routes_no_new_vehicles(
                inst, routes_by_sat, assignment, t_rel_arr, rel_from_sat, remaining, debug_msgs
            )
            if ok_global:
                updated_arcs.extend(arcs_global)
                return True, updated_arcs

            debug_msgs.append(f"strict_reinsert_failed:{c}")
            return False, updated_arcs

        _, s, ridx, order, nodes2, tmap, rmap = best
        updated_arcs.extend(
            _strict_apply_reinsertion_result(
                routes_by_sat, assignment, t_rel_arr, rel_from_sat,
                remaining, s, ridx, order, nodes2, tmap, rmap
            )
        )

    return True, updated_arcs


def _global_rebuild_collect_inputs(
    inst: Instance2E,
    routes_by_sat: Dict[int, List[Route]],
    remaining: Set[int],
) -> Tuple[List[int], List[int], int]:
    """Collects the available vehicle slots and the full set of clients that must be redistributed."""
    vehicle_ids: List[int] = []
    all_clients: List[int] = []
    k_slots = 0

    for _, routes in routes_by_sat.items():
        for route in routes:
            vehicle_ids.append(route.vehicle_id)
            all_clients.extend(_extract_clients(route, inst))
            k_slots += 1

    all_clients.extend(list(remaining))
    all_clients = sorted(set(all_clients))
    return vehicle_ids, all_clients, k_slots


def _global_rebuild_client_priority(inst: Instance2E, c: int) -> Tuple[float, float, float]:
    """Ranks clients for the global rebuild by urgency, demand, and proximity to the best satellite."""
    best_sat = min(_dist(inst, s, c) for s in inst.satellite_ids)
    slack = float(inst.tw_late(c)) - float(inst.tw_early(c))
    return (slack, -float(inst.node_by_id[c].demand), -best_sat)


def _global_rebuild_materialize_any_sat(
    inst: Instance2E,
    order: List[int],
    cache: Dict[Tuple[int, ...], Optional[Tuple[int, List[int], Dict[int, float], Dict[int, float], float]]],
) -> Optional[Tuple[int, List[int], Dict[int, float], Dict[int, float], float]]:
    """
    Materializes a route on the best satellite for a client order, using 
    exact resolution on small subsets and caching results.
    """
    key = tuple(order)
    if key in cache:
        return cache[key]

    if len(order) <= 10:
        exact = _exact_route_any_sat_for_subset(inst, order)
        if exact is not None:
            cache[key] = exact
            return exact

    best = None
    for s in sorted(inst.satellite_ids):
        mat = _materialize_route_from_clients(inst, s, order)
        if mat is None:
            continue

        nodes2, tmap, rmap, dist2 = mat
        score = (dist2, len(nodes2), s)
        if best is None or score < best[0]:
            best = (score, s, nodes2, tmap, rmap, dist2)

    cache[key] = None if best is None else (best[1], best[2], best[3], best[4], best[5])
    return cache[key]


def _global_rebuild_slot_score(
    inst: Instance2E,
    slot_orders: List[List[int]],
    k: int,
    c: int,
) -> Tuple[float, float, int]:
    """Scores how promising a slot is for receiving a given client during the global rebuild."""
    existing = slot_orders[k]
    if not existing:
        best_sat = min(_dist(inst, s, c) for s in inst.satellite_ids)
        return (0.0, best_sat, 0)

    compat = max(_client_compatibility(inst, c, v) for v in existing)
    return (-compat, min(_dist(inst, s, c) for s in inst.satellite_ids), len(existing))


def _global_rebuild_encode_state(idx: int, slot_orders: List[List[int]]) -> Tuple[int, Tuple[Tuple[int, ...], ...]]:
    """Encodes the current DFS state for memoization and duplicate pruning."""
    return (idx, tuple(tuple(order) for order in slot_orders))


def _global_rebuild_apply_solution(
    inst: Instance2E,
    routes_by_sat: Dict[int, List[Route]],
    assignment: Dict[int, int],
    t_rel_arr: Dict[int, float],
    rel_from_sat: Dict[int, float],
    remaining: Set[int],
    best_solution: List[List[int]],
    vehicle_ids: List[int],
    cache: Dict[Tuple[int, ...], Optional[Tuple[int, List[int], Dict[int, float], Dict[int, float], float]]],
    debug_msgs: List[str],
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Applies the rebuilt solution to the current structures and updates assignments, timing maps, and route arcs."""
    updated_arcs: List[Tuple[int, int]] = []
    assignment.clear()
    t_rel_arr.clear()
    rel_from_sat.clear()
    rebuilt: Dict[int, List[Route]] = {s: [] for s in inst.satellite_ids}

    for order, vehicle_id in zip(best_solution, vehicle_ids):
        mat = _global_rebuild_materialize_any_sat(inst, order, cache)
        if mat is None:
            debug_msgs.append('global_rebuild_materialization_failed')
            return False, []

        sat, nodes2, tmap, rmap, _ = mat
        if len(nodes2) > 2:
            rebuilt[sat].append(Route(nodes=nodes2, vehicle_id=vehicle_id))

        updated_arcs.extend(list(zip(nodes2[:-1], nodes2[1:])))
        for cc in order:
            assignment[cc] = sat
        t_rel_arr.update(tmap)
        rel_from_sat.update(rmap)

    routes_by_sat.clear()
    for s in inst.satellite_ids:
        routes_by_sat[s] = rebuilt.get(s, [])

    remaining.clear()
    debug_msgs.append('global_rebuild_success')
    return True, updated_arcs


def _global_rebuild_try_assign_client(
    inst: Instance2E,
    clients: List[int],
    idx: int,
    k_slots: int,
    slot_orders: List[List[int]],
    slot_loads: List[float],
    cache: Dict[Tuple[int, ...], Optional[Tuple[int, List[int], Dict[int, float], Dict[int, float], float]]],
) -> List[Tuple[int, List[int], float]]:
    """Builds the most promising slot-and-order candidates for assigning the current client."""
    c = clients[idx]
    demand_c = float(inst.node_by_id[c].demand)
    candidates: List[Tuple[int, List[int], float]] = []

    slots = sorted(
        range(k_slots),
        key=lambda k: _global_rebuild_slot_score(inst, slot_orders, k, c)
    )

    for k in slots:
        if slot_loads[k] + demand_c > float(inst.Q2) + 1e-9:
            continue

        trials = []
        for order in _candidate_orders(inst, slot_orders[k], c):
            mat = _global_rebuild_materialize_any_sat(inst, order, cache)
            if mat is None:
                continue

            sat, nodes2, _, _, dist2 = mat
            trials.append((dist2, len(nodes2), sat, order))

        trials.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, _, _, order in trials[:4]:
            candidates.append((k, order, demand_c))

    return candidates


def _global_rebuild_dfs(
    inst: Instance2E,
    clients: List[int],
    idx: int,
    k_slots: int,
    slot_orders: List[List[int]],
    slot_loads: List[float],
    cache: Dict[Tuple[int, ...], Optional[Tuple[int, List[int], Dict[int, float], Dict[int, float], float]]],
    seen_states: Set[Tuple[int, Tuple[Tuple[int, ...], ...]]],
) -> Optional[List[List[int]]]:
    """Searches for a feasible redistribution of all clients over the existing route slots."""
    if idx >= len(clients):
        return [list(order) for order in slot_orders]

    state = _global_rebuild_encode_state(idx, slot_orders)
    if state in seen_states:
        return None
    seen_states.add(state)

    # Try the most promising slot-and-order combinations for the current client
    candidates = _global_rebuild_try_assign_client(
        inst, clients, idx, k_slots, slot_orders, slot_loads, cache
    )

    for k, order, demand_c in candidates:
        prev_order = slot_orders[k]
        prev_load = slot_loads[k]

        slot_orders[k] = list(order)
        slot_loads[k] = prev_load + demand_c

        solution = _global_rebuild_dfs(
            inst, clients, idx + 1, k_slots,
            slot_orders, slot_loads, cache, seen_states
        )
        if solution is not None:
            return solution

        slot_orders[k] = prev_order
        slot_loads[k] = prev_load

    return None


def _rebuild_all_existing_routes_no_new_vehicles(
    inst: Instance2E,
    routes_by_sat: Dict[int, List[Route]],
    assignment: Dict[int, int],
    t_rel_arr: Dict[int, float],
    rel_from_sat: Dict[int, float],
    remaining: Set[int],
    debug_msgs: List[str],
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Globally rebuilds all existing EV routes by redistributing served and remaining clients over the same number of vehicles without creating new ones."""
    # Collect the available vehicle slots and the complete set of clients to reassign
    vehicle_ids, all_clients, k_slots = _global_rebuild_collect_inputs(inst, routes_by_sat, remaining)

    if k_slots <= 0:
        debug_msgs.append('global_rebuild_no_route_slot')
        return False, []

    total_demand = sum(float(inst.node_by_id[c].demand) for c in all_clients)
    if total_demand > k_slots * float(inst.Q2) + 1e-9:
        debug_msgs.append('global_rebuild_total_capacity_impossible')
        return False, []

    # Prepare the client ordering and the search state for the global redistribution
    clients = sorted(all_clients, key=lambda c: _global_rebuild_client_priority(inst, c))
    slot_orders: List[List[int]] = [[] for _ in range(k_slots)]
    slot_loads: List[float] = [0.0 for _ in range(k_slots)]
    cache: Dict[Tuple[int, ...], Optional[Tuple[int, List[int], Dict[int, float], Dict[int, float], float]]] = {}
    seen_states: Set[Tuple[int, Tuple[Tuple[int, ...], ...]]] = set()

    best_solution = _global_rebuild_dfs(
        inst, clients, 0, k_slots,
        slot_orders, slot_loads, cache, seen_states
    )

    if best_solution is None:
        debug_msgs.append(f'global_rebuild_failed_for:{sorted(remaining)}')
        return False, []

    # Apply the rebuilt assignment and materialized routes back to the current solution
    return _global_rebuild_apply_solution(
        inst,
        routes_by_sat,
        assignment,
        t_rel_arr,
        rel_from_sat,
        remaining,
        best_solution,
        vehicle_ids,
        cache,
        debug_msgs,
    )


def seed_pheromone_from_solution(ph: Pheromones, sol: Solution2E, params: ACSParams) -> List[Tuple[int, int, float]]:
    """Inject the Clarke-and-Wright seed solution directly into the pheromone matrix."""
    boosted: List[Tuple[int, int, float]] = []
    base_total = float(sol.meta.get("seed_distance", sol.meta.get("dist_lvl2", 1.0)) or 1.0)
    delta = max(1.0, float(params.seed_pheromone_boost)) / max(1.0, base_total)

    used_satellites = sorted(s for s, routes in sol.routes_lvl2.items() if routes)
    for sat in used_satellites:
        ph.global_reinforce(START_NODE_ID, sat, delta)
        boosted.append((START_NODE_ID, sat, delta))

    for sat, routes in sol.routes_lvl2.items():
        for route in routes:
            for pos, (i, j) in enumerate(route.arcs(), start=1):
                arc_delta = delta * (1.0 + 1.0 / float(pos))
                ph.global_reinforce(i, j, arc_delta)
                boosted.append((i, j, arc_delta))
    for c, s in sol.assignment.items():
        ph.global_reinforce_assign(int(c), int(s), delta * 1.5)

    return boosted


def _seed_log(debug_msgs: List[str], msg: str) -> None:
    """Prints and stores a seed-construction debug message."""
    line = f"[SEED] {msg}"
    print(line)
    debug_msgs.append(msg)


def _seed_clients_sorted(inst: Instance2E) -> List[int]:
    """Returns clients sorted by urgency and tie-broken by earliest time window and id."""
    return sorted(
        inst.client_ids,
        key=lambda c: (float(inst.tw_late(c)), float(inst.tw_early(c)), c)
    )


def _seed_get_order_mat(
    inst: Instance2E,
    sat: int,
    order: List[int],
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]:
    """Returns the cached materialization of a client order for a satellite, computing it if needed."""
    key = (sat, tuple(order))
    if key not in order_cache:
        order_cache[key] = _materialize_route_from_clients(inst, sat, list(order))
    return order_cache[key]


def _seed_eval_merge(
    inst: Instance2E,
    sat: int,
    left: List[int],
    right: List[int],
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]:
    """Evaluates the best feasible merge between two route client sequences for a given satellite."""
    best = None
    seen = set()

    candidates = [
        list(left) + list(right),
        list(left) + list(reversed(right)),
        list(reversed(left)) + list(right),
        list(reversed(left)) + list(reversed(right)),
    ]

    for order in candidates:
        tup = tuple(order)
        if tup in seen:
            continue
        seen.add(tup)

        mat = _seed_get_order_mat(inst, sat, order, order_cache)
        if mat is None:
            continue

        nodes2, tmap, rmap, dist2 = mat
        score = (dist2, len(nodes2), tup)
        if best is None or score < best[0]:
            best = (score, list(order), tmap, rmap, dist2)

    return None if best is None else (best[1], best[2], best[3], best[4])


def _seed_assign_singletons(
    inst: Instance2E,
    debug_msgs: List[str],
) -> Tuple[
    Dict[Tuple[int, int], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
    Dict[int, int],
    Dict[int, List[int]],
    List[int],
]:
    """Assigns each client to the best feasible singleton satellite route."""
    singleton_cache: Dict[Tuple[int, int], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]] = {}
    assignment: Dict[int, int] = {}
    clients_by_sat: Dict[int, List[int]] = {s: [] for s in inst.satellite_ids}
    unassigned: List[int] = []

    # Try all satellites for each client and keep the best singleton route
    for c in _seed_clients_sorted(inst):
        best_sat = None
        best_score = None

        for s in sorted(inst.satellite_ids):
            key = (s, c)
            if key not in singleton_cache:
                singleton_cache[key] = _materialize_route_from_clients(inst, s, [c])

            mat = singleton_cache[key]
            if mat is None:
                continue

            nodes2, _, _, dist2 = mat
            score = (dist2, len(nodes2), s)
            if best_score is None or score < best_score:
                best_score = score
                best_sat = s

        if best_sat is None:
            unassigned.append(c)
            _seed_log(debug_msgs, f"singleton_fail client={c}")
            continue

        assignment[c] = int(best_sat)
        clients_by_sat[int(best_sat)].append(c)

    return singleton_cache, assignment, clients_by_sat, unassigned


def _seed_build_initial_routes_for_sat(
    inst: Instance2E,
    sat: int,
    sat_clients: List[int],
    singleton_cache: Dict[Tuple[int, int], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Tuple[Dict[int, Dict[str, object]], Dict[int, int], int]:
    """Builds singleton route records for one satellite before Clarke-Wright merges."""
    routes: Dict[int, Dict[str, object]] = {}
    route_of_client: Dict[int, int] = {}
    next_rid = 0

    for c in sat_clients:
        mat = singleton_cache.get((sat, c))
        if mat is None:
            continue

        _, _, _, dist2 = mat
        routes[next_rid] = {
            "rid": next_rid,
            "clients": [c],
            "load": float(inst.node_by_id[c].demand),
            "dist": float(dist2),
        }
        route_of_client[c] = next_rid
        next_rid += 1

    return routes, route_of_client, next_rid


def _seed_build_savings_list(inst: Instance2E, sat: int, sat_clients: List[int]) -> List[Tuple[float, int, int]]:
    """Builds and sorts the Clarke-Wright savings list for one satellite."""
    savings: List[Tuple[float, int, int]] = []

    for i in range(len(sat_clients)):
        a = sat_clients[i]
        for j in range(i + 1, len(sat_clients)):
            b = sat_clients[j]
            sav = _dist(inst, sat, a) + _dist(inst, sat, b) - _dist(inst, a, b)
            tw_bonus = 0.05 / (1.0 + abs(float(inst.tw_late(a)) - float(inst.tw_late(b))))
            savings.append((sav + tw_bonus, a, b))

    savings.sort(reverse=True)
    return savings


def _seed_try_merge_pair(
    inst: Instance2E,
    sat: int,
    a: int,
    b: int,
    routes: Dict[int, Dict[str, object]],
    route_of_client: Dict[int, int],
    next_rid: int,
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Tuple[bool, int]:
    """Attempts to merge two current routes through an endpoint-based Clarke-Wright move."""
    ra_id = route_of_client.get(a)
    rb_id = route_of_client.get(b)

    if ra_id is None or rb_id is None or ra_id == rb_id:
        return False, next_rid
    if ra_id not in routes or rb_id not in routes:
        return False, next_rid

    ra = routes[ra_id]
    rb = routes[rb_id]
    cli_a = list(ra["clients"])
    cli_b = list(rb["clients"])

    if (a not in (cli_a[0], cli_a[-1])) or (b not in (cli_b[0], cli_b[-1])):
        return False, next_rid

    load_new = float(ra["load"]) + float(rb["load"])
    if load_new > float(inst.Q2) + 1e-9:
        return False, next_rid

    merged = _seed_eval_merge(inst, sat, cli_a, cli_b, order_cache)
    if merged is None:
        return False, next_rid

    order, _, _, dist_merged = merged
    if float(ra["dist"]) + float(rb["dist"]) < dist_merged - 1e-9:
        return False, next_rid

    # Replace the two old routes by the merged one
    new_id = next_rid
    routes[new_id] = {
        "rid": new_id,
        "clients": list(order),
        "load": load_new,
        "dist": float(dist_merged),
    }
    for c in order:
        route_of_client[c] = new_id

    del routes[ra_id]
    del routes[rb_id]
    return True, next_rid + 1


def _seed_merge_satellite_routes(
    inst: Instance2E,
    sat: int,
    sat_clients: List[int],
    singleton_cache: Dict[Tuple[int, int], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Tuple[List[Dict[str, object]], int, int]:
    """
    Runs a single Clarke-Wright pass on one satellite and returns the remaining
    route records.
    """
    routes, route_of_client, next_rid = _seed_build_initial_routes_for_sat(
        inst, sat, sat_clients, singleton_cache
    )
    savings = _seed_build_savings_list(inst, sat, sat_clients)

    merge_checks = 0
    merge_success = 0

    # Process savings in descending order and keep only feasible beneficial merges
    for _, a, b in savings:
        merge_checks += 1
        ok, next_rid = _seed_try_merge_pair(
            inst, sat, a, b, routes, route_of_client, next_rid, order_cache
        )
        if ok:
            merge_success += 1

    route_records = [
        {"sat": sat, "clients": list(rec["clients"]), "dist": float(rec["dist"])}
        for rec in routes.values()
    ]
    return route_records, merge_checks, merge_success


def _seed_build_route_records(
    inst: Instance2E,
    clients_by_sat: Dict[int, List[int]],
    singleton_cache: Dict[Tuple[int, int], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Tuple[List[Dict[str, object]], int, int]:
    """Builds all second-level seed route records satellite by satellite."""
    route_records: List[Dict[str, object]] = []
    merge_checks = 0
    merge_success = 0

    for sat in sorted(inst.satellite_ids):
        sat_clients = sorted(
            clients_by_sat.get(sat, []),
            key=lambda c: (float(inst.tw_late(c)), float(inst.tw_early(c)), c),
        )
        if not sat_clients:
            continue

        sat_records, sat_checks, sat_success = _seed_merge_satellite_routes(
            inst, sat, sat_clients, singleton_cache, order_cache
        )
        route_records.extend(sat_records)
        merge_checks += sat_checks
        merge_success += sat_success

    return route_records, merge_checks, merge_success


def _seed_materialize_route_records(
    inst: Instance2E,
    route_records: List[Dict[str, object]],
    gen: VehicleIdGenerator,
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Tuple[Dict[int, List[Route]], Dict[int, float], Dict[int, float], Dict[int, int], Set[int]]:
    """Materializes route records into actual routes and updates timing and assignment structures."""
    routes_by_sat: Dict[int, List[Route]] = {s: [] for s in inst.satellite_ids}
    t_rel_arr: Dict[int, float] = {}
    rel_from_sat: Dict[int, float] = {}
    assignment_final: Dict[int, int] = {}

    # Turn each client order into a concrete route, using exact fallback on small subsets when needed
    for rr in route_records:
        sat = int(rr["sat"])
        order = list(rr["clients"])

        mat = _seed_get_order_mat(inst, sat, order, order_cache)
        if mat is None and len(order) <= 10:
            mat = _exact_route_for_subset(inst, sat, order)
        if mat is None:
            continue

        nodes2, tmap, rmap, _ = mat
        routes_by_sat[sat].append(Route(nodes=nodes2, vehicle_id=gen.new_ev_id()))

        for c in order:
            assignment_final[c] = sat
        t_rel_arr.update(tmap)
        rel_from_sat.update(rmap)

    remaining = set(inst.client_ids) - set(assignment_final.keys())
    return routes_by_sat, t_rel_arr, rel_from_sat, assignment_final, remaining


def _seed_find_best_budget_merge(
    inst: Instance2E,
    route_records: List[Dict[str, object]],
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Optional[Tuple[Tuple[float, int, int, int, int], int, int, int, List[int], float]]:
    """Finds the best route-pair merge when reducing the number of routes to satisfy the EV budget."""
    best = None

    for i in range(len(route_records)):
        ri = route_records[i]
        cli_i = list(ri["clients"])
        load_i = sum(float(inst.node_by_id[c].demand) for c in cli_i)

        for j in range(i + 1, len(route_records)):
            rj = route_records[j]
            cli_j = list(rj["clients"])
            load_j = sum(float(inst.node_by_id[c].demand) for c in cli_j)

            if load_i + load_j > float(inst.Q2) + 1e-9:
                continue

            candidate_sats = sorted({int(ri["sat"]), int(rj["sat"])} | set(inst.satellite_ids))
            for sat in candidate_sats:
                merged = _seed_eval_merge(inst, sat, cli_i, cli_j, order_cache)
                if merged is None:
                    continue

                order, _, _, md = merged
                delta = float(md) - float(ri["dist"]) - float(rj["dist"])
                score = (delta, len(order), sat, i, j)

                if best is None or score < best[0]:
                    best = (score, i, j, sat, order, md)

    return best


def _seed_try_budget_relocation(
    inst: Instance2E,
    route_records: List[Dict[str, object]],
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> Optional[List[Dict[str, object]]]:
    """Tries to remove one donor route by reinserting its clients into the other existing routes."""
    order_idx = sorted(
        range(len(route_records)),
        key=lambda k: (len(route_records[k]["clients"]), float(route_records[k]["dist"]))
    )

    # Try the smallest and cheapest donor routes first
    for donor_idx in order_idx:
        donor = route_records[donor_idx]
        donor_clients = list(donor["clients"])
        updated_routes = [dict(r) for idx_r, r in enumerate(route_records) if idx_r != donor_idx]
        ok_reloc = True

        for c in donor_clients:
            best_ins = None
            dem_c = float(inst.node_by_id[c].demand)

            for tgt_idx, tgt in enumerate(updated_routes):
                cli_t = list(tgt["clients"])
                load_t = sum(float(inst.node_by_id[x].demand) for x in cli_t)
                if load_t + dem_c > float(inst.Q2) + 1e-9:
                    continue

                sat_t = int(tgt["sat"])
                for pos in range(len(cli_t) + 1):
                    order = cli_t[:pos] + [c] + cli_t[pos:]
                    mat = _seed_get_order_mat(inst, sat_t, order, order_cache)
                    if mat is None:
                        continue

                    _, _, _, md = mat
                    delta = float(md) - float(tgt["dist"])
                    score = (delta, len(order), sat_t, tgt_idx, pos)

                    if best_ins is None or score < best_ins[0]:
                        best_ins = (score, tgt_idx, order, md)

            if best_ins is None:
                ok_reloc = False
                break

            _, tgt_idx, order_best, md_best = best_ins
            updated_routes[tgt_idx] = {
                "sat": int(updated_routes[tgt_idx]["sat"]),
                "clients": list(order_best),
                "dist": float(md_best),
            }

        if ok_reloc:
            return updated_routes

    return None


def _seed_repair_route_budget(
    inst: Instance2E,
    route_records: List[Dict[str, object]],
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]],
) -> List[Dict[str, object]]:
    """Repairs the seed solution so that the number of routes does not exceed the EV fleet size."""
    route_records = [dict(r) for r in route_records]
    route_count = len(route_records)

    while route_count > int(inst.nv2):
        best = _seed_find_best_budget_merge(inst, route_records, order_cache)

        # Prefer route merges; if none is feasible, try relocating one full donor route
        if best is None:
            relocated = _seed_try_budget_relocation(inst, route_records, order_cache)
            if relocated is None:
                break
            route_records = relocated
            route_count = len(route_records)
            continue

        _, i, j, sat, order, md = best
        if i > j:
            i, j = j, i

        route_records.pop(j)
        route_records.pop(i)
        route_records.append({"sat": int(sat), "clients": list(order), "dist": float(md)})
        route_count = len(route_records)

    return route_records


def _seed_build_fallback_routes(
    inst: Instance2E,
    gen: VehicleIdGenerator,
) -> Dict[int, List[Route]]:
    """Builds an empty fallback route skeleton with exactly nv2 route slots distributed over satellites."""
    routes_by_sat: Dict[int, List[Route]] = {s: [] for s in inst.satellite_ids}
    sats = sorted(inst.satellite_ids)

    for k in range(int(inst.nv2)):
        s = sats[k % len(sats)]
        routes_by_sat[s].append(Route(nodes=[s, s], vehicle_id=gen.new_ev_id()))

    return routes_by_sat


def _seed_global_fallback_rebuild(
    inst: Instance2E,
    gen: VehicleIdGenerator,
    debug_msgs: List[str],
) -> Tuple[Dict[int, List[Route]], Dict[int, int], Dict[int, float], Dict[int, float], Set[int]]:
    """Rebuilds all clients globally over a fixed number of empty EV route slots."""
    routes_by_sat = _seed_build_fallback_routes(inst, gen)
    assignment_final: Dict[int, int] = {}
    t_rel_arr: Dict[int, float] = {}
    rel_from_sat: Dict[int, float] = {}
    remaining = set(sorted(inst.client_ids))

    ok, _ = _rebuild_all_existing_routes_no_new_vehicles(
        inst,
        routes_by_sat,
        assignment_final,
        t_rel_arr,
        rel_from_sat,
        remaining,
        debug_msgs,
    )

    if not ok:
        _seed_log(debug_msgs, "seed_global_rebuild_fail")
    else:
        _seed_log(debug_msgs, "seed_global_rebuild_success")

    return routes_by_sat, assignment_final, t_rel_arr, rel_from_sat, remaining


def _seed_latest_sat_arrival(
    inst: Instance2E,
    assignment_final: Dict[int, int],
    rel_from_sat: Dict[int, float],
) -> Dict[int, float]:
    """Computes the latest feasible satellite arrival time induced by assigned client schedules."""
    latest_sat_arrival: Dict[int, float] = {}

    for s in inst.satellite_ids:
        vals = [
            float(inst.tw_late(c)) - rel_from_sat[c]
            for c, ss in assignment_final.items()
            if ss == s and c in rel_from_sat
        ]
        latest_sat_arrival[s] = min(vals) if vals else float("inf")

    return latest_sat_arrival


def construct_seed_solution_lvl2(inst: Instance2E, params: ACSParams, gen: VehicleIdGenerator) -> Solution2E:
    """Builds a fast second-level seed solution using singleton assignment, one-pass Clarke-Wright merges, and fallback rebuilding if needed."""
    import time

    t_seed0 = time.perf_counter()
    debug_msgs: List[str] = []

    # Build the initial singleton assignment of clients to feasible satellites
    singleton_cache, _, clients_by_sat, unassigned = _seed_assign_singletons(inst, debug_msgs)

    # Build route records with a fast Clarke-Wright procedure on each satellite
    order_cache: Dict[Tuple[int, Tuple[int, ...]], Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]] = {}
    route_records, merge_checks, merge_success = _seed_build_route_records(
        inst, clients_by_sat, singleton_cache, order_cache
    )

    # Materialize the merged route records into actual second-level routes
    routes_by_sat, t_rel_arr, rel_from_sat, assignment_final, remaining = _seed_materialize_route_records(
        inst, route_records, gen, order_cache
    )
    route_count = sum(len(v) for v in routes_by_sat.values())

    # Repair the solution if the route budget is exceeded, then rematerialize
    if route_count > int(inst.nv2):
        route_records = _seed_repair_route_budget(inst, route_records, order_cache)
        route_count = len(route_records)

        if route_count <= int(inst.nv2):
            routes_by_sat, t_rel_arr, rel_from_sat, assignment_final, remaining = _seed_materialize_route_records(
                inst, route_records, gen, order_cache
            )
            route_count = sum(len(v) for v in routes_by_sat.values())

    # If some clients remain unserved or the route budget is still violated, use a full global fallback rebuild
    if remaining or route_count > int(inst.nv2):
        routes_by_sat, assignment_final, t_rel_arr, rel_from_sat, remaining = _seed_global_fallback_rebuild(
            inst, gen, debug_msgs
        )

    latest_sat_arrival = _seed_latest_sat_arrival(inst, assignment_final, rel_from_sat)
    success = (
        set(assignment_final.keys()) == set(inst.client_ids)
        and sum(len(v) for v in routes_by_sat.values()) <= int(inst.nv2)
    )
    seed_cpu = time.perf_counter() - t_seed0

    return Solution2E(
        assignment=dict(assignment_final),
        routes_lvl2={s: rs for s, rs in routes_by_sat.items() if rs},
        routes_lvl1=[],
        meta={
            "seed_solution": True,
            "seed_method": "fast_modified_clarke_wright_lvl2",
            "seed_stats": {
                "cpu_seed_total": float(seed_cpu),
                "assigned_clients": int(len(assignment_final)),
                "unassigned_after_assignment": int(len(unassigned)),
                "merge_checks": int(merge_checks),
                "merge_success": int(merge_success),
                "routes_seed_lvl2": int(sum(len(v) for v in routes_by_sat.values())),
                "nv2_limit": int(inst.nv2),
            },
            "t_rel_arr": {int(k): float(v) for k, v in t_rel_arr.items()},
            "rel_from_sat": {int(k): float(v) for k, v in rel_from_sat.items()},
            "latest_sat_arrival": {int(k): float(v) for k, v in latest_sat_arrival.items()},
            "unserved_clients": sorted(list(set(inst.client_ids) - set(assignment_final.keys()))),
            "repair_penalty_lvl2": 0.0 if success else float(params.penalty_repair_fail),
            "debug_msgs": debug_msgs,
            "infeasible": not success,
            "ant_accept": bool(success),
            "ant_attempt": 0,
        },
    )

def _build_one_attempt(
    inst: Instance2E,
    ph: Pheromones,
    hm: HeuristicMemory,
    params: ACSParams,
    rng: random.Random,
    gen: VehicleIdGenerator
) -> Tuple[Dict[int, List[Route]], Dict[int, int], Dict[int, float], Dict[int, float], Set[int], List[str], List[Tuple[int, int]]]:
    """
    Performs one full hierarchical construction attempt by first assigning 
    clients to satellites and then building feasible routes.
    """
    
    # Phase 1: construct a client-to-satellite assignment using one ant
    assignment_phase, sat_clients, sat_loads, debug_assign = _construct_assignment_one_ant(
        inst, ph, params, rng
    )

    # Phase 2: build routes for each satellite based on the assignment
    routes_by_sat, assignment_final, t_rel_arr, rel_from_sat, unserved, debug_route, used_arcs = _build_routes_for_assignment(
        inst, ph, hm, params, rng, gen, sat_clients
    )

    # Combine debug messages from both phases and add a global summary
    debug_msgs = list(debug_assign) + list(debug_route)
    debug_msgs.append(
        f"hierarchical_two_phase assignment_clients={len(assignment_phase)} "
        f"routed_clients={len(assignment_final)} "
        f"total_routes={sum(len(v) for v in routes_by_sat.values())}"
    )

    return routes_by_sat, assignment_final, t_rel_arr, rel_from_sat, unserved, debug_msgs, used_arcs