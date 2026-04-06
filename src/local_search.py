from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import random
import time

from .instance import Instance2E
from .solution import Route, Solution2E
from .aco import _materialize_route_from_clients, _exact_route_for_subset, _dist, _sat_start, distance_of_nodes
from .lvl1_cw import simulate_truck_arrivals
from .sync import _rebuild_satellite_routes_absolute_global, repair_and_replay_lvl2_absolute, _refresh_solution_consistency

EPS = 1e-9


def _time_remaining(deadline: Optional[float]) -> float:
    """Returns the remaining time until a deadline, or infinity when no deadline is provided."""
    if deadline is None:
        return float("inf")
    return max(0.0, float(deadline) - time.perf_counter())


def _time_exceeded(deadline: Optional[float]) -> bool:
    """Checks whether the current time has reached or passed the given deadline.""" 
    return deadline is not None and time.perf_counter() >= float(deadline)



@dataclass
class LSResult:
    """Stores the outcome of a local-search phase, including the resulting solution and search statistics."""
    solution: Solution2E
    improved: bool
    moves: int
    best_dist2: float
    debug: List[str]


@dataclass
class RoutePoolEntry:
    """Represents one route candidate stored in the route pool for recombination or reuse."""
    sat: int
    clients: Tuple[int, ...]
    nodes: List[int]
    cost: float

    @property
    def client_set(self) -> frozenset[int]:
        """Returns the client subset of the route as an immutable set for fast comparisons."""
        return frozenset(int(c) for c in self.clients)


def _latest_sat_arrival(inst: Instance2E, assignment: Dict[int, int], rel_from_sat: Dict[int, float]) -> Dict[int, float]:
    """Computes, for each satellite, the latest relative arrival value induced by its assigned clients."""
    latest: Dict[int, float] = {}
    for c, s in assignment.items():
        latest[s] = max(latest.get(s, 0.0), float(rel_from_sat.get(c, 0.0)))
    return latest


def _clone_client_routes(sol: Solution2E) -> Dict[int, List[List[int]]]:
    """Extracts and clones the client-only orders from the second-level routes of a solution."""
    out: Dict[int, List[List[int]]] = {}
    client_set = set(sol.assignment.keys())
    
    # Keep only customer nodes inside each second-level route
    for sat, routes in sol.routes_lvl2.items():
        out[int(sat)] = []
        for r in routes:
            out[int(sat)].append([n for n in r.nodes[1:-1] if n in client_set])
    return out


def _clone_orders(orders_by_sat: Dict[int, List[List[int]]]) -> Dict[int, List[List[int]]]:
    """Creates a deep copy of a satellite-to-route-orders dictionary."""
    return {int(s): [list(rt) for rt in rts] for s, rts in orders_by_sat.items()}


def _route_key(sat: int, order: List[int], use_exact: bool) -> Tuple[int, Tuple[int, ...], bool]:
    """Builds a canonical cache key for a route order on a given satellite."""
    return int(sat), tuple(int(x) for x in order), bool(use_exact)


def _simulate_explicit_route(inst: Instance2E, sat: int, nodes: List[int]) -> Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]:
    """Simulates an explicit second-level route and returns its timing data and distance if it is feasible."""
    if not nodes or nodes[0] != sat or nodes[-1] != sat:
        return None
    battery = float(inst.BCe)
    load = 0.0
    t = _sat_start(inst, sat)
    t_rel_arr: Dict[int, float] = {}
    rel_from_sat: Dict[int, float] = {}
    cur = sat
    start_shift = _sat_start(inst, sat)
    
    # Traverse the route node by node while checking battery, time windows, and capacity
    for nxt in nodes[1:]:
        battery -= float(inst.h) * _dist(inst, cur, nxt)
        if battery < -1e-9:
            return None
        t += _dist(inst, cur, nxt)
        if nxt in inst.station_ids:
            t += float(inst.ge) * max(0.0, float(inst.BCe) - battery)
            battery = float(inst.BCe)
        elif nxt in inst.client_ids:
            start = max(t, float(inst.tw_early(nxt)))
            if start > float(inst.tw_late(nxt)) + 1e-9:
                return None
            t_rel_arr[nxt] = float(start)
            rel_from_sat[nxt] = float(start - start_shift)
            t = start + float(inst.service_time(nxt))
            load += float(inst.node_by_id[nxt].demand)
            if load > float(inst.Q2) + 1e-9:
                return None
        elif nxt != sat:
            return None
        cur = nxt
    return list(nodes), t_rel_arr, rel_from_sat, distance_of_nodes(inst, list(nodes))


def _recharge_cleanup(inst: Instance2E, sat: int, mat: Tuple[List[int], Dict[int, float], Dict[int, float], float]) -> Tuple[List[int], Dict[int, float], Dict[int, float], float]:
    """Improves a materialized route by removing useless charging stations or replacing them with better ones when possible."""
    nodes = list(mat[0])
    improved = True
    
    # Repeatedly simplify the route while a better feasible variant can still be found
    while improved:
        improved = False
        # First try to remove charging stations that are no longer necessary
        for idx in range(1, len(nodes) - 1):
            if nodes[idx] not in inst.station_ids:
                continue
            cand_nodes = nodes[:idx] + nodes[idx + 1 :]
            sim = _simulate_explicit_route(inst, sat, cand_nodes)
            if sim is not None and sim[3] + EPS < distance_of_nodes(inst, nodes):
                nodes = cand_nodes
                improved = True
                break
        if improved:
            continue
        
        # If no station can be removed, try replacing one by a better nearby station
        for idx in range(1, len(nodes) - 1):
            old = nodes[idx]
            if old not in inst.station_ids:
                continue
            prev_n = nodes[idx - 1]
            next_n = nodes[idx + 1]
            candidates = sorted(
                [r for r in inst.station_ids if r != old],
                key=lambda r: _dist(inst, prev_n, r) + _dist(inst, r, next_n),
            )[:5]
            base_dist = distance_of_nodes(inst, nodes)
            for new_st in candidates:
                cand_nodes = list(nodes)
                cand_nodes[idx] = new_st
                sim = _simulate_explicit_route(inst, sat, cand_nodes)
                if sim is not None and sim[3] + EPS < base_dist:
                    nodes = cand_nodes
                    improved = True
                    break
            if improved:
                break
    sim = _simulate_explicit_route(inst, sat, nodes)
    return sim if sim is not None else mat


def _route_cost(
    inst: Instance2E,
    sat: int,
    order: List[int],
    exact_threshold: int,
    *,
    cache: Optional[Dict[Tuple[int, Tuple[int, ...], bool], Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]]] = None,
    use_exact: bool = True,
) -> Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]:
    """
    Computes the cost of a client order on a satellite and returns its materialized
    route, using exact resolution when appropriate.
    """
    key = _route_key(sat, order, use_exact)
    if cache is not None and key in cache:
        return cache[key]
    
    # Try the exact route construction first on small enough routes
    res: Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]
    if use_exact and len(order) <= exact_threshold:
        mat = _exact_route_for_subset(inst, sat, list(order))
        if mat is not None:
            mat = _recharge_cleanup(inst, sat, mat)
            res = (float(mat[3]), mat)
            if cache is not None:
                cache[key] = res
            return res
        
    # Fall back to the constructive materialization if exact construction is not used or fails
    mat = _materialize_route_from_clients(inst, sat, list(order))
    if mat is None:
        res = (float("inf"), None)
    else:
        mat = _recharge_cleanup(inst, sat, mat)
        res = (float(mat[3]), mat)
    if cache is not None:
        cache[key] = res
    return res


def _route_seq_cost(inst: Instance2E, sat: int, order: List[int]) -> float:
    """Computes the simple depot-satellite route distance induced by a client visit order."""
    if not order:
        return 0.0
    return distance_of_nodes(inst, [sat] + list(order) + [sat])


def _build_solution_from_orders(inst: Instance2E, base: Solution2E, orders_by_sat: Dict[int, List[List[int]]], exact_threshold: int, *, exact_small_threshold: int = 6) -> Optional[Solution2E]:
    """
    Builds a full second-level solution from route orders by materializing each
    route and reconstructing the associated metadata.
    """
    assignment: Dict[int, int] = {}
    rel_from_sat: Dict[int, float] = {}
    t_rel_arr: Dict[int, float] = {}
    routes_lvl2: Dict[int, List[Route]] = {}
    base_meta = dict(base.meta)
    base_vids: Dict[Tuple[int, int], int] = {}
    cache: Dict[Tuple[int, Tuple[int, ...], bool], Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]] = {}
    
    # Preserve original vehicle ids whenever a route index matches the base solution
    for sat, routes in base.routes_lvl2.items():
        for idx, r in enumerate(routes):
            base_vids[(int(sat), idx)] = int(r.vehicle_id)

    # Materialize each client order into a feasible route and rebuild the route-level metadata
    for sat, route_orders in orders_by_sat.items():
        sat_routes: List[Route] = []
        for ridx, order in enumerate(route_orders):
            if not order:
                continue
            _, mat = _route_cost(inst, sat, order, exact_threshold, cache=cache, use_exact=len(order) <= exact_small_threshold)
            if mat is None:
                return None
            nodes, t_map, rel_map, _ = mat
            for c in order:
                assignment[int(c)] = int(sat)
            t_rel_arr.update({int(k): float(v) for k, v in t_map.items()})
            rel_from_sat.update({int(k): float(v) for k, v in rel_map.items()})
            vid = base_vids.get((int(sat), ridx), 100000 + len(sat_routes))
            sat_routes.append(Route(nodes=list(nodes), vehicle_id=vid))
        if sat_routes:
            routes_lvl2[int(sat)] = sat_routes
            
    # Recompute the satellite timing summary and assemble the final solution object
    latest = _latest_sat_arrival(inst, assignment, rel_from_sat)
    sol = Solution2E(
        assignment=assignment,
        routes_lvl2=routes_lvl2,
        routes_lvl1=list(base.routes_lvl1),
        meta=base_meta,
    )
    sol.meta.update({
        "t_rel_arr": t_rel_arr,
        "rel_from_sat": rel_from_sat,
        "latest_sat_arrival": latest,
        "unserved_clients": [],
        "infeasible": False,
    })
    return sol


def _dist2(inst: Instance2E, sol: Solution2E) -> float:
    """Computes the total distance traveled by all second-level routes of a solution."""
    return sum(distance_of_nodes(inst, r.nodes) for rs in sol.routes_lvl2.values() for r in rs)


def _orders_total_seq_cost(inst: Instance2E, orders_by_sat: Dict[int, List[List[int]]]) -> float:
    """Computes the total sequential route cost induced by all client orders across satellites."""
    total = 0.0
    for sat, routes in orders_by_sat.items():
        for order in routes:
            total += _route_seq_cost(inst, sat, order)
    return total


def _changed_keys(base_orders: Dict[int, List[List[int]]], cand_orders: Dict[int, List[List[int]]]) -> List[Tuple[int, int]]:
    """Identifies the satellite-route positions whose client orders differ between two order dictionaries."""
    keys = set()
    sats = set(base_orders.keys()) | set(cand_orders.keys())
    
    # Compare route orders satellite by satellite and route index by route index
    for sat in sats:
        a = base_orders.get(sat, [])
        b = cand_orders.get(sat, [])
        m = max(len(a), len(b))
        for ridx in range(m):
            oa = a[ridx] if ridx < len(a) else []
            ob = b[ridx] if ridx < len(b) else []
            if oa != ob:
                keys.add((sat, ridx))
    return sorted(keys)


def _candidate_real_cost(
    inst: Instance2E,
    base_orders: Dict[int, List[List[int]]],
    cand_orders: Dict[int, List[List[int]]],
    base_route_costs: Dict[Tuple[int, int], float],
    exact_threshold: int,
    *,
    exact_small_threshold: int = 6,
    cache: Optional[Dict[Tuple[int, Tuple[int, ...], bool], Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]]] = None,
) -> float:
    """Computes the real route cost of a candidate order dictionary by re-evaluating only the modified routes."""
    total = sum(float(v) for v in base_route_costs.values())
    
    # Remove the contribution of all routes that changed compared with the base solution
    for key in _changed_keys(base_orders, cand_orders):
        total -= float(base_route_costs.get(key, 0.0))
        
    # Recompute the cost of each changed route using exact or constructive materialization
    for sat, ridx in _changed_keys(base_orders, cand_orders):
        routes = cand_orders.get(sat, [])
        order = routes[ridx] if ridx < len(routes) else []
        if not order:
            continue
        c, mat = _route_cost(inst, sat, order, exact_threshold, cache=cache, use_exact=len(order) <= exact_small_threshold)
        if mat is None:
            return float("inf")
        total += c
    return total


def _route_load(inst: Instance2E, order: List[int]) -> float:
    """Computes the total demand carried by a client order."""
    return sum(float(inst.node_by_id[c].demand) for c in order)


def _optimize_route_recharges(inst: Instance2E, sat: int, route: Route) -> Route:
    """Improves one route by cleaning up unnecessary charging stops and testing beneficial station insertions."""
    nodes = list(route.nodes)
    sim = _simulate_explicit_route(inst, sat, nodes)
    if sim is None:
        return route
    
    # Start from a cleaned feasible route and keep improving it while better variants are found
    best_nodes, _, _, best_dist = _recharge_cleanup(inst, sat, sim)
    improved = True
    while improved:
        improved = False
        current_nodes = list(best_nodes)
        current_dist = float(best_dist)
        
        # Try inserting one promising charging station inside each arc of the route
        for idx in range(len(current_nodes) - 1):
            a = current_nodes[idx]
            b = current_nodes[idx + 1]
            candidates = sorted(
                inst.station_ids,
                key=lambda st: _dist(inst, a, st) + _dist(inst, st, b) - _dist(inst, a, b),
            )[:6]
            for st in candidates:
                if st == a or st == b:
                    continue
                cand_nodes = current_nodes[: idx + 1] + [int(st)] + current_nodes[idx + 1 :]
                sim2 = _simulate_explicit_route(inst, sat, cand_nodes)
                if sim2 is None:
                    continue
                sim2 = _recharge_cleanup(inst, sat, sim2)
                if sim2[3] + EPS < current_dist:
                    best_nodes, _, _, best_dist = sim2
                    improved = True
                    break
            if improved:
                break
    if list(best_nodes) != list(route.nodes):
        return Route(nodes=list(best_nodes), vehicle_id=route.vehicle_id)
    return route


def optimize_solution_recharges(inst: Instance2E, sol: Solution2E) -> LSResult:
    """Optimizes charging decisions on all second-level routes of a solution and rebuilds consistency if needed."""
    debug: List[str] = []
    routes_lvl2_new: Dict[int, List[Route]] = {}
    improved = False
    
    # Optimize the charging pattern of each second-level route independently
    for sat, routes in sol.routes_lvl2.items():
        sat = int(sat)
        new_routes: List[Route] = []
        for r in routes:
            before = distance_of_nodes(inst, r.nodes)
            new_r = _optimize_route_recharges(inst, sat, r)
            after = distance_of_nodes(inst, new_r.nodes)
            if after + EPS < before:
                improved = True
                debug.append(f"recharge_opt:sat={sat}:v={r.vehicle_id}:dist={before:.3f}->{after:.3f}")
            new_routes.append(new_r)
        routes_lvl2_new[sat] = new_routes
    if not improved:
        debug.append('recharge_opt:no_improvement')
        return LSResult(sol, False, 0, _dist2(inst, sol), debug)
    
    # Build the improved solution and replay second-level routes if first-level routes are present
    new_sol = Solution2E(assignment=dict(sol.assignment), routes_lvl2=routes_lvl2_new, routes_lvl1=list(sol.routes_lvl1), meta=dict(sol.meta))
    if new_sol.routes_lvl1:
        A_s = simulate_truck_arrivals(inst, new_sol.routes_lvl1)
        rebuilt, _, _, _, dbg2, _ = repair_and_replay_lvl2_absolute(inst, new_sol, A_s, max_attempts=2)
        rebuilt = _refresh_solution_consistency(inst, rebuilt)
        new_sol = rebuilt
        debug.extend(list(dbg2))
    new_sol.meta['post_acs_debug'] = list(new_sol.meta.get('post_acs_debug', [])) + debug
    new_sol.meta['recharge_opt_applied'] = True
    return LSResult(new_sol, True, 1, _dist2(inst, new_sol), debug)


def _enumerate_relocate_moves(inst: Instance2E, base_orders: Dict[int, List[List[int]]]):
    """Yields candidate solutions obtained by relocating one client from one route position to another."""
    sat_keys = list(base_orders.keys())
    
    # Try removing each client from its current route and reinserting it into every possible target position
    for sat_a in sat_keys:
        for ra, order_a in enumerate(base_orders[sat_a]):
            for i, _client in enumerate(order_a):
                for sat_b in sat_keys:
                    for rb, order_b in enumerate(base_orders[sat_b]):
                        for j in range(len(order_b) + 1):
                            if sat_a == sat_b and ra == rb and (j == i or j == i + 1):
                                continue
                            cand = _clone_orders(base_orders)
                            moved = cand[sat_a][ra].pop(i)
                            insert_pos = j
                            if sat_a == sat_b and ra == rb and j > i:
                                insert_pos = j - 1
                            cand[sat_b][rb].insert(insert_pos, moved)
                            
                            # Enforce capacity feasibility when the move changes the target route load
                            if sat_a != sat_b:
                                if _route_load(inst, cand[sat_b][rb]) > float(inst.Q2) + 1e-9:
                                    continue
                            
                            # Remove empty routes and empty satellite entries created by the move
                            if not cand[sat_a][ra]:
                                cand[sat_a].pop(ra)
                                if not cand[sat_a]:
                                    del cand[sat_a]
                            yield cand


def _enumerate_swap_moves(inst: Instance2E, base_orders: Dict[int, List[List[int]]]):
    """Yields candidate solutions obtained by swapping two clients between route positions."""
    sat_keys = list(base_orders.keys())
    
    # Exchange one client from a route position with another client from the same or another route
    for sat_a in sat_keys:
        for ra, order_a in enumerate(base_orders[sat_a]):
            for i, _ca in enumerate(order_a):
                for sat_b in sat_keys:
                    for rb, order_b in enumerate(base_orders[sat_b]):
                        start_j = i + 1 if (sat_a == sat_b and ra == rb) else 0
                        for j in range(start_j, len(order_b)):
                            cand = _clone_orders(base_orders)
                            cand[sat_a][ra][i], cand[sat_b][rb][j] = cand[sat_b][rb][j], cand[sat_a][ra][i]
                            
                            # Keep only swaps that preserve route capacity feasibility
                            if _route_load(inst, cand[sat_a][ra]) > float(inst.Q2) + 1e-9:
                                continue
                            if _route_load(inst, cand[sat_b][rb]) > float(inst.Q2) + 1e-9:
                                continue
                            yield cand


def _enumerate_two_opt_moves(base_orders: Dict[int, List[List[int]]]):
    """Yields candidate solutions obtained by applying an intra-route 2-opt reversal."""
    # Reverse every possible subsequence of sufficient length inside each route
    for sat, routes in base_orders.items():
        for ridx, order in enumerate(routes):
            n = len(order)
            if n < 4:
                continue
            for i in range(n - 2):
                for j in range(i + 2, n + 1):
                    cand = _clone_orders(base_orders)
                    cand[sat][ridx] = order[:i] + list(reversed(order[i:j])) + order[j:]
                    yield cand


def _enumerate_or_opt_moves(base_orders: Dict[int, List[List[int]]], block_len: int):
    """Yields candidate solutions obtained by relocating a contiguous block within the same route."""
    # Remove each block of the given length and reinsert it at another position in the route
    for sat, routes in base_orders.items():
        for ridx, order in enumerate(routes):
            n = len(order)
            if n < block_len + 1:
                continue
            for i in range(0, n - block_len + 1):
                block = order[i : i + block_len]
                rest = order[:i] + order[i + block_len :]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    cand = _clone_orders(base_orders)
                    cand[sat][ridx] = rest[:j] + block + rest[j:]
                    if cand[sat][ridx] != order:
                        yield cand


def _enumerate_two_opt_star_moves(inst: Instance2E, base_orders: Dict[int, List[List[int]]]):
    """Yields candidate solutions obtained by exchanging route tails between two different routes."""
    route_keys = [(sat, ridx) for sat, routes in base_orders.items() for ridx, _ in enumerate(routes)]
    
    # Split two routes at every possible cut pair and exchange their tails
    for idx_a in range(len(route_keys)):
        sat_a, ra = route_keys[idx_a]
        order_a = base_orders[sat_a][ra]
        for idx_b in range(idx_a + 1, len(route_keys)):
            sat_b, rb = route_keys[idx_b]
            order_b = base_orders[sat_b][rb]
            for cut_a in range(1, len(order_a)):
                for cut_b in range(1, len(order_b)):
                    new_a = order_a[:cut_a] + order_b[cut_b:]
                    new_b = order_b[:cut_b] + order_a[cut_a:]
                    if not new_a or not new_b:
                        continue
                    if _route_load(inst, new_a) > float(inst.Q2) + 1e-9:
                        continue
                    if _route_load(inst, new_b) > float(inst.Q2) + 1e-9:
                        continue
                    cand = _clone_orders(base_orders)
                    cand[sat_a][ra] = new_a
                    cand[sat_b][rb] = new_b
                    yield cand


def _best_insertion_position(inst: Instance2E, sat: int, route: List[int], client: int) -> int:
    """Returns the insertion position that minimizes the extra travel distance when adding a client to a route."""
    best_pos = 0
    best_delta = float("inf")
    prev = sat
    
    # Evaluate the marginal distance increase of inserting the client at each possible position
    for pos in range(len(route) + 1):
        nxt = sat if pos == len(route) else route[pos]
        delta = _dist(inst, prev, client) + _dist(inst, client, nxt) - _dist(inst, prev, nxt)
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
        if pos < len(route):
            prev = route[pos]
    return best_pos


def _enumerate_satellite_cluster_reassign_moves(inst: Instance2E, base_orders: Dict[int, List[List[int]]]):
    """Yields candidate solutions obtained by moving short client blocks from one satellite to another."""
    used_sats = _used_satellites(base_orders)
    if len(used_sats) < 2:
        return
    yielded = 0
    for sat_a in used_sats:
        for ra, order_a in enumerate(base_orders[sat_a]):
            n = len(order_a)
            if n == 0:
                continue
            # Focus on segments that may belong to another satellite.
            for block_len in range(1, min(3, n) + 1):
                for i in range(0, n - block_len + 1):
                    block = list(order_a[i:i + block_len])
                    block_demand = _route_load(inst, block)
                    for sat_b in sorted(inst.satellite_ids):
                        sat_b = int(sat_b)
                        if sat_b == sat_a:
                            continue
                        # Coarse filter: only keep target satellites that are competitive for this block.
                        mean_to_a = sum(_dist(inst, sat_a, c) for c in block) / len(block)
                        mean_to_b = sum(_dist(inst, sat_b, c) for c in block) / len(block)
                        if mean_to_b > mean_to_a + 15.0:
                            continue
                        # New dedicated route at the target satellite.
                        cand = _clone_orders(base_orders)
                        cand[sat_a][ra] = order_a[:i] + order_a[i + block_len:]
                        if not cand[sat_a][ra]:
                            cand[sat_a].pop(ra)
                            if not cand[sat_a]:
                                del cand[sat_a]
                        cand.setdefault(sat_b, []).append(list(block))
                        yield cand
                        yielded += 1
                        if yielded >= 80:
                            return
                        # Insert the block into existing routes of the target satellite.
                        for rb, route_b in enumerate(base_orders.get(sat_b, [])):
                            if _route_load(inst, route_b) + block_demand > float(inst.Q2) + 1e-9:
                                continue
                            for pos in range(len(route_b) + 1):
                                cand = _clone_orders(base_orders)
                                cand[sat_a][ra] = order_a[:i] + order_a[i + block_len:]
                                if not cand[sat_a][ra]:
                                    cand[sat_a].pop(ra)
                                    if not cand[sat_a]:
                                        del cand[sat_a]
                                cand[sat_b][rb] = list(route_b[:pos]) + list(block) + list(route_b[pos:])
                                yield cand
                                yielded += 1
                                if yielded >= 80:
                                    return


def _enumerate_route_reassign_moves(inst: Instance2E, base_orders: Dict[int, List[List[int]]]):
    """Yields candidate solutions obtained by moving a full route from one satellite to another."""
    used_sats = _used_satellites(base_orders)
    if len(used_sats) < 2:
        return
    route_keys = [(sat, ridx, route) for sat, routes in base_orders.items() for ridx, route in enumerate(routes)]
    route_keys.sort(key=lambda x: (len(x[2]), _sat_distance_to_route_centroid(inst, x[0], x[2])), reverse=True)
    yielded = 0
    
    # Reassign complete routes to another satellite, either directly or by greedily spilling their clients
    for sat_a, ridx_a, route in route_keys:
        if not route:
            continue
        for sat_b in sorted(inst.satellite_ids):
            sat_b = int(sat_b)
            if sat_b == sat_a:
                continue
            
            # Entire route moved as a dedicated route under another satellite.
            cand = _clone_orders(base_orders)
            del cand[sat_a][ridx_a]
            if not cand[sat_a]:
                del cand[sat_a]
            cand.setdefault(sat_b, []).append(list(route))
            yield cand
            yielded += 1
            if yielded >= 40:
                return
            
            # Greedy spill of the whole route into target satellite existing routes.
            partial = _clone_orders(base_orders)
            del partial[sat_a][ridx_a]
            if not partial[sat_a]:
                del partial[sat_a]
            feasible = True
            for c in route:
                choices = []
                for rb, route_b in enumerate(partial.get(sat_b, [])):
                    if _route_load(inst, route_b) + float(inst.node_by_id[c].demand) > float(inst.Q2) + 1e-9:
                        continue
                    pos = _best_insertion_position(inst, sat_b, route_b, c)
                    prev = sat_b if pos == 0 else route_b[pos - 1]
                    nxt = sat_b if pos == len(route_b) else route_b[pos]
                    delta = _dist(inst, prev, c) + _dist(inst, c, nxt) - _dist(inst, prev, nxt)
                    choices.append((delta, rb, pos))
                if not choices:
                    feasible = False
                    break
                _, rb, pos = min(choices)
                partial[sat_b][rb].insert(pos, c)
            if feasible:
                yield partial
                yielded += 1
                if yielded >= 40:
                    return


def _enumerate_satellite_close_moves(inst: Instance2E, base_orders: Dict[int, List[List[int]]]):
    """Yields candidate solutions obtained by closing a lightly used satellite and redistributing its clients."""
    used_sats = _used_satellites(base_orders)
    if len(used_sats) < 2:
        return
    sat_rank = []
    for sat in used_sats:
        clients = [c for route in base_orders[sat] for c in route]
        sat_rank.append((len(clients), sat))
    sat_rank.sort()
    
    # Focus on the lightest used satellites and try to absorb all their clients elsewhere
    for _, sat_drop in sat_rank[:2]:
        clients = [c for route in base_orders[sat_drop] for c in route]
        if not clients or len(clients) > 8:
            continue
        partial = {int(s): [list(rt) for rt in routes] for s, routes in base_orders.items() if int(s) != int(sat_drop)}
        feasible = True
        for c in clients:
            best = None
            best_delta = float('inf')
            for sat in sorted(inst.satellite_ids):
                sat = int(sat)
                if sat == sat_drop:
                    continue
                for ridx, route in enumerate(partial.get(sat, [])):
                    if _route_load(inst, route) + float(inst.node_by_id[c].demand) > float(inst.Q2) + 1e-9:
                        continue
                    pos = _best_insertion_position(inst, sat, route, c)
                    prev = sat if pos == 0 else route[pos - 1]
                    nxt = sat if pos == len(route) else route[pos]
                    delta = _dist(inst, prev, c) + _dist(inst, c, nxt) - _dist(inst, prev, nxt)
                    if delta < best_delta:
                        best_delta = delta
                        best = (sat, ridx, pos)
                
                # Also allow reopening the client as a dedicated route at a different satellite
                if best is None or _dist(inst, sat, c) + _dist(inst, c, sat) < best_delta:
                    best = (sat, None, None)
                    best_delta = _dist(inst, sat, c) + _dist(inst, c, sat)
            if best is None:
                feasible = False
                break
            sat, ridx, pos = best
            partial.setdefault(sat, [])
            if ridx is None:
                partial[sat].append([c])
            else:
                partial[sat][ridx].insert(pos, c)
        if feasible:
            yield partial

def _enumerate_route_elimination_moves(inst: Instance2E, base_orders: Dict[int, List[List[int]]]):
    """Yields candidate solutions obtained by removing a small route and reinserting its clients elsewhere."""
    route_keys = [(sat, ridx, route) for sat, routes in base_orders.items() for ridx, route in enumerate(routes)]
    route_keys.sort(key=lambda x: (len(x[2]), _route_load(inst, x[2])))
    
    # Eliminate only small routes and try to absorb all their clients into the remaining routes
    for sat_drop, ridx_drop, route_drop in route_keys:
        if len(route_keys) <= 1 or len(route_drop) > 3:
            continue
        cand = _clone_orders(base_orders)
        clients = list(cand[sat_drop][ridx_drop])
        del cand[sat_drop][ridx_drop]
        if not cand[sat_drop]:
            del cand[sat_drop]
        feasible = True
        for c in clients:
            best_choice = None
            best_delta = float("inf")
            for sat, routes in cand.items():
                for ridx, route in enumerate(routes):
                    if _route_load(inst, route) + float(inst.node_by_id[c].demand) > float(inst.Q2) + 1e-9:
                        continue
                    pos = _best_insertion_position(inst, sat, route, c)
                    prev = sat if pos == 0 else route[pos - 1]
                    nxt = sat if pos == len(route) else route[pos]
                    delta = _dist(inst, prev, c) + _dist(inst, c, nxt) - _dist(inst, prev, nxt)
                    if delta < best_delta:
                        best_delta = delta
                        best_choice = (sat, ridx, pos)
            if best_choice is None:
                feasible = False
                break
            sat, ridx, pos = best_choice
            cand[sat][ridx].insert(pos, c)
        if feasible:
            yield cand


def _rvnd_step(
    inst: Instance2E,
    base_orders: Dict[int, List[List[int]]],
    base_real_cost: float,
    base_seq_cost: float,
    base_route_costs: Dict[Tuple[int, int], float],
    exact_threshold: int,
    *,
    exact_small_threshold: int = 6,
    max_neighbors: int = 120,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[Dict[int, List[List[int]]]], str, float, int]:
    """
    Executes one RVND step by exploring multiple neighborhoods until an 
    improving move is found or the search budget is exhausted.
    """
    cache: Dict[Tuple[int, Tuple[int, ...], bool], Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]] = {}
    tested = 0
    rng = rng or random.Random(0)

    neighborhoods = [
        ("relocate", lambda bo: _enumerate_relocate_moves(inst, bo)),
        ("swap", lambda bo: _enumerate_swap_moves(inst, bo)),
        ("two_opt", lambda bo: _enumerate_two_opt_moves(bo)),
        ("or_opt_2", lambda bo: _enumerate_or_opt_moves(bo, 2)),
        ("or_opt_3", lambda bo: _enumerate_or_opt_moves(bo, 3)),
        ("two_opt_star", lambda bo: _enumerate_two_opt_star_moves(inst, bo)),
        ("route_eliminate", lambda bo: _enumerate_route_elimination_moves(inst, bo)),
        ("sat_cluster_reassign", lambda bo: _enumerate_satellite_cluster_reassign_moves(inst, bo)),
        ("route_reassign", lambda bo: _enumerate_route_reassign_moves(inst, bo)),
        ("satellite_close", lambda bo: _enumerate_satellite_close_moves(inst, bo)),
    ]
    active = list(neighborhoods)
    rng.shuffle(active)
    
    # Explore neighborhoods in random order and stop at the first improving move
    while active:
        idx = rng.randrange(len(active))
        name, builder = active[idx]
        improved_here = False
        for cand in builder(base_orders):
            tested += 1
            cand_seq_cost = _orders_total_seq_cost(inst, cand)
            
            # Discard clearly unattractive candidates using the cheap sequential surrogate cost
            if cand_seq_cost > base_seq_cost + 20.0:
                if tested >= max_neighbors:
                    return None, "neighbor_cap", base_real_cost, tested
                continue
            
            # Evaluate the real cost only for promising candidates
            cand_real_cost = _candidate_real_cost(
                inst,
                base_orders,
                cand,
                base_route_costs,
                exact_threshold,
                exact_small_threshold=exact_small_threshold,
                cache=cache,
            )
            if cand_real_cost + EPS < base_real_cost:
                return cand, f"rvnd:{name}", cand_real_cost, tested
            if tested >= max_neighbors:
                return None, "neighbor_cap", base_real_cost, tested
        if not improved_here:
            active.pop(idx)
    return None, "rvnd_no_improving_move", base_real_cost, tested




def _used_satellites(orders_by_sat: Dict[int, List[List[int]]]) -> List[int]:
    """Returns the list of satellites that currently have at least one non-empty route."""
    return [int(s) for s, routes in orders_by_sat.items() if any(routes)]


def _route_centroid(inst: Instance2E, route: List[int]) -> Tuple[float, float]:
    """Computes the geometric centroid of the client nodes contained in a route."""
    pts = [(float(inst.node_by_id[c].x), float(inst.node_by_id[c].y)) for c in route if c in inst.client_ids]
    if not pts:
        return (0.0, 0.0)
    return (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))


def _sat_distance_to_route_centroid(inst: Instance2E, sat: int, route: List[int]) -> float:
    """Computes the Euclidean distance between a satellite and the centroid of a route."""
    cx, cy = _route_centroid(inst, route)
    sn = inst.node_by_id[int(sat)]
    dx = float(sn.x) - cx
    dy = float(sn.y) - cy
    return (dx * dx + dy * dy) ** 0.5

def _count_lvl2_routes(orders_by_sat: Dict[int, List[List[int]]]) -> int:
    """Counts the total number of second-level routes across all satellites."""
    return sum(len(routes) for routes in orders_by_sat.values())


def _flatten_clients(orders_by_sat: Dict[int, List[List[int]]]) -> List[int]:
    """Flattens all client orders into a single list of client identifiers."""
    out: List[int] = []
    for routes in orders_by_sat.values():
        for route in routes:
            out.extend(int(c) for c in route)
    return out


def _remove_clients_from_orders(base_orders: Dict[int, List[List[int]]], removed: List[int]) -> Dict[int, List[List[int]]]:
    """Builds a new order dictionary after removing a given set of clients from all routes."""
    removed_set = set(int(c) for c in removed)
    cand: Dict[int, List[List[int]]] = {}
    
    # Remove the selected clients from every route and keep only non-empty routes
    for sat, routes in base_orders.items():
        kept_routes: List[List[int]] = []
        for route in routes:
            kept = [int(c) for c in route if int(c) not in removed_set]
            if kept:
                kept_routes.append(kept)
        if kept_routes:
            cand[int(sat)] = kept_routes
    return cand


def _sample_destroy_set_base(inst: Instance2E, base_orders: Dict[int, List[List[int]]], rng: random.Random, destroy_count: int) -> List[int]:
    """Samples a destroy set by combining costly clients, related clients, and random completion."""
    clients = _flatten_clients(base_orders)
    if not clients:
        return []
    destroy_count = max(1, min(int(destroy_count), len(clients)))
    
    # Estimate a simple marginal sequence cost for each client inside its current route
    seq_costs: Dict[int, float] = {}
    for sat, routes in base_orders.items():
        for route in routes:
            prev = sat
            for idx, c in enumerate(route):
                nxt = sat if idx == len(route) - 1 else route[idx + 1]
                seq_costs[int(c)] = float(_dist(inst, prev, c) + _dist(inst, c, nxt) - _dist(inst, prev, nxt))
                prev = c
    worst_sorted = sorted(clients, key=lambda c: seq_costs.get(int(c), 0.0), reverse=True)
    removed: List[int] = []

    n_worst = min(len(worst_sorted), max(1, destroy_count // 2))
    removed.extend(int(c) for c in worst_sorted[:n_worst])

    remaining = [int(c) for c in clients if int(c) not in set(removed)]
    if remaining and len(removed) < destroy_count:
        seed = int(rng.choice(remaining))
        related = sorted(
            remaining,
            key=lambda c: (
                _dist(inst, seed, int(c)),
                abs(float(inst.tw_early(seed)) - float(inst.tw_early(int(c)))),
            ),
        )
        for c in related:
            if c not in removed:
                removed.append(int(c))
            if len(removed) >= destroy_count:
                break

    remaining = [int(c) for c in clients if int(c) not in set(removed)]
    rng.shuffle(remaining)
    for c in remaining:
        if len(removed) >= destroy_count:
            break
        removed.append(int(c))
    return removed[:destroy_count]


def _sample_destroy_set_inter_sat(inst: Instance2E, base_orders: Dict[int, List[List[int]]], rng: random.Random, destroy_count: int) -> List[int]:
    """Builds a destroy set biased toward clients that may benefit from reassignment to another satellite."""
    used_sats = _used_satellites(base_orders)
    if len(used_sats) < 2:
        return _sample_destroy_set_base(inst, base_orders, rng, destroy_count)
    scored = []
    
    # Score each used satellite by how promising it is as a donor for inter-satellite reassignment
    for sat in used_sats:
        clients = [int(c) for route in base_orders[sat] for c in route]
        if not clients:
            continue
        alt_gain = []
        for c in clients:
            cur = _dist(inst, sat, c)
            best_alt = min(_dist(inst, s2, c) for s2 in used_sats if s2 != sat)
            alt_gain.append(cur - best_alt)
        scored.append((max(alt_gain) if alt_gain else -1e9, -len(clients), sat, clients))
    scored.sort(reverse=True)
    _, _, donor_sat, donor_clients = scored[0]
    
    # Remove first the clients that look most attractive for another satellite
    donor_clients = sorted(
        donor_clients,
        key=lambda c: min(_dist(inst, s2, c) for s2 in used_sats if s2 != donor_sat) - _dist(inst, donor_sat, c)
    )
    removed = []
    for c in donor_clients:
        removed.append(int(c))
        if len(removed) >= destroy_count:
            break
    
    # Complete the destroy set with the standard destroy policy if needed
    if len(removed) < destroy_count:
        extra = _sample_destroy_set_base(inst, base_orders, rng, destroy_count - len(removed))
        for c in extra:
            if c not in removed:
                removed.append(int(c))
            if len(removed) >= destroy_count:
                break
    return removed[:destroy_count]


def _sample_destroy_route_bundle(inst: Instance2E, base_orders: Dict[int, List[List[int]]], rng: random.Random, destroy_count: int) -> List[int]:
    """Builds a destroy set by targeting a whole promising route bundle before completing it if necessary."""
    route_keys = [(sat, route) for sat, routes in base_orders.items() for route in routes if route]
    if not route_keys:
        return []
    
    # Select the most promising donor route according to its size and satellite-centroid mismatch
    route_keys.sort(key=lambda x: (len(x[1]), _sat_distance_to_route_centroid(inst, x[0], x[1])), reverse=True)
    sat, route = route_keys[0]
    removed = list(route[:destroy_count])
    
    # Complete the destroy set with the default destroy policy if the chosen route is too short
    if len(removed) < destroy_count:
        rest = _sample_destroy_set_base(inst, base_orders, rng, destroy_count - len(removed))
        for c in rest:
            if c not in removed:
                removed.append(int(c))
            if len(removed) >= destroy_count:
                break
    return removed[:destroy_count]


def _sample_destroy_set(inst: Instance2E, base_orders: Dict[int, List[List[int]]], rng: random.Random, destroy_count: int) -> List[int]:
    """Samples a destroy set by randomly choosing among several destroy strategies."""
    modes = ['base', 'inter_sat', 'route_bundle']
    if len(_used_satellites(base_orders)) < 2:
        modes = ['base', 'route_bundle']
    mode = rng.choice(modes)
    if mode == 'inter_sat':
        return _sample_destroy_set_inter_sat(inst, base_orders, rng, destroy_count)
    if mode == 'route_bundle':
        return _sample_destroy_route_bundle(inst, base_orders, rng, destroy_count)
    return _sample_destroy_set_base(inst, base_orders, rng, destroy_count)




def _total_real_cost_from_orders(
    inst: Instance2E,
    orders_by_sat: Dict[int, List[List[int]]],
    exact_threshold: int,
    *,
    exact_small_threshold: int = 6,
    cache: Optional[Dict[Tuple[int, Tuple[int, ...], bool], Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]]] = None,
) -> float:
    """Computes the total real routing cost induced by a dictionary of client orders."""
    total = 0.0
    
    # Materialize every route order and accumulate its real routing cost
    for sat, routes in orders_by_sat.items():
        for order in routes:
            c, mat = _route_cost(inst, int(sat), list(order), exact_threshold, cache=cache, use_exact=len(order) <= exact_small_threshold)
            if mat is None:
                return float('inf')
            total += float(c)
    return total

def _evaluate_insertion_candidates(
    inst: Instance2E,
    orders_by_sat: Dict[int, List[List[int]]],
    client: int,
    exact_threshold: int,
    *,
    exact_small_threshold: int = 6,
    route_limit: Optional[int] = None,
    cache: Optional[Dict[Tuple[int, Tuple[int, ...], bool], Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]]] = None,
) -> List[Tuple[float, Dict[int, List[List[int]]]]]:
    """Evaluates feasible insertion candidates for one client across all satellites and routes."""
    candidates: List[Tuple[float, Dict[int, List[List[int]]]]] = []
    demand = float(inst.node_by_id[int(client)].demand)
    
    # Try inserting the client into every feasible position of every existing route
    for sat in inst.satellite_ids:
        sat = int(sat)
        sat_routes = orders_by_sat.get(sat, [])
        for ridx, route in enumerate(sat_routes):
            if _route_load(inst, route) + demand > float(inst.Q2) + 1e-9:
                continue
            for pos in range(len(route) + 1):
                cand = _clone_orders(orders_by_sat)
                cand.setdefault(sat, [])
                cand[sat][ridx] = list(route[:pos]) + [int(client)] + list(route[pos:])
                total = _candidate_real_cost(
                    inst,
                    orders_by_sat,
                    cand,
                    {},
                    exact_threshold,
                    exact_small_threshold=exact_small_threshold,
                    cache=cache,
                )
                if total < float('inf'):
                    candidates.append((float(total), cand))
        
        # Also evaluate opening a new singleton route if the route budget still allows it
        if route_limit is None or _count_lvl2_routes(orders_by_sat) < int(route_limit):
            cand = _clone_orders(orders_by_sat)
            cand.setdefault(sat, []).append([int(client)])
            total = _total_real_cost_from_orders(
                inst,
                cand,
                exact_threshold,
                exact_small_threshold=exact_small_threshold,
                cache=cache,
            )
            if total < float('inf'):
                candidates.append((float(total), cand))

    candidates.sort(key=lambda x: x[0])
    return candidates


def periodic_destroy_repair_lns(
    inst: Instance2E,
    base_sol: Solution2E,
    rng: random.Random,
    *,
    destroy_fraction: float = 0.20,
    min_destroy: int = 2,
    max_destroy: int = 12,
    repair_passes: int = 2,
    exact_threshold: int = 10,
    exact_small_threshold: int = 6,
    max_neighbors: int = 120,
) -> LSResult:
    """Applies a destroy-repair large neighborhood search around a feasible second-level solution."""
    debug: List[str] = []
    
    # Skip the procedure when the base solution is already incomplete or too small to perturb meaningfully
    if base_sol.meta.get('unserved_clients'):
        debug.append('lns_skip:unserved_clients')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)

    base_orders = _clone_client_routes(base_sol)
    clients = _flatten_clients(base_orders)
    if len(clients) <= 1:
        debug.append('lns_skip:not_enough_clients')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)

    destroy_count = max(int(min_destroy), int(round(len(clients) * float(destroy_fraction))))
    destroy_count = min(int(max_destroy), destroy_count, len(clients))
    removed = _sample_destroy_set(inst, base_orders, rng, destroy_count)
    if not removed:
        debug.append('lns_skip:destroy_empty')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)
    
    # Destroy part of the current solution and keep the remaining partial structure
    partial = _remove_clients_from_orders(base_orders, removed)
    debug.append(f'lns_destroy:removed={len(removed)} clients={sorted(int(c) for c in removed)}')

    route_limit = int(inst.nv2)
    unassigned = [int(c) for c in removed]
    cache: Dict[Tuple[int, Tuple[int, ...], bool], Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]] = {}
    
    # Reinsert removed clients one by one using a regret-like repair strategy
    while unassigned:
        scored: List[Tuple[float, float, int, Dict[int, List[List[int]]]]] = []
        for client in list(unassigned):
            cand_list = _evaluate_insertion_candidates(
                inst,
                partial,
                client,
                exact_threshold,
                exact_small_threshold=exact_small_threshold,
                route_limit=route_limit,
                cache=cache,
            )
            if not cand_list:
                continue
            best_cost = float(cand_list[0][0])
            second_cost = float(cand_list[1][0]) if len(cand_list) > 1 else best_cost + 1.0
            regret = second_cost - best_cost
            scored.append((regret, -best_cost, int(client), cand_list[0][1]))

        if not scored:
            debug.append(f'lns_stop:repair_failed remaining={sorted(unassigned)}')
            return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)

        scored.sort(reverse=True)
        _, _, chosen_client, chosen_orders = scored[0]
        partial = chosen_orders
        unassigned.remove(chosen_client)
        debug.append(f'lns_repair:client={chosen_client}:remaining={len(unassigned)}')
        
    # Rebuild a full solution from the repaired route orders
    cand_sol = _build_solution_from_orders(inst, base_sol, partial, exact_threshold, exact_small_threshold=exact_small_threshold)
    if cand_sol is None:
        debug.append('lns_stop:rebuild_failed')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)

    # Stabilize the rebuilt structure with a short local search phase
    post = cand_sol
    total_moves = 0
    adaptive_neighbors = max(20, int(max_neighbors))
    for _ in range(max(0, int(repair_passes))):
        ls_res = intensify_lvl2_solution(
            inst,
            post,
            max_passes=max(1, min(4, int(repair_passes) + 1)),
            exact_threshold=exact_threshold,
            exact_small_threshold=exact_small_threshold,
            max_neighbors=adaptive_neighbors,
        )
        debug.extend(list(ls_res.debug))
        if not ls_res.improved:
            break
        total_moves += int(ls_res.moves)
        post = ls_res.solution
        
    # Accept the candidate only if it strictly improves the second-level distance
    base_cost = _dist2(inst, base_sol)
    cand_cost = _dist2(inst, post)
    if cand_cost + EPS < base_cost:
        post.meta['post_acs_debug'] = list(post.meta.get('post_acs_debug', [])) + debug
        post.meta['lns_applied'] = True
        post.meta['lns_removed_clients'] = sorted(int(c) for c in removed)
        return LSResult(post, True, max(1, total_moves), cand_cost, debug)

    debug.append(f'lns_reject:dist2={base_cost:.3f}->{cand_cost:.3f}')
    return LSResult(base_sol, False, 0, base_cost, debug)

def intensify_lvl2_solution(
    inst: Instance2E,
    base_sol: Solution2E,
    max_passes: int = 12,
    exact_threshold: int = 10,
    *,
    exact_small_threshold: int = 6,
    max_neighbors: int = 120,
    rng: Optional[random.Random] = None,
    recharge_every: int = 1,
    time_budget_s: Optional[float] = None,
) -> LSResult:
    """
    Improves a feasible second-level solution with an RVND-style local search 
    and periodic recharge optimization.
    """
    debug: List[str] = []
    
    # Abort immediately when the starting solution is incomplete
    if base_sol.meta.get("unserved_clients"):
        debug.append("ls_skip:unserved_clients")
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)

    rng = rng or random.Random(0)
    deadline = None if time_budget_s is None else (time.perf_counter() + max(0.0, float(time_budget_s)))
    cur_sol = base_sol
    cur_cost = _dist2(inst, cur_sol)
    moves = 0
    improved = False
    
    # Start with a recharge optimization pass before exploring neighborhoods
    recharge_res0 = optimize_solution_recharges(inst, cur_sol)
    if recharge_res0.improved and recharge_res0.best_dist2 + EPS < cur_cost:
        cur_sol = recharge_res0.solution
        cur_cost = recharge_res0.best_dist2
        improved = True
        moves += 1
        debug.extend(list(recharge_res0.debug))
        
    # Run repeated RVND steps until no improvement or the stopping limits are reached
    while moves < max_passes:
        if _time_exceeded(deadline):
            debug.append("ls_stop:time_budget")
            break
        orders = _clone_client_routes(cur_sol)
        base_seq_cost = _orders_total_seq_cost(inst, orders)
        base_route_costs: Dict[Tuple[int, int], float] = {}
        route_cache: Dict[Tuple[int, Tuple[int, ...], bool], Tuple[float, Optional[Tuple[List[int], Dict[int, float], Dict[int, float], float]]]] = {}
        feasible = True
        
        # Materialize the current route orders to obtain accurate base route costs
        for sat, routes in orders.items():
            for ridx, order in enumerate(routes):
                c, mat = _route_cost(inst, sat, order, exact_threshold, cache=route_cache, use_exact=len(order) <= exact_small_threshold)
                if mat is None:
                    feasible = False
                    break
                base_route_costs[(sat, ridx)] = c
            if not feasible:
                break
        if not feasible:
            debug.append("ls_stop:base_infeasible")
            break
        cur_cost = sum(base_route_costs.values())

        adaptive_neighbors = max(20, int(max_neighbors))
        
        # Tighten the neighborhood budget when little time remains
        if deadline is not None:
            rem = _time_remaining(deadline)
            if rem < 0.25:
                debug.append("ls_stop:time_budget_low")
                break
            if rem < 1.0:
                adaptive_neighbors = max(20, min(adaptive_neighbors, 40))
            elif rem < 2.0:
                adaptive_neighbors = max(30, min(adaptive_neighbors, 80))
        cand_orders, move_name, cand_cost, tested = _rvnd_step(
            inst,
            orders,
            cur_cost,
            base_seq_cost,
            base_route_costs,
            exact_threshold,
            exact_small_threshold=exact_small_threshold,
            max_neighbors=adaptive_neighbors,
            rng=rng,
        )
        debug.append(f"ls_scan:{move_name}:tested={tested}")
        if cand_orders is None:
            break
        
        # Rebuild and optionally recharge-optimize the improving candidate
        cand_sol = _build_solution_from_orders(inst, cur_sol, cand_orders, exact_threshold, exact_small_threshold=exact_small_threshold)
        if cand_sol is None:
            debug.append(f"ls_reject:{move_name}:rebuild_failed")
            break

        if recharge_every > 0 and ((moves + 1) % recharge_every == 0):
            recharge_res = optimize_solution_recharges(inst, cand_sol)
            debug.extend(list(recharge_res.debug))
            if recharge_res.improved:
                cand_sol = recharge_res.solution
        
        # Accept the move only if it yields a true second-level improvement
        cand_cost = _dist2(inst, cand_sol)
        if cand_cost + EPS < cur_cost:
            improved = True
            moves += 1
            debug.append(f"ls_accept:{move_name}:dist2={cur_cost:.3f}->{cand_cost:.3f}")
            cur_sol = cand_sol
            cur_cost = cand_cost
            continue
        debug.append(f"ls_stop:{move_name}:no_real_improvement")
        break
    
    # Finish with one last recharge optimization pass
    final_recharge = optimize_solution_recharges(inst, cur_sol)
    debug.extend(list(final_recharge.debug))
    if final_recharge.improved and final_recharge.best_dist2 + EPS < cur_cost:
        cur_sol = final_recharge.solution
        cur_cost = final_recharge.best_dist2
        improved = True
        moves += 1

    cur_sol.meta["post_acs_debug"] = list(cur_sol.meta.get("post_acs_debug", [])) + debug
    cur_sol.meta["ls_applied"] = improved
    cur_sol.meta["ls_moves"] = moves
    cur_sol.meta["ls_best_dist2"] = cur_cost
    cur_sol.meta["ls_mode"] = "rvnd"
    return LSResult(cur_sol, improved, moves, cur_cost, debug)




def _route_entry_from_route(inst: Instance2E, sat: int, route: Route) -> Optional[RoutePoolEntry]:
    """Builds a route-pool entry from a route after validating and cleaning its explicit charging pattern."""
    clients = tuple(int(n) for n in route.nodes if n in inst.client_ids)
    if not clients:
        return None
    sim = _simulate_explicit_route(inst, int(sat), list(route.nodes))
    if sim is None:
        return None
    sim = _recharge_cleanup(inst, int(sat), sim)
    return RoutePoolEntry(sat=int(sat), clients=clients, nodes=list(sim[0]), cost=float(sim[3]))


def _collect_route_pool_entries(inst: Instance2E, solutions: List[Solution2E], max_entries: int = 160) -> List[RoutePoolEntry]:
    """
    Collects a diverse set of route-pool entries from several solutions while
    keeping the best version of each signature.
    """
    best_by_sig: Dict[Tuple[int, Tuple[int, ...]], RoutePoolEntry] = {}
    
    # Keep only the cheapest version of each satellite-client signature
    for sol in solutions:
        for sat, routes in sol.routes_lvl2.items():
            for r in routes:
                entry = _route_entry_from_route(inst, int(sat), r)
                if entry is None:
                    continue
                sig = (entry.sat, tuple(entry.clients))
                cur = best_by_sig.get(sig)
                if cur is None or entry.cost + EPS < cur.cost:
                    best_by_sig[sig] = entry
    entries = list(best_by_sig.values())
    entries.sort(key=lambda e: (len(e.clients), -e.cost))

    # First keep a balanced subset by route cardinality
    kept: List[RoutePoolEntry] = []
    by_k: Dict[int, int] = {}
    for e in sorted(entries, key=lambda x: (x.cost, len(x.clients), x.sat)):
        k = len(e.clients)
        if by_k.get(k, 0) < 18:
            kept.append(e)
            by_k[k] = by_k.get(k, 0) + 1
        if len(kept) >= max_entries:
            break
        
    # Then complete the pool with globally strong routes if space remains
    if len(kept) < min(len(entries), max_entries):
        existing = {(e.sat, e.clients) for e in kept}
        for e in sorted(entries, key=lambda x: (x.cost / max(1, len(x.clients)), x.cost)):
            if (e.sat, e.clients) in existing:
                continue
            kept.append(e)
            existing.add((e.sat, e.clients))
            if len(kept) >= max_entries:
                break
    return kept


def _build_solution_from_explicit_pool(inst: Instance2E, base: Solution2E, entries: List[RoutePoolEntry]) -> Optional[Solution2E]:
    """Builds a full solution from explicit pooled routes and reconstructs the associated timing metadata."""
    assignment: Dict[int, int] = {}
    rel_from_sat: Dict[int, float] = {}
    t_rel_arr: Dict[int, float] = {}
    routes_lvl2: Dict[int, List[Route]] = {}
    used_clients: Set[int] = set()
    vid = 100000
    
    # Rebuild all routes explicitly and reject overlapping client coverage
    for entry in entries:
        sim = _simulate_explicit_route(inst, entry.sat, list(entry.nodes))
        if sim is None:
            return None
        nodes, t_map, rel_map, _ = sim
        route_clients = [int(c) for c in entry.clients]
        if any(c in used_clients for c in route_clients):
            return None
        used_clients.update(route_clients)
        for c in route_clients:
            assignment[c] = int(entry.sat)
        t_rel_arr.update({int(k): float(v) for k, v in t_map.items()})
        rel_from_sat.update({int(k): float(v) for k, v in rel_map.items()})
        routes_lvl2.setdefault(int(entry.sat), []).append(Route(nodes=list(nodes), vehicle_id=vid))
        vid += 1
        
    # Ensure that the pooled routes cover exactly the same client set as the base solution
    if used_clients != set(int(c) for c in base.assignment.keys()):
        return None
    latest = _latest_sat_arrival(inst, assignment, rel_from_sat)
    sol = Solution2E(assignment=assignment, routes_lvl2=routes_lvl2, routes_lvl1=list(base.routes_lvl1), meta=dict(base.meta))
    sol.meta.update({
        't_rel_arr': t_rel_arr,
        'rel_from_sat': rel_from_sat,
        'latest_sat_arrival': latest,
        'unserved_clients': [],
        'infeasible': False,
    })
    return sol


def _solve_set_partition_small(
    clients: List[int],
    entries: List[RoutePoolEntry],
    max_routes: int,
) -> Optional[List[RoutePoolEntry]]:
    """
    Solves a small set-partitioning problem exactly by selecting disjoint 
    route-pool entries that cover all clients.
    """
    n = len(clients)
    if n == 0 or n > 24:
        return None
    idx = {int(c): i for i, c in enumerate(clients)}
    route_items = []
    best_per_first: Dict[int, List[Tuple[float, int]]] = {}
    
    # Convert each route-pool entry into a bitmask representation over the client set
    for ridx, e in enumerate(entries):
        mask = 0
        valid = True
        for c in e.clients:
            if c not in idx:
                valid = False
                break
            mask |= 1 << idx[c]
        if not valid or mask == 0:
            continue
        route_items.append((ridx, mask, float(e.cost)))
    if not route_items:
        return None
    
    # Build candidate lists and lower-bound information for each client position
    for ridx, mask, cost in route_items:
        mm = mask
        while mm:
            b = (mm & -mm)
            j = b.bit_length() - 1
            best_per_first.setdefault(j, []).append((cost, ridx))
            mm ^= b
    min_cost_cover = [float('inf')] * n
    for j in range(n):
        vals = best_per_first.get(j, [])
        if not vals:
            return None
        min_cost_cover[j] = min(c for c, _ in vals)
        vals.sort()
        best_per_first[j] = vals[:16]
    full_mask = (1 << n) - 1
    best_cost = float('inf')
    best_sel: Optional[List[int]] = None
    
    # Compute a simple lower bound from the cheapest route covering each uncovered client
    def lower_bound(mask: int) -> float:
        lb = 0.0
        mm = (~mask) & full_mask
        while mm:
            b = (mm & -mm)
            j = b.bit_length() - 1
            lb = max(lb, min_cost_cover[j])
            mm ^= b
        return lb
    
    # Explore feasible disjoint covers with branch-and-bound
    def dfs(mask: int, cost: float, chosen: List[int]):
        nonlocal best_cost, best_sel
        if cost >= best_cost - EPS or len(chosen) > max_routes:
            return
        if mask == full_mask:
            best_cost = cost
            best_sel = list(chosen)
            return
        if cost + lower_bound(mask) >= best_cost - EPS:
            return
        first_uncovered = ((~mask) & full_mask & -((~mask) & full_mask)).bit_length() - 1
        cands = best_per_first.get(first_uncovered, [])
        for _, ridx in cands:
            rmask = next(m for rr, m, _ in route_items if rr == ridx)
            if rmask & mask:
                continue
            rcost = next(c for rr, _, c in route_items if rr == ridx)
            chosen.append(ridx)
            dfs(mask | rmask, cost + rcost, chosen)
            chosen.pop()

    dfs(0, 0.0, [])
    if best_sel is None:
        return None
    return [entries[ridx] for ridx in best_sel]


def _solve_set_partition_greedy(
    clients: List[int],
    entries: List[RoutePoolEntry],
    max_routes: int,
) -> Optional[List[RoutePoolEntry]]:
    """
    Builds a feasible client cover greedily by repeatedly selecting the best 
    cost-per-new-client route entry.
    """
    uncovered = set(int(c) for c in clients)
    selected: List[RoutePoolEntry] = []
    used: Set[int] = set()
    
    # Iteratively select the entry with the best ratio between route cost and newly covered clients
    while uncovered and len(selected) < max_routes:
        best = None
        best_score = float('inf')
        for e in entries:
            e_set = set(int(c) for c in e.clients)
            if e_set & used:
                continue
            gain = len(e_set & uncovered)
            if gain <= 0:
                continue
            score = e.cost / gain
            if score < best_score:
                best_score = score
                best = e
        if best is None:
            return None
        selected.append(best)
        used.update(int(c) for c in best.clients)
        uncovered -= set(int(c) for c in best.clients)
    if uncovered:
        return None
    return selected


def recombine_route_pool(
    inst: Instance2E,
    base_sol: Solution2E,
    elite_solutions: List[Solution2E],
    *,
    exact_threshold: int = 10,
    exact_small_threshold: int = 6,
    max_pool_routes: int = 160,
    max_routes: Optional[int] = None,
    rng: Optional[random.Random] = None,
    time_budget_s: Optional[float] = None,
) -> LSResult:
    """
    Builds a new solution by recombining route fragments from an elite route 
    pool and post-optimizing the result.
    """
    debug: List[str] = []
    deadline = None if time_budget_s is None else (time.perf_counter() + max(0.0, float(time_budget_s)))
    
    # Skip recombination when the base solution is incomplete or when the time budget is already exhausted
    if base_sol.meta.get('unserved_clients'):
        debug.append('sp_skip:unserved_clients')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)
    if not elite_solutions:
        elite_solutions = [base_sol]
    if _time_exceeded(deadline):
        debug.append('sp_stop:time_budget')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)
    
    # Build the pool of explicit route candidates extracted from the base and elite solutions
    entries = _collect_route_pool_entries(inst, [base_sol] + list(elite_solutions), max_entries=max_pool_routes)
    debug.append(f'sp_pool:solutions={1+len(elite_solutions)} routes={len(entries)}')
    clients = sorted(int(c) for c in base_sol.assignment.keys())
    route_cap = int(max_routes if max_routes is not None else max(1, int(inst.nv2)))
    
    # Try an exact set-partitioning cover first, then fall back to a greedy cover if needed
    selected = _solve_set_partition_small(clients, entries, route_cap)
    solver = 'exact'
    if selected is None:
        selected = _solve_set_partition_greedy(clients, entries, route_cap)
        solver = 'greedy'
    if selected is None:
        debug.append('sp_stop:no_cover')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)
    if _time_exceeded(deadline):
        debug.append('sp_stop:time_budget_before_rebuild')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)
    
    # Rebuild a full solution from the selected pool routes
    cand_sol = _build_solution_from_explicit_pool(inst, base_sol, selected)
    if cand_sol is None:
        debug.append('sp_stop:rebuild_failed')
        return LSResult(base_sol, False, 0, _dist2(inst, base_sol), debug)
    
    # Apply recharge optimization and a short second-level local search on the rebuilt solution
    recharge_res = optimize_solution_recharges(inst, cand_sol)
    if recharge_res.improved:
        cand_sol = recharge_res.solution
        debug.extend(recharge_res.debug)
    ls_budget = None
    if deadline is not None:
        ls_budget = max(0.0, _time_remaining(deadline) - 0.05)
    ls_res = LSResult(cand_sol, False, 0, _dist2(inst, cand_sol), [])
    if ls_budget is None or ls_budget > 0.05:
        ls_res = intensify_lvl2_solution(
            inst,
            cand_sol,
            max_passes=4,
            exact_threshold=exact_threshold,
            exact_small_threshold=exact_small_threshold,
            max_neighbors=100,
            rng=rng or random.Random(0),
            recharge_every=1,
            time_budget_s=ls_budget,
        )
    else:
        debug.append('sp_stop:time_budget_before_ls')
    debug.extend(ls_res.debug)
    if ls_res.improved:
        cand_sol = ls_res.solution
        
    # Accept the recombined solution only if it strictly improves the second-level distance
    base_cost = _dist2(inst, base_sol)
    cand_cost = _dist2(inst, cand_sol)
    debug.append(f'sp_solver:{solver}:selected_routes={len(selected)}')
    if cand_cost + EPS < base_cost:
        cand_sol.meta['sp_pool_applied'] = True
        cand_sol.meta['sp_pool_route_count'] = len(entries)
        cand_sol.meta['post_acs_debug'] = list(cand_sol.meta.get('post_acs_debug', [])) + debug
        return LSResult(cand_sol, True, 1 + int(ls_res.moves), cand_cost, debug)
    debug.append(f'sp_reject:dist2={base_cost:.3f}->{cand_cost:.3f}')
    return LSResult(base_sol, False, 0, base_cost, debug)


def intensify_absolute_with_fixed_lvl1(
    inst: Instance2E,
    sol: Solution2E,
    max_attempts: int = 2,
    *,
    time_budget_s: Optional[float] = None,
    per_satellite_budget_s: Optional[float] = None,
    max_clients_per_satellite: Optional[int] = None,
    max_routes_per_satellite: Optional[int] = None,
) -> LSResult:
    """
    Improves second-level routes in absolute time while keeping the first-level 
    schedule fixed.
    """
    debug: List[str] = []
    deadline = None if time_budget_s is None else (time.perf_counter() + max(0.0, float(time_budget_s)))
    A_s = simulate_truck_arrivals(inst, sol.routes_lvl1)
    routes_lvl2_new: Dict[int, List[Route]] = {int(s): list(rs) for s, rs in sol.routes_lvl2.items()}
    improved = False
    
    # Process satellites independently and try to rebuild their routes with alternative route 
    for sat, routes in list(sol.routes_lvl2.items()):
        sat = int(sat)
        if not routes:
            continue
        if _time_exceeded(deadline):
            debug.append(f"abs_ls_stop:time_budget:sat={sat}")
            break
        clients: List[int] = []
        vids: List[int] = []
        for r in routes:
            vids.append(int(r.vehicle_id))
            clients.extend([n for n in r.nodes[1:-1] if n in inst.client_ids])
        n_clients_sat = len(clients)
        if max_clients_per_satellite is not None and n_clients_sat > int(max_clients_per_satellite):
            debug.append(f"abs_ls_skip:too_many_clients:sat={sat}:n={n_clients_sat}")
            continue
        if max_routes_per_satellite is not None and len(routes) > int(max_routes_per_satellite):
            debug.append(f"abs_ls_skip:too_many_routes:sat={sat}:r={len(routes)}")
            continue
        routes = [_optimize_route_recharges(inst, sat, r) for r in routes]
        current_dist = sum(distance_of_nodes(inst, r.nodes) for r in routes)
        best_dist = current_dist
        best_routes = routes
        
        # Try several route-count configurations for this satellite
        k_candidates = [len(routes)]
        if len(routes) > 1:
            k_candidates = sorted(set([len(routes), max(1, len(routes)-1), max(1, len(routes)-2), 1]))
        sat_deadline = None
        if per_satellite_budget_s is not None:
            sat_deadline = time.perf_counter() + max(0.0, float(per_satellite_budget_s))
            if deadline is not None:
                sat_deadline = min(sat_deadline, deadline)
        elif deadline is not None:
            sat_deadline = deadline

        for K in k_candidates:
            if sat_deadline is not None and time.perf_counter() >= sat_deadline:
                debug.append(f"abs_ls_sat_stop:time_budget:sat={sat}")
                break
            rr, _, _, dbg, ok = _rebuild_satellite_routes_absolute_global(inst, sat, clients, A_s.get(sat, 0.0), vids[:K])
            if not ok:
                continue
            rr = [_optimize_route_recharges(inst, sat, r) for r in rr]
            d = sum(distance_of_nodes(inst, r.nodes) for r in rr)
            if d + EPS < best_dist:
                best_dist = d
                best_routes = rr
        if best_dist + EPS < current_dist:
            routes_lvl2_new[sat] = best_routes
            improved = True
            debug.append(f"abs_ls_accept:sat={sat}:dist2={current_dist:.3f}->{best_dist:.3f}:routes={len(routes)}->{len(best_routes)}")

    if not improved:
        debug.append("abs_ls_stop:no_improvement")
        return LSResult(sol, False, 0, _dist2(inst, sol), debug)
    
    # Replay the improved second level against the fixed first level and restore consistency
    cand = Solution2E(assignment=dict(sol.assignment), routes_lvl2=routes_lvl2_new, routes_lvl1=list(sol.routes_lvl1), meta=dict(sol.meta))
    rebuilt, tw_pen, _, _, dbg2, ok2 = repair_and_replay_lvl2_absolute(inst, cand, A_s, max_attempts=max_attempts)
    rebuilt = _refresh_solution_consistency(inst, rebuilt)
    rebuilt_cost = _dist2(inst, rebuilt)
    
    # Keep the explicit rebuilt routes if replay increases the second-level distance too much
    if rebuilt_cost > sum(distance_of_nodes(inst, r.nodes) for rs in routes_lvl2_new.values() for r in rs) + EPS:
        rebuilt = cand
        rebuilt_cost = _dist2(inst, cand)
    
    # Finish with one recharge optimization pass on the rebuilt candidate
    recharge_post = optimize_solution_recharges(inst, rebuilt)
    if recharge_post.improved and recharge_post.best_dist2 + EPS < rebuilt_cost:
        rebuilt = recharge_post.solution
        rebuilt_cost = recharge_post.best_dist2
        debug.extend(list(recharge_post.debug))
    rebuilt.meta["post_acs_debug"] = list(rebuilt.meta.get("post_acs_debug", [])) + debug + list(dbg2)
    rebuilt.meta["abs_ls_applied"] = bool(improved)
    rebuilt.meta["abs_ls_moves"] = 1 if improved else 0
    return LSResult(rebuilt, bool(improved), 1 if improved else 0, rebuilt_cost, debug + list(dbg2))
