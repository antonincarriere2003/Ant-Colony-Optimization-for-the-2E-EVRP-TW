from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from src.instance import Instance2E
from src.solution import Route


@dataclass
class CWRoute:
    """Represents an intermediate Clarke-Wright first-level route together with its aggregated load."""
    nodes: List[int]
    load: float


def _dist(inst: Instance2E, a: int, b: int) -> float:
    """Returns the distance between two nodes from the instance distance matrix."""
    return float(inst.require_dist()[inst.idx(a)][inst.idx(b)])


def build_satellite_demands(inst: Instance2E, assignment: Dict[int, int]) -> Dict[int, float]:
    """Aggregates client demands by satellite according to the current client-to-satellite assignment."""
    D: Dict[int, float] = {s: 0.0 for s in inst.satellite_ids}
    
    # Sum the demand of every client into its assigned satellite
    for client_id, sat_id in assignment.items():
        D[sat_id] += float(inst.node_by_id[client_id].demand)
    return D


def distance_of_route(inst: Instance2E, nodes: List[int]) -> float:
    """Computes the total travel distance along a route defined by an ordered list of nodes."""
    return sum(_dist(inst, nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))


def simulate_truck_arrivals(inst: Instance2E, routes_lvl1: List[Route]) -> Dict[int, float]:
    """Computes the earliest arrival time of first-level trucks at each satellite."""
    depot = inst.depot_id
    A: Dict[int, float] = {}
    
    # Traverse each truck route and record the first arrival time at every visited satellite
    for r in routes_lvl1:
        t = 0.0
        u = depot
        for v in r.nodes[1:]:
            t += _dist(inst, u, v)
            if v in inst.satellite_ids:
                A[v] = min(A.get(v, float("inf")), t)
            u = v
    return A


def _route_load(route: CWRoute, D: Dict[int, float]) -> float:
    """Computes the total first-level load delivered along a Clarke-Wright route."""
    return float(sum(D.get(s, 0.0) for s in route.nodes[1:-1]))


def _route_arrivals(inst: Instance2E, nodes: List[int]) -> Dict[int, float]:
    """Computes the arrival time at each satellite along a first-level route."""
    depot = inst.depot_id
    t = 0.0
    u = depot
    out: Dict[int, float] = {}
    
    # Simulate the route from the depot and record the first arrival at each satellite
    for v in nodes[1:]:
        t += _dist(inst, u, v)
        if v in inst.satellite_ids and v not in out:
            out[v] = t
        u = v
    return out


def _route_respects_latest(inst: Instance2E, nodes: List[int], latest_sat_arrival: Optional[Dict[int, float]]) -> bool:
    """
    Checks whether a first-level route reaches every visited satellite before 
    its latest allowed arrival time.
    """
    if not latest_sat_arrival:
        return True
    arr = _route_arrivals(inst, nodes)
    for s, t in arr.items():
        lim = float(latest_sat_arrival.get(s, float("inf")))
        if t > lim + 1e-9:
            return False
    return True


def split_oversized_satellites(inst: Instance2E, D: Dict[int, float], latest_sat_arrival: Optional[Dict[int, float]] = None):
    """
    Splits oversized satellite demands into direct depot-satellite-depot routes 
    and returns the remaining unsplit demands.
    """
    depot = inst.depot_id
    Q1 = float(inst.Q1)
    direct_routes: List[Route] = []
    residual_D: Dict[int, float] = {}
    vid = 2000
    debug: List[str] = []
    
    # Create direct truck routes for satellites whose demand exceeds first-level vehicle capacity
    for s in inst.satellite_ids:
        demand = float(D.get(s, 0.0))
        if demand <= 1e-9:
            continue
        if demand > Q1 + 1e-9:
            remaining = demand
            
            # Add as many full-capacity direct routes as needed
            while remaining > Q1 + 1e-9:
                nodes = [depot, s, depot]
                if not _route_respects_latest(inst, nodes, latest_sat_arrival):
                    debug.append(f"direct_route_late:{s}")
                direct_routes.append(Route(nodes=nodes, vehicle_id=vid))
                vid += 1
                remaining -= Q1
                
            # Add the final residual direct route if some demand remains
            if remaining > 1e-9:
                nodes = [depot, s, depot]
                if not _route_respects_latest(inst, nodes, latest_sat_arrival):
                    debug.append(f"direct_route_late:{s}")
                direct_routes.append(Route(nodes=nodes, vehicle_id=vid))
                vid += 1
        else:
            residual_D[s] = demand
    return direct_routes, residual_D, debug


def _cw_find_route_containing(routes: List[CWRoute], sat: int) -> int:
    """Returns the index of the Clarke-Wright route currently containing a given satellite."""
    for i, r in enumerate(routes):
        if sat in r.nodes[1:-1]:
            return i
    return -1


def _cw_build_savings(inst: Instance2E, sats: List[int], depot: int) -> List[Tuple[float, int, int]]:
    """Builds and sorts the Clarke-Wright savings list for the remaining satellites."""
    savings: List[Tuple[float, int, int]] = []

    for i in range(len(sats)):
        for j in range(i + 1, len(sats)):
            a = sats[i]
            b = sats[j]
            sav = _dist(inst, depot, a) + _dist(inst, depot, b) - _dist(inst, a, b)
            savings.append((sav, a, b))

    savings.sort(reverse=True)
    return savings


def _cw_is_left_end(route: CWRoute, sat: int) -> bool:
    """Checks whether a satellite is the left endpoint of a Clarke-Wright route."""
    return route.nodes[1] == sat


def _cw_is_right_end(route: CWRoute, sat: int) -> bool:
    """Checks whether a satellite is the right endpoint of a Clarke-Wright route."""
    return route.nodes[-2] == sat


def _cw_reverse_internal_path(depot: int, nodes: List[int]) -> List[int]:
    """Reverses the internal satellite sequence of a depot-to-depot route."""
    return [depot] + list(reversed(nodes[1:-1])) + [depot]


def _cw_candidate_merges(depot: int, ra: CWRoute, rb: CWRoute, a: int, b: int) -> List[List[int]]:
    """Generates all valid endpoint-based merge candidates for two Clarke-Wright routes."""
    candidates: List[List[int]] = []

    if _cw_is_right_end(ra, a) and _cw_is_left_end(rb, b):
        candidates.append(ra.nodes[:-1] + rb.nodes[1:])

    if _cw_is_left_end(ra, a) and _cw_is_right_end(rb, b):
        ra_rev = _cw_reverse_internal_path(depot, ra.nodes)
        rb_rev = _cw_reverse_internal_path(depot, rb.nodes)
        candidates.append(ra_rev[:-1] + rb_rev[1:])

    if _cw_is_right_end(ra, a) and _cw_is_right_end(rb, b):
        rb_rev = _cw_reverse_internal_path(depot, rb.nodes)
        candidates.append(ra.nodes[:-1] + rb_rev[1:])

    if _cw_is_left_end(ra, a) and _cw_is_left_end(rb, b):
        ra_rev = _cw_reverse_internal_path(depot, ra.nodes)
        candidates.append(ra_rev[:-1] + rb.nodes[1:])

    return candidates


def _cw_select_feasible_merge(
    inst: Instance2E,
    candidates: List[List[int]],
    latest_sat_arrival: Optional[Dict[int, float]],
) -> Optional[List[int]]:
    """Selects the first merged route candidate that satisfies the latest-arrival constraints."""
    for nodes in candidates:
        if _route_respects_latest(inst, nodes, latest_sat_arrival):
            return nodes
    return None


def clarke_wright_lvl1(inst: Instance2E, D: Dict[int, float], latest_sat_arrival: Optional[Dict[int, float]] = None) -> Tuple[List[Route], List[str]]:
    """
    Builds first-echelon routes with a Clarke-Wright savings heuristic while 
    respecting truck capacity and optional latest satellite arrivals.
    """
    depot = inst.depot_id

    # First isolate oversized satellite demands into fixed direct routes
    fixed_routes, D_small, debug = split_oversized_satellites(inst, D, latest_sat_arrival)
    sats = [s for s in sorted(inst.satellite_ids) if D_small.get(s, 0.0) > 1e-9]
    if not sats:
        return fixed_routes, debug

    # Initialize one depot-satellite-depot route per remaining satellite
    routes: List[CWRoute] = [CWRoute(nodes=[depot, s, depot], load=float(D_small[s])) for s in sats]
    savings = _cw_build_savings(inst, sats, depot)

    # Process savings in descending order and merge compatible route endpoints when feasible
    for _, a, b in savings:
        ia = _cw_find_route_containing(routes, a)
        ib = _cw_find_route_containing(routes, b)
        if ia == -1 or ib == -1 or ia == ib:
            continue

        ra = routes[ia]
        rb = routes[ib]
        if ra.load + rb.load > float(inst.Q1) + 1e-9:
            continue

        candidates = _cw_candidate_merges(depot, ra, rb, a, b)
        selected = _cw_select_feasible_merge(inst, candidates, latest_sat_arrival)
        if selected is None:
            continue

        routes[ia] = CWRoute(nodes=selected, load=ra.load + rb.load)
        routes.pop(ib)

    # Convert the internal Clarke-Wright routes into standard Route objects and append fixed routes
    out = fixed_routes + [Route(nodes=r.nodes, vehicle_id=1000 + k) for k, r in enumerate(routes)]
    return out, debug


def repair_merge_lvl1_to_fleet(inst: Instance2E, routes_lvl1: List[Route], D: Dict[int, float], nv1: int, latest_sat_arrival: Optional[Dict[int, float]] = None) -> Tuple[List[Route], List[str], bool]:
    """
    Repairs a first-level solution by merging truck routes until the fleet 
    limit is satisfied, preferring merges that respect latest satellite arrivals.
    """
    debug: List[str] = []
    routes = [CWRoute(nodes=list(r.nodes), load=_route_load(CWRoute(list(r.nodes), 0.0), D)) for r in routes_lvl1]
    depot = inst.depot_id
    
    # Repeatedly merge the best pair of routes until the fleet-size limit is met
    while len(routes) > int(nv1):
        best_pair = None
        best_relaxed_pair = None
        best_delta = float("inf")
        best_relaxed_delta = float("inf")
        
        # Evaluate all feasible route-pair merges under capacity and timing considerations
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                ri, rj = routes[i], routes[j]
                if ri.load + rj.load > float(inst.Q1) + 1e-9:
                    continue
                inner_i = ri.nodes[1:-1]
                inner_j = rj.nodes[1:-1]
                candidates = [
                    [depot] + inner_i + inner_j + [depot],
                    [depot] + inner_i + list(reversed(inner_j)) + [depot],
                    [depot] + list(reversed(inner_i)) + inner_j + [depot],
                    [depot] + list(reversed(inner_i)) + list(reversed(inner_j)) + [depot],
                ]
                for nodes in candidates:
                    delta = distance_of_route(inst, nodes) - distance_of_route(inst, ri.nodes) - distance_of_route(inst, rj.nodes)
                    if _route_respects_latest(inst, nodes, latest_sat_arrival):
                        if delta < best_delta:
                            best_delta = delta
                            best_pair = (i, j, nodes, ri.load + rj.load)
                    elif delta < best_relaxed_delta:
                        best_relaxed_delta = delta
                        best_relaxed_pair = (i, j, nodes, ri.load + rj.load)
                        
        # Prefer a merge satisfying the latest-arrival limits, otherwise fall back to the best relaxed merge
        chosen = best_pair if best_pair is not None else best_relaxed_pair
        if chosen is None:
            debug.append(f"lvl1_repair_failed_remaining_routes:{len(routes)}")
            return [Route(nodes=r.nodes, vehicle_id=3000 + k) for k, r in enumerate(routes)], debug, False
        i, j, nodes, load = chosen
        routes[i] = CWRoute(nodes=nodes, load=load)
        routes.pop(j)
        if chosen is best_pair:
            debug.append(f"lvl1_merge_repair:{nodes}")
        else:
            debug.append(f"lvl1_merge_repair_relaxed_latest:{nodes}")
    return [Route(nodes=r.nodes, vehicle_id=3000 + k) for k, r in enumerate(routes)], debug, True
