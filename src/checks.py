# checks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Set

from .instance import Instance2E
from .solution import Solution2E, Route


@dataclass
class CheckResult:
    """Stores the outcome of a solution check, including hard feasibility status and soft violation magnitudes."""
    hard_ok: bool # does the solution violate a blocking constraint ?
    hard_violations: List[str] # readable list of hard errors
    
    # Soft components
    cap_violation_lvl2: float = 0.0
    cap_violation_lvl1: float = 0.0
    bat_violation_lvl2: float = 0.0


def ok_result() -> CheckResult:
    """Returns a default check result corresponding to a fully feasible solution."""
    return CheckResult(True, [], 0.0, 0.0, 0.0)


def _get_routes2(sol: Solution2E) -> Dict[int, List[Route]]:
    """
    Return second-echelon routes grouped by satellite:
        {sat_id: [Route, Route, ...]}
    """
    return sol.routes_lvl2


def _get_routes1(sol: Solution2E) -> List[Route]:
    """
    Return first-echelon truck routes:
        [Route, Route, ...]
    """
    return sol.routes_lvl1


def _route_has_unknown_ids(inst: Instance2E, route: Route) -> Optional[str]:
    """Checks whether a route contains node identifiers that do not exist in the instance."""
    nb = inst.node_by_id
    for nid in route.nodes:
        if nid not in nb:
            return f"Unknown node id in route: {nid}"
    return None


def check_routes2_hard(inst: Instance2E, sol: Solution2E) -> CheckResult:
    """
    HARD checks for level-2 routes:
      - routes_lvl2 keys are valid satellites
      - each route starts/ends at that satellite
      - no depot inside EV route
      - no other satellite inside EV route
      - all node ids exist in instance
    """
    res = ok_result()
    routes2 = _get_routes2(sol)

    # 1) keys must be satellites
    for sat_id, routes in routes2.items():
        if sat_id not in inst.satellite_ids:
            res.hard_ok = False
            res.hard_violations.append(f"routes_lvl2 contains non-satellite key: {sat_id}")
            continue

        # 2) each route must be a Route and start/end at sat_id
        for r_idx, route in enumerate(routes):
            if not isinstance(route, Route):
                res.hard_ok = False
                res.hard_violations.append(f"routes_lvl2[{sat_id}][{r_idx}] is not a Route object.")
                continue

            msg = _route_has_unknown_ids(inst, route)
            if msg:
                res.hard_ok = False
                res.hard_violations.append(f"routes_lvl2[{sat_id}][{r_idx}]: {msg}")
                continue
            
            # C23: no self-loops in level-2 routes (no l->l)
            for a, b in route.arcs():
                if a == b:
                    res.hard_ok = False
                    res.hard_violations.append(
                        f"routes_lvl2[{sat_id}][{r_idx}] contains self-loop arc ({a}->{b}) (C23 violation)."
                    )
            
            if route.start != sat_id or route.end != sat_id:
                res.hard_ok = False
                res.hard_violations.append(
                    f"routes_lvl2[{sat_id}][{r_idx}] must start/end at satellite {sat_id}. "
                    f"Got {route.start}..{route.end}"
                )

            # 3) forbid depot inside
            for nid in route.inner_nodes():
                if nid == inst.depot_id:
                    res.hard_ok = False
                    res.hard_violations.append(
                        f"routes_lvl2[{sat_id}][{r_idx}] visits depot inside route (forbidden)."
                    )

            # 4) forbid other satellites inside
            for nid in route.inner_nodes():
                if nid in inst.satellite_ids and nid != sat_id:
                    res.hard_ok = False
                    res.hard_violations.append(
                        f"routes_lvl2[{sat_id}][{r_idx}] visits another satellite {nid} (forbidden)."
                    )

    return res


def check_clients_served_and_assignment_hard(inst: Instance2E, sol: Solution2E) -> "CheckResult":
    """
    HARD checks:
      1) Every client is served exactly once in level-2 routes.
      2) If a client is visited in routes_lvl2[sat_id], then sol.assignment[client] must equal sat_id.
         Also, every client must appear in sol.assignment.
    """
    res = ok_result()

    routes2 = _get_routes2(sol)

    # Count visits per client
    visit_count: Dict[int, int] = {cid: 0 for cid in inst.client_ids}

    # Track which satellite served each client (for assignment consistency)
    served_by: Dict[int, int] = {}

    for sat_id, routes in routes2.items():
        for r_idx, route in enumerate(routes):
            for nid in route.inner_nodes():
                if nid in inst.client_ids:
                    visit_count[nid] += 1
                    # if served multiple times by different sats, we keep last, but it'll be flagged by count anyway
                    served_by[nid] = sat_id

    missing = [cid for cid, k in visit_count.items() if k == 0]
    multi = [cid for cid, k in visit_count.items() if k > 1]

    if missing:
        res.hard_ok = False
        res.hard_violations.append(f"Clients not served in level-2 routes: {missing}")

    if multi:
        res.hard_ok = False
        res.hard_violations.append(f"Clients served more than once in level-2 routes: {multi}")

    # Assignment checks (hard)
    # 1) all clients must be assigned
    unassigned = [cid for cid in inst.client_ids if cid not in sol.assignment]
    if unassigned:
        res.hard_ok = False
        res.hard_violations.append(f"Clients missing from assignment dict: {unassigned}")

    # 2) served satellite must match assignment
    for cid, sat_serving in served_by.items():
        sat_assigned = sol.assignment.get(cid, None)
        if sat_assigned is None:
            continue  # already reported as unassigned
        if sat_assigned != sat_serving:
            res.hard_ok = False
            res.hard_violations.append(
                f"Assignment mismatch for client {cid}: served by sat {sat_serving}, "
                f"but assignment says {sat_assigned}"
            )

    return res


def capacity_violation_lvl2(inst: Instance2E, sol: Solution2E) -> float:
    """
    SOFT: Sum of capacity overload (Q2) across all second-echelon EV routes.

    For each EV route r:
        load(r) = sum(demand[client] for client in r.inner_nodes() if client is a client)
        overload(r) = max(0, load(r) - Q2)

    Returns:
        total_overload >= 0
    """
    Q2 = float(inst.Q2)
    total_over = 0.0

    routes2 = _get_routes2(sol)
    for sat_id, routes in routes2.items():
        for route in routes:
            load = 0.0
            for nid in route.inner_nodes():
                if nid in inst.client_ids:
                    load += inst.demand_by_idx[inst.idx(nid)]
            total_over += max(0.0, load - Q2)

    return float(total_over)


def check_routes1_and_interechelon_hard(inst: Instance2E, sol: Solution2E) -> "CheckResult":
    """
    HARD checks for first echelon + inter-echelon coherence.

    1) Each lvl1 route:
       - starts/ends at depot
       - inner nodes are satellites only
       - no degenerate [depot, depot] route (and no consecutive depots)
    2) Any satellite used in lvl2 must be visited by at least one lvl1 route (supplied).
    """
    res = ok_result()

    routes1 = _get_routes1(sol)
    routes2 = _get_routes2(sol)

    # (A) lvl1 route structure
    visited_sats_lvl1: Set[int] = set() 

    for r_idx, route in enumerate(routes1):
        if not isinstance(route, Route):
            res.hard_ok = False
            res.hard_violations.append(f"routes_lvl1[{r_idx}] is not a Route object.")
            continue

        # start/end must be depot
        if route.start != inst.depot_id or route.end != inst.depot_id:
            res.hard_ok = False
            res.hard_violations.append(
                f"routes_lvl1[{r_idx}] must start/end at depot {inst.depot_id}. Got {route.start}..{route.end}"
            )

        # forbid degenerate route [depot, depot]
        if len(route.nodes) == 2 and route.nodes[0] == inst.depot_id and route.nodes[1] == inst.depot_id:
            res.hard_ok = False
            res.hard_violations.append(
                f"routes_lvl1[{r_idx}] is degenerate [depot, depot] (forbidden by C4 spirit)."
            )
            continue

        # inner nodes must be satellites only
        for nid in route.inner_nodes():
            if nid not in inst.satellite_ids:
                res.hard_ok = False
                res.hard_violations.append(
                    f"routes_lvl1[{r_idx}] contains non-satellite internal node {nid} (forbidden in lvl1)."
                )
            else:
                visited_sats_lvl1.add(nid)

        # forbid consecutive depots inside the route (depot->depot move)
        for a, b in route.arcs():
            if a == inst.depot_id and b == inst.depot_id:
                res.hard_ok = False
                res.hard_violations.append(
                    f"routes_lvl1[{r_idx}] contains depot->depot arc (forbidden by C4)."
                )
        
        # C22: no self-loops in level-1 routes (no i->i)
        for a, b in route.arcs():
            if a == b:
                res.hard_ok = False
                res.hard_violations.append(
                    f"routes_lvl1[{r_idx}] contains self-loop arc ({a}->{b}) (C22 violation)."
                )

    # (B) inter-echelon: used sats must be fed
    used_sats_lvl2: Set[int] = set()
    for sat_id, rts in routes2.items():
        if rts:  # satellite réellement utilisé au lvl2
            used_sats_lvl2.add(sat_id)

    not_supplied = sorted(list(used_sats_lvl2 - visited_sats_lvl1))
    if not_supplied:
        res.hard_ok = False
        res.hard_violations.append(
            f"Satellites used in lvl2 but never visited in lvl1 (not supplied): {not_supplied}"
        )

    return res


def _satellite_demands_from_assignment(inst: Instance2E, sol: Solution2E) -> Dict[int, float]:
    """
    sat_demand[sat] = sum(demand[client] for client assigned to sat)
    """
    sat_demand: Dict[int, float] = {s: 0.0 for s in inst.satellite_ids}

    for cid, sid in sol.assignment.items():
        if cid in inst.client_ids and sid in inst.satellite_ids:
            sat_demand[sid] += inst.demand_by_idx[inst.idx(cid)]

    return sat_demand


def capacity_violation_lvl1(inst: Instance2E, sol: Solution2E) -> float:
    """
    SOFT: Sum of capacity overload (Q1) across all first-echelon truck routes.
    Load per route = sum of satellite demands for satellites visited by the truck.
    """
    Q1 = float(inst.Q1)
    sat_demand = _satellite_demands_from_assignment(inst, sol)

    total_over = 0.0
    for route in sol.routes_lvl1:
        load = 0.0
        for nid in route.inner_nodes():
            if nid in inst.satellite_ids:
                load += sat_demand.get(nid, 0.0)
        total_over += max(0.0, load - Q1)

    return float(total_over)
