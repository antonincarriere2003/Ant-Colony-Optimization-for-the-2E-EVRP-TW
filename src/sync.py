from __future__ import annotations
from typing import Dict, Tuple, List, Set, Optional
import heapq

from src.instance import Instance2E
from src.solution import Solution2E, Route


def _safe_route(nodes: List[int], vehicle_id: int, fallback_start: int) -> Route:
    """Builds a safe route object by enforcing a non-empty closed sequence of nodes."""
    seq = list(nodes) if nodes is not None else []
    
    # Normalize empty, singleton, or open sequences into a valid closed route
    if not seq:
        seq = [fallback_start, fallback_start]
    elif len(seq) == 1:
        seq = [seq[0], seq[0]]
    elif seq[0] != seq[-1]:
        seq.append(seq[0])
    return Route(nodes=seq, vehicle_id=vehicle_id)


def _refresh_solution_consistency(inst: Instance2E, sol: Solution2E) -> Solution2E:
    """Rebuilds the assignment and unserved-client metadata from the current second-level routes."""
    assignment: Dict[int, int] = {}
    served = set()
    
    # Recover the served clients and their serving satellite directly from the routes
    for sat, routes in sol.routes_lvl2.items():
        for r in routes:
            start = r.nodes[0] if r.nodes else sat
            for n in r.nodes[1:-1]:
                if n in inst.client_ids:
                    assignment[int(n)] = int(start)
                    served.add(int(n))
    
    # Update the solution metadata to match the actual route content
    unserved = sorted(int(c) for c in inst.client_ids if int(c) not in served)
    sol.assignment = assignment
    sol.meta['unserved_clients'] = unserved
    sol.meta['assignment_consistent_with_routes'] = True
    return sol


def _dist(inst: Instance2E, a: int, b: int) -> float:
    """Returns the distance between two nodes from the instance distance matrix."""
    return float(inst.require_dist()[inst.idx(a)][inst.idx(b)])


def _reachable_after(inst, battery_after: float, node: int, sat: int) -> bool:
    """
    Checks whether the remaining battery still allows reaching either the satellite
    or at least one charging station.
    """
    h = float(inst.h)

    # Try a direct return to the satellite first
    d = _dist(inst, node, sat)
    if battery_after - h * d >= -1e-9:
        return True

    # Otherwise check whether at least one charging station remains reachable
    for r in inst.station_ids:
        d = _dist(inst, node, r)
        if battery_after - h * d >= -1e-9:
            return True

    return False


def _feasible_customer(inst, sat, cur, battery, load, t, c):
    """
    Checks whether a customer can be feasibly visited from the current state 
    under battery, capacity, and time-window constraints.
    """
    h = float(inst.h)
    d = _dist(inst, cur, c)

    # Check battery feasibility for the move to the customer
    if battery - h * d < -1e-9:
        return False

    arr = t + d
    start = max(arr, float(inst.tw_early(c)))

    # Check time-window feasibility at the customer
    if start > float(inst.tw_late(c)) + 1e-9:
        return False

    # Check vehicle capacity after serving the customer
    if load + float(inst.demand(c)) > float(inst.Q2) + 1e-9:
        return False

    # Ensure the vehicle is not stranded after reaching the customer
    battery_after = battery - h * d
    if not _reachable_after(inst, battery_after, c, sat):
        return False

    return True


def _validate_explicit_absolute_route(inst: Instance2E, sat: int, nodes: List[int], start_time: float, vehicle_id: int) -> Tuple[Route, float, Dict[int, float], List[str], bool]:
    """
    Validates an explicit absolute-time second-level route by simulating travel,
    charging, service, and feasibility constraints.
    """
    route = _safe_route(list(nodes), vehicle_id, sat)
    if not route.nodes or route.nodes[0] != sat or route.nodes[-1] != sat:
        return route, 0.0, {}, ['absolute_explicit_bad_endpoints'], False
    t = float(start_time)
    battery = float(inst.BCe)
    h = float(inst.h)
    load = 0.0
    cur = sat
    arrivals: Dict[int, float] = {}
    debug: List[str] = []
    ok = True
    seen_clients: Set[int] = set()
    
    # Simulate the route node by node and check all operational constraints
    for nxt in route.nodes[1:]:
        d = _dist(inst, cur, nxt)
        battery -= h * d
        if battery < -1e-9:
            debug.append(f'absolute_explicit_battery_fail:{cur}->{nxt}')
            ok = False
            break
        t += d
        if nxt in inst.station_ids:
            t += float(inst.ge) * max(0.0, float(inst.BCe) - battery)
            battery = float(inst.BCe)
        elif nxt in inst.client_ids:
            if nxt in seen_clients:
                debug.append(f'absolute_explicit_duplicate_client:{nxt}')
                ok = False
                break
            start = max(t, float(inst.tw_early(nxt)))
            if start > float(inst.tw_late(nxt)) + 1e-9:
                debug.append(f'absolute_explicit_tw_fail:{nxt}')
                ok = False
                break
            load += float(inst.demand(nxt))
            if load > float(inst.Q2) + 1e-9:
                debug.append(f'absolute_explicit_capacity_fail:{nxt}')
                ok = False
                break
            arrivals[int(nxt)] = float(start)
            seen_clients.add(int(nxt))
            t = start + float(inst.service_time(nxt))
            if nxt != sat and not _reachable_after(inst, battery, nxt, sat):
                debug.append(f'absolute_explicit_stranded_after:{nxt}')
                ok = False
                break
        elif nxt != sat:
            debug.append(f'absolute_explicit_unknown_node:{nxt}')
            ok = False
            break
        cur = nxt
    return route, 0.0, arrivals, debug, ok


def _absolute_state_key(cur_node: int, remaining: Set[int], battery: float, cur_t: float) -> Tuple[int, Tuple[int, ...], float, float]:
    """
    Builds a compact state signature used to detect repeated states during 
    absolute route reconstruction.
    """
    return (
        int(cur_node),
        tuple(sorted(int(c) for c in remaining)),
        round(float(battery), 3),
        round(float(cur_t), 3),
    )


def _absolute_client_score(inst: Instance2E, cur: int, t: float, c: int) -> Tuple[float, float, int]:
    """Scores a feasible client by urgency first, then by travel distance, then by client id."""
    d = _dist(inst, cur, c)
    arr = t + d
    start = max(arr, float(inst.tw_early(c)))
    slack = float(inst.tw_late(c)) - start
    return (slack, d, int(c))


def _absolute_feasible_clients(
    inst: Instance2E,
    sat: int,
    cur: int,
    battery: float,
    t: float,
    remaining: Set[int],
) -> List[int]:
    """Returns the clients that can be served immediately while preserving future reachability."""
    h = float(inst.h)
    feasible: List[int] = []

    for c in remaining:
        dist_cc = _dist(inst, cur, c)
        after = battery - h * dist_cc
        if after < -1e-9:
            continue

        arr = t + dist_cc
        start = max(arr, float(inst.tw_early(c)))
        if start > float(inst.tw_late(c)) + 1e-9:
            continue

        if _reachable_after(inst, after, c, sat):
            feasible.append(int(c))

    return feasible


def _absolute_append_client(
    inst: Instance2E,
    route: List[int],
    arrivals: Dict[int, float],
    cur: int,
    battery: float,
    t: float,
    c: int,
) -> Tuple[int, float, float]:
    """Appends a client to the route and updates the current position, battery, and time."""
    battery -= float(inst.h) * _dist(inst, cur, c)
    t += _dist(inst, cur, c)
    t = max(t, float(inst.tw_early(c)))
    arrivals[int(c)] = float(t)
    t += float(inst.service_time(c))
    route.append(int(c))
    return int(c), float(battery), float(t)


def _absolute_best_helpful_station(
    inst: Instance2E,
    sat: int,
    cur: int,
    battery: float,
    t: float,
    remaining: Set[int],
) -> Optional[int]:
    """Selects the most promising charging station when no client is directly feasible."""
    best_station = None
    best_station_score = None

    for r in inst.station_ids:
        d_cr = _dist(inst, cur, r)

        # Skip zero-distance stations that do not improve the battery state
        if d_cr <= 1e-12 and abs(battery - float(inst.BCe)) <= 1e-9:
            continue
        if battery - float(inst.h) * d_cr < -1e-9:
            continue

        batt_after_r = battery - float(inst.h) * d_cr
        t_r = t + d_cr + float(inst.ge) * max(0.0, float(inst.BCe) - batt_after_r)
        batt_full = float(inst.BCe)

        helped_clients = []
        for c in remaining:
            if _feasible_customer(inst, sat, r, batt_full, 0.0, t_r, c):
                arr_c = t_r + _dist(inst, r, c)
                start_c = max(arr_c, float(inst.tw_early(c)))
                slack_c = float(inst.tw_late(c)) - start_c
                helped_clients.append((slack_c, _dist(inst, r, c), int(c)))

        if not helped_clients:
            continue

        helped_clients.sort()
        score = (helped_clients[0][0], d_cr, len(helped_clients))
        if best_station_score is None or score < best_station_score:
            best_station_score = score
            best_station = int(r)

    return best_station


def _absolute_append_station(
    inst: Instance2E,
    route: List[int],
    cur: int,
    battery: float,
    t: float,
    station: int,
) -> Tuple[int, float, float]:
    """Appends a charging station to the route and updates the current position, battery, and time."""
    d = _dist(inst, cur, station)
    battery -= float(inst.h) * d
    t += d
    t += float(inst.ge) * max(0.0, float(inst.BCe) - battery)
    battery = float(inst.BCe)
    route.append(int(station))
    return int(station), float(battery), float(t)


def _absolute_best_return_station(
    inst: Instance2E,
    sat: int,
    cur: int,
    battery: float,
) -> Optional[int]:
    """Finds the best intermediate charging station that enables the return to the satellite."""
    best_back_station = None
    best_extra = float("inf")

    for r in inst.station_ids:
        d1 = _dist(inst, cur, r)
        d2 = _dist(inst, r, sat)

        if battery - float(inst.h) * d1 < -1e-9:
            continue
        if float(inst.BCe) - float(inst.h) * d2 < -1e-9:
            continue

        # Skip zero-distance stations that do not change anything
        if d1 <= 1e-12 and abs(battery - float(inst.BCe)) <= 1e-9:
            continue

        extra = d1 + d2
        if extra < best_extra:
            best_extra = extra
            best_back_station = int(r)

    return best_back_station


def _absolute_finalize_return(
    inst: Instance2E,
    sat: int,
    route: List[int],
    cur: int,
    battery: float,
    t: float,
    debug: List[str],
) -> Tuple[List[int], float, float, bool]:
    """Finalizes the return to the satellite, adding a charging stop if necessary."""
    ok = True

    if cur != sat:
        if battery - float(inst.h) * _dist(inst, cur, sat) < -1e-9:
            best_back_station = _absolute_best_return_station(inst, sat, cur, battery)
            if best_back_station is None:
                debug.append(f"absolute_rebuild_cannot_return_sat_from:{cur}")
                ok = False
            else:
                cur, battery, t = _absolute_append_station(inst, route, cur, battery, t, best_back_station)

        if battery - float(inst.h) * _dist(inst, cur, sat) >= -1e-9:
            battery -= float(inst.h) * _dist(inst, cur, sat)
            t += _dist(inst, cur, sat)
            route.append(int(sat))
        else:
            debug.append(f"absolute_rebuild_final_return_fail:{cur}")
            ok = False
            if route[-1] != sat:
                route.append(int(sat))

    return route, float(battery), float(t), bool(ok)


def _absolute_residual_lateness(
    inst: Instance2E,
    arrivals: Dict[int, float],
    debug: List[str],
) -> Tuple[float, bool]:
    """Computes the residual lateness penalty and flags late arrivals in the debug log."""
    residual = 0.0
    ok = True

    for c, arr in arrivals.items():
        late = max(0.0, arr - float(inst.tw_late(c)))
        residual += late
        if late > 1e-9:
            ok = False
            debug.append(f"absolute_rebuild_late:{c}:{late:.3f}")

    return float(residual), bool(ok)


def _rebuild_route_absolute(inst: Instance2E, sat: int, clients: List[int], start_time: float, vehicle_id: int) -> Tuple[Route, float, Dict[int, float], List[str], bool]:
    """
    Rebuilds a robust absolute-time route by serving feasible clients, inserting
    only useful charging stops, and preventing cycling states.
    """
    remaining = set(clients)
    route = [int(sat)]
    arrivals: Dict[int, float] = {}
    debug: List[str] = []

    t = float(start_time)
    cur = int(sat)
    battery = float(inst.BCe)

    seen_states = set()
    consecutive_station_moves = 0
    max_station_moves = max(6, 3 * (len(clients) + len(inst.station_ids) + 1))
    ok = True

    # Repeatedly serve a feasible client or move to a helpful charging station
    while remaining:
        key = _absolute_state_key(cur, remaining, battery, t)
        if key in seen_states:
            debug.append(f"absolute_rebuild_cycle_at:{cur}")
            ok = False
            break
        seen_states.add(key)

        feasible = _absolute_feasible_clients(inst, sat, cur, battery, t, remaining)
        if feasible:
            c = min(feasible, key=lambda x: _absolute_client_score(inst, cur, t, x))
            cur, battery, t = _absolute_append_client(inst, route, arrivals, cur, battery, t, c)
            remaining.remove(c)
            consecutive_station_moves = 0
            continue

        best_station = _absolute_best_helpful_station(inst, sat, cur, battery, t, remaining)
        if best_station is None:
            debug.append(f"absolute_rebuild_no_station_help_for:{sorted(remaining)}")
            ok = False
            break

        consecutive_station_moves += 1
        if consecutive_station_moves > max_station_moves:
            debug.append(f"absolute_rebuild_too_many_station_moves:{consecutive_station_moves}")
            ok = False
            break

        cur, battery, t = _absolute_append_station(inst, route, cur, battery, t, best_station)

    # Try to return to the satellite, possibly through one last charging station
    route, battery, t, return_ok = _absolute_finalize_return(inst, sat, route, cur, battery, t, debug)
    ok = ok and return_ok

    # Compute any residual lateness and update the final feasibility flag
    residual, lateness_ok = _absolute_residual_lateness(inst, arrivals, debug)
    ok = ok and lateness_ok

    return _safe_route(route, vehicle_id, sat), float(residual), arrivals, debug, bool(ok)


def _absolute_exact_failure(vehicle_id: int, sat_id: int, msg: str) -> Tuple[Route, float, Dict[int, float], List[str], bool]:
    """Builds a standard failure return for the exact absolute-time route solver."""
    return Route(nodes=[sat_id, sat_id], vehicle_id=vehicle_id), 0.0, {}, [msg], False


def _absolute_exact_add_label(
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
    """Adds a non-dominated label to the exact absolute-time DP frontier."""
    key = (node, mask)
    cur = labels.get(key, [])

    # Discard labels dominated by an existing state with earlier time and higher battery
    for t0, b0, _ in cur:
        if t0 <= time_val + 1e-9 and b0 >= battery_val - 1e-9:
            return None

    # Remove labels dominated by the new one before inserting it
    new_labels = [
        (t0, b0, lid)
        for (t0, b0, lid) in cur
        if not (t0 >= time_val - 1e-9 and b0 <= battery_val + 1e-9)
    ]

    lid = next_id_ref[0]
    next_id_ref[0] += 1
    new_labels.append((time_val, battery_val, lid))
    labels[key] = new_labels
    prev[lid] = (parent, node, raw_arrival)
    info[lid] = (node, mask, time_val, battery_val)
    heapq.heappush(pq, (time_val, -battery_val, lid))
    return lid


def _absolute_exact_try_client_extensions(
    inst: Instance2E,
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
    """Generates feasible client-visit extensions from one exact absolute-time label."""
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

        _absolute_exact_add_label(
            labels,
            prev,
            info,
            pq,
            next_id_ref,
            c,
            mask | bit,
            start + float(inst.service_time(c)),
            battery - h * d,
            lid,
            start,
        )


def _absolute_exact_try_station_extensions(
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
    BCe: float,
    h: float,
) -> None:
    """Generates feasible charging-station extensions from one exact absolute-time label."""
    for r in inst.station_ids:
        if r == node:
            continue

        d = _dist(inst, node, r)
        if battery - h * d < -1e-9:
            continue

        after = battery - h * d
        time2 = cur_time + d + float(inst.ge) * max(0.0, BCe - after)
        _absolute_exact_add_label(
            labels,
            prev,
            info,
            pq,
            next_id_ref,
            r,
            mask,
            time2,
            BCe,
            lid,
            time2,
        )


def _absolute_exact_reconstruct_route(
    inst: Instance2E,
    sat_id: int,
    vehicle_id: int,
    prev: Dict[int, Tuple[Optional[int], int, Optional[float]]],
    end_lid: int,
) -> Tuple[Route, float, Dict[int, float], List[str], bool]:
    """Reconstructs the final exact absolute-time route from the predecessor map."""
    seq = []
    lid = end_lid
    while lid is not None:
        parent, node, raw = prev[lid]
        seq.append((node, raw))
        lid = parent
    seq.reverse()

    # Rebuild the explicit node sequence and client arrival times
    nodes = [sat_id]
    arrs: Dict[int, float] = {}
    for node, raw in seq[1:]:
        nodes.append(node)
        if node in inst.client_ids and raw is not None:
            arrs[int(node)] = float(raw)
    nodes.append(sat_id)

    return _safe_route(nodes, vehicle_id, sat_id), 0.0, arrs, [], True


def _exact_absolute_route_for_subset(inst: Instance2E, sat_id: int, clients: List[int], start_time: float, vehicle_id: int) -> Tuple[Route, float, Dict[int, float], List[str], bool]:
    """Solves a small absolute-time routing subproblem exactly with label-setting dynamic programming."""
    clients = sorted(set(clients))
    if not clients:
        return Route(nodes=[sat_id, sat_id], vehicle_id=vehicle_id), 0.0, {}, [], True

    # Keep the exact solver only for genuinely small subsets and reject immediate infeasibilities
    if len(clients) > 6:
        return _absolute_exact_failure(vehicle_id, sat_id, 'absolute_exact_subset_too_large')
    if sum(float(inst.demand(c)) for c in clients) > float(inst.Q2) + 1e-9:
        return _absolute_exact_failure(vehicle_id, sat_id, 'absolute_exact_capacity_fail')

    client_to_bit = {c: i for i, c in enumerate(clients)}
    full_mask = (1 << len(clients)) - 1
    BCe = float(inst.BCe)
    h = float(inst.h)

    labels: Dict[Tuple[int, int], List[Tuple[float, float, int]]] = {}
    prev: Dict[int, Tuple[Optional[int], int, Optional[float]]] = {}
    info: Dict[int, Tuple[int, int, float, float]] = {}
    pq: List[Tuple[float, float, int]] = []
    next_id_ref = [0]

    _absolute_exact_add_label(
        labels,
        prev,
        info,
        pq,
        next_id_ref,
        sat_id,
        0,
        float(start_time),
        BCe,
        None,
        None,
    )

    best_end = None
    expanded = 0

    # Explore non-dominated labels ordered by time until a complete feasible route is found
    while pq:
        expanded += 1
        if next_id_ref[0] > 12000 or expanded > 25000:
            return _absolute_exact_failure(vehicle_id, sat_id, 'absolute_exact_budget_exceeded')

        _, _, lid = heapq.heappop(pq)
        node, mask, cur_time, battery = info[lid]

        if mask == full_mask:
            dback = _dist(inst, node, sat_id)
            if battery - h * dback >= -1e-9:
                best_end = (lid, cur_time + dback)
                break

        _absolute_exact_try_client_extensions(
            inst,
            clients,
            client_to_bit,
            labels,
            prev,
            info,
            pq,
            next_id_ref,
            lid,
            node,
            mask,
            cur_time,
            battery,
            h,
        )
        _absolute_exact_try_station_extensions(
            inst,
            labels,
            prev,
            info,
            pq,
            next_id_ref,
            lid,
            node,
            mask,
            cur_time,
            battery,
            BCe,
            h,
        )

    if best_end is None:
        return _absolute_exact_failure(vehicle_id, sat_id, 'absolute_exact_no_route')

    # Reconstruct the best exact route and its client arrival times
    end_lid, _ = best_end
    return _absolute_exact_reconstruct_route(inst, sat_id, vehicle_id, prev, end_lid)


def _absolute_global_failure(msgs: List[str]) -> Tuple[List[Route], float, Dict[int, float], List[str], bool]:
    """Builds a standard failure return for the global absolute-time satellite rebuild."""
    return [], 0.0, {}, list(msgs), False


def _absolute_global_exact_or_heuristic_subset(
    inst: Instance2E,
    sat: int,
    orderless_clients: List[int],
    start_time: float,
    vehicle_id: int,
    subset_cache: Dict[Tuple[int, ...], Tuple[Route, float, Dict[int, float], List[str], bool]],
    exact_budget_ref: List[int],
) -> Tuple[Route, float, Dict[int, float], List[str], bool]:
    """
    Evaluates one client subset with exact absolute reconstruction when small 
    enough, otherwise with the heuristic rebuild.
    """
    key = tuple(sorted(orderless_clients))
    if key not in subset_cache:
        # Use the exact solver only on very small subsets and under a bounded exact-search budget
        if len(key) > 6 or exact_budget_ref[0] > 200:
            subset_cache[key] = _rebuild_route_absolute(inst, sat, list(key), start_time, vehicle_id)
        else:
            subset_cache[key] = _exact_absolute_route_for_subset(inst, sat, list(key), start_time, vehicle_id)
            exact_budget_ref[0] += 1
    return subset_cache[key]


def _absolute_global_greedy_repartition(
    inst: Instance2E,
    sat: int,
    clients: List[int],
    start_time: float,
    vehicle_ids: List[int],
) -> Tuple[Optional[List[List[int]]], Dict[int, float], List[str], bool]:
    """Builds a bounded greedy repartition of clients over the available vehicles for one satellite."""
    K = len(vehicle_ids)
    local_bins = [[] for _ in range(K)]
    local_loads = [0.0 for _ in range(K)]
    debug = ['absolute_global_greedy_fallback']
    arrivals_local: Dict[int, float] = {}

    # Insert clients one by one into the best currently feasible vehicle subset
    for c in clients:
        best_k = None
        best_score = None
        best_arr: Dict[int, float] = {}

        for k in range(K):
            if local_loads[k] + float(inst.demand(c)) > float(inst.Q2) + 1e-9:
                continue

            trial = sorted(local_bins[k] + [c])
            route, pen, arr, dbg, ok = _rebuild_route_absolute(inst, sat, trial, start_time, vehicle_ids[k])
            if not ok:
                continue

            score = (
                len(route.nodes),
                pen,
                sum(_dist(inst, route.nodes[i], route.nodes[i + 1]) for i in range(len(route.nodes) - 1)),
                local_loads[k],
            )
            if best_score is None or score < best_score:
                best_score = score
                best_k = k
                best_arr = arr

        if best_k is None:
            debug.append(f'absolute_global_greedy_failed_client:{c}')
            return None, {}, debug, False

        local_bins[best_k] = sorted(local_bins[best_k] + [c])
        local_loads[best_k] += float(inst.demand(c))
        arrivals_local.update(best_arr)

    return local_bins, arrivals_local, debug, True


def _absolute_global_state(i: int, bins: List[List[int]]) -> Tuple[int, Tuple[Tuple[int, ...], ...]]:
    """Builds a compact DFS state signature for the global satellite repartition search."""
    return (i, tuple(tuple(sorted(b)) for b in bins))


def _absolute_global_dfs_repartition(
    inst: Instance2E,
    sat: int,
    clients: List[int],
    start_time: float,
    vehicle_ids: List[int],
    bins: List[List[int]],
    loads: List[float],
    seen: Set[Tuple[int, Tuple[Tuple[int, ...], ...]]],
    subset_cache: Dict[Tuple[int, ...], Tuple[Route, float, Dict[int, float], List[str], bool]],
    exact_budget_ref: List[int],
    dfs_budget_ref: List[int],
) -> Optional[List[List[int]]]:
    """
    Searches a feasible repartition of clients over vehicles with a bounded DFS
    and subset feasibility checks.
    """
    K = len(vehicle_ids)

    def dfs(i: int) -> bool:
        dfs_budget_ref[0] += 1
        if dfs_budget_ref[0] > 12000 or len(seen) > 12000:
            return False

        if i >= len(clients):
            return True

        st = _absolute_global_state(i, bins)
        if st in seen:
            return False
        seen.add(st)

        c = clients[i]
        order = sorted(range(K), key=lambda k: (len(bins[k]), loads[k]))

        # Try assigning the next client to each feasible vehicle bin
        for k in order:
            if loads[k] + float(inst.demand(c)) > float(inst.Q2) + 1e-9:
                continue

            trial = sorted(bins[k] + [c])
            _, _, _, _, ok = _absolute_global_exact_or_heuristic_subset(
                inst, sat, trial, start_time, vehicle_ids[k], subset_cache, exact_budget_ref
            )
            if not ok:
                continue

            old = list(bins[k])
            oldload = loads[k]
            bins[k] = trial
            loads[k] = oldload + float(inst.demand(c))

            if dfs(i + 1):
                return True

            bins[k] = old
            loads[k] = oldload

        return False

    if dfs(0):
        return [list(b) for b in bins]
    return None


def _absolute_global_build_final_routes(
    inst: Instance2E,
    sat: int,
    best: List[List[int]],
    start_time: float,
    vehicle_ids: List[int],
    final_debug: List[str],
) -> Tuple[List[Route], float, Dict[int, float], List[str], bool]:
    """
    Builds the final explicit routes from the chosen client repartition and 
    applies a robust fallback when needed.
    """
    routes: List[Route] = []
    arrivals: Dict[int, float] = {}

    # Materialize each subset as an explicit route, preferring exact reconstruction on small subsets
    for subset, vid in zip(best, vehicle_ids):
        if len(subset) <= 6:
            route, pen, arr, dbg, ok = _exact_absolute_route_for_subset(inst, sat, subset, start_time, vid)
        else:
            route, pen, arr, dbg, ok = _rebuild_route_absolute(inst, sat, subset, start_time, vid)

        if not ok:
            route, pen, arr, dbg2, ok2 = _rebuild_route_absolute(inst, sat, subset, start_time, vid)
            dbg = list(dbg) + list(dbg2) + ['absolute_global_rebuild_fallback']
            ok = ok2

        if not ok:
            return _absolute_global_failure(dbg + ['absolute_global_rebuild_final_fail'])

        if len(route.nodes) > 2:
            routes.append(route)
        arrivals.update(arr)
        final_debug.extend(dbg)

    return routes, 0.0, arrivals, final_debug, True


def _rebuild_satellite_routes_absolute_global(
    inst: Instance2E,
    sat: int,
    clients: List[int],
    start_time: float,
    vehicle_ids: List[int],
) -> Tuple[List[Route], float, Dict[int, float], List[str], bool]:
    """
    Globally rebuilds all second-level routes of one satellite in absolute time
    under a fixed number of vehicles.
    """
    K = len(vehicle_ids)
    if K <= 0:
        return _absolute_global_failure(['absolute_global_no_vehicle'])

    clients = sorted(
        set(clients),
        key=lambda c: (float(inst.tw_late(c)) - float(inst.tw_early(c)), -float(inst.demand(c))),
    )
    if sum(float(inst.demand(c)) for c in clients) > K * float(inst.Q2) + 1e-9:
        return _absolute_global_failure(['absolute_global_capacity_impossible'])

    bins = [[] for _ in range(K)]
    loads = [0.0 for _ in range(K)]
    subset_cache: Dict[Tuple[int, ...], Tuple[Route, float, Dict[int, float], List[str], bool]] = {}
    seen: Set[Tuple[int, Tuple[Tuple[int, ...], ...]]] = set()
    dfs_budget_ref = [0]
    exact_budget_ref = [0]

    # Use a greedy repartition first on larger or more difficult satellite subproblems
    use_greedy_only = (len(clients) >= 8 and K >= 2) or len(clients) >= 10
    greedy_debug: List[str] = []
    best: Optional[List[List[int]]] = None

    if use_greedy_only:
        greedy_best, _, greedy_debug, greedy_ok = _absolute_global_greedy_repartition(
            inst, sat, clients, start_time, vehicle_ids
        )
        if greedy_ok and greedy_best is not None:
            best = greedy_best

    # If needed, try the bounded DFS repartition, then fall back again to the greedy version
    if best is None:
        best = _absolute_global_dfs_repartition(
            inst,
            sat,
            clients,
            start_time,
            vehicle_ids,
            bins,
            loads,
            seen,
            subset_cache,
            exact_budget_ref,
            dfs_budget_ref,
        )

    if best is None:
        greedy_best, _, greedy_debug, greedy_ok = _absolute_global_greedy_repartition(
            inst, sat, clients, start_time, vehicle_ids
        )
        if not greedy_ok or greedy_best is None:
            return _absolute_global_failure(['absolute_global_repartition_failed'] + greedy_debug)
        best = greedy_best

    # Materialize the chosen repartition into final explicit routes
    final_debug = ['absolute_global_repartition_success'] + greedy_debug
    return _absolute_global_build_final_routes(inst, sat, best, start_time, vehicle_ids, final_debug)


def repair_and_replay_lvl2_absolute(inst: Instance2E, sol: Solution2E, A_s: Dict[int, float], max_attempts: int = 4) -> Tuple[Solution2E, float, Dict[int, float], Dict[int, float], List[str], bool]:
    """
    Repairs and replays all second-level routes in absolute time by validating, 
    rebuilding, or globally reconstructing satellite routes.
    """
    routes_new: Dict[int, List[Route]] = {s: [] for s in inst.satellite_ids}
    abs_arrival: Dict[int, float] = {}
    abs_depart_sat: Dict[int, float] = {}
    total_pen = 0.0
    debug: List[str] = []
    ok_all = True
    
    # Process each satellite independently by replaying its EV routes in absolute time
    for s, routes in sol.routes_lvl2.items():
        abs_depart_sat[s] = float(A_s.get(s, float("inf")))
        sat_start = float(A_s.get(s, float("inf")))
        sat_clients: List[int] = []
        sat_vehicle_ids: List[int] = []
        sat_ok = True
        sat_routes_trial: List[Route] = []
        sat_arrivals_trial: Dict[int, float] = {}
        sat_penalty_trial = 0.0
        
        # First attempt: validate each existing route, otherwise rebuild it individually
        for r in routes:
            clients = [n for n in r.nodes if n in inst.client_ids]
            sat_clients.extend(clients)
            sat_vehicle_ids.append(r.vehicle_id)
            explicit, pen, arr, dbg, ok = _validate_explicit_absolute_route(inst, s, list(r.nodes), sat_start, r.vehicle_id)
            if ok:
                rebuilt, pen, arr = explicit, pen, arr
                dbg = list(dbg) + ['absolute_explicit_preserved']
            else:
                rebuilt, pen, arr, dbg, ok = _rebuild_route_absolute(inst, s, clients, sat_start, r.vehicle_id)
            sat_routes_trial.append(rebuilt)
            sat_arrivals_trial.update(arr)
            sat_penalty_trial += pen
            debug.extend(dbg)
            sat_ok = sat_ok and ok
            
        # If all routes are valid after local repair, keep them
        if sat_ok:
            routes_new[s].extend(sat_routes_trial)
            abs_arrival.update(sat_arrivals_trial)
            total_pen += sat_penalty_trial
            
        # Otherwise, attempt a global reconstruction of all routes for this satellite
        elif sat_clients:
            rebuilt_routes, pen2, arr2, dbg2, ok2 = _rebuild_satellite_routes_absolute_global(
                inst, s, sat_clients, sat_start, sat_vehicle_ids
            )
            debug.extend(dbg2)
            if ok2:
                routes_new[s].extend(rebuilt_routes)
                abs_arrival.update(arr2)
                total_pen += pen2
            else:
                # Last-resort fallback: keep original routes to avoid losing all clients
                preserved = [_safe_route(list(r.nodes), r.vehicle_id, s) for r in routes]
                routes_new[s].extend(preserved)
                debug.append(f'absolute_global_preserve_original_routes:{s}')
                sat_ok = False
            ok_all = ok_all and sat_ok
        else:
            ok_all = ok_all and sat_ok
    
    # Build the final consistent solution and refresh assignment metadata
    sol2 = Solution2E(assignment=dict(sol.assignment), routes_lvl2={s: rs for s, rs in routes_new.items() if rs}, routes_lvl1=list(sol.routes_lvl1), meta=dict(sol.meta))
    sol2 = _refresh_solution_consistency(inst, sol2)
    return sol2, float(total_pen), abs_arrival, abs_depart_sat, debug, ok_all

