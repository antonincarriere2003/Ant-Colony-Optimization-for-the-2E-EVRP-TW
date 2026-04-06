# src/solution.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# alias to make the code more readable
NodeId = int

@dataclass
class Route:
    """
    Generic route representation.

    A route is a closed sequence of node IDs:
      - Second echelon (EV): [sat_id, c1, c2, ..., sat_id]
      - First echelon (truck): [depot_id, sA, sB, ..., depot_id]
    """

    nodes: List[NodeId]
    vehicle_id: int

    # Function summary:   post init  .
    def __post_init__(self):
        """Normalize and validate the route representation.

        The solver sometimes produces partially rebuilt routes while repairing
        large 100-customer instances. Instead of crashing on benign structural
        issues, the route is normalized in-place:
          - tuples/iterables are converted to a list,
          - a singleton [s] becomes [s, s],
          - an open route [s, ..., x] becomes [s, ..., x, s].

        Truly empty routes are still rejected because there is no meaningful
        depot/satellite to infer.
        """
        if not isinstance(self.nodes, list):
            self.nodes = list(self.nodes)
        if len(self.nodes) == 0:
            raise ValueError("Route.nodes must contain at least one node.")
        if len(self.nodes) == 1:
            self.nodes = [self.nodes[0], self.nodes[0]]
        elif self.nodes[0] != self.nodes[-1]:
            self.nodes = list(self.nodes) + [self.nodes[0]]

        if self.vehicle_id is None:
            raise ValueError("vehicle_id is required for Route when enforcing vehicle constraints.")

    @property
    def start(self) -> NodeId:
        """Starting node (depot or satellite)."""
        return self.nodes[0]

    @property
    def end(self) -> NodeId:
        """Ending node (same as start since route is closed)."""
        return self.nodes[-1]

    def arcs(self) -> List[Tuple[NodeId, NodeId]]:
        """
        Returns consecutive arcs (i, j) in the route.
        Example:
            [100, 1, 3, 100]
        becomes:
            (100,1), (1,3), (3,100)
        """
        return list(zip(self.nodes[:-1], self.nodes[1:]))

    def inner_nodes(self) -> List[NodeId]:
        """
        Returns internal nodes (excluding start/end).
        """
        return self.nodes[1:-1]


@dataclass
class VehicleIdGenerator:
    """Generates unique identifiers for vehicles used in the solution."""
    next_truck_id: int = 1
    next_ev_id: int = 1

    def new_ev_id(self) -> int:
        """Returns a new unique identifier for a second-level electric vehicle."""
        vid = self.next_ev_id
        self.next_ev_id += 1
        return vid



@dataclass
class Solution2E:
    """
    Two-Echelon EVRP solution structure (without time windows for now).

    This class is solver-independent and will later be used for:
        - objective computation
        - feasibility checks
        - visualization
        - ACS integration
    """

    # Client-to-satellite assignment:
    # assignment[client_id] = satellite_id
    assignment: Dict[NodeId, NodeId] = field(default_factory=dict)

    # Second echelon routes:
    # routes_lvl2[satellite_id] = list of EV routes
    routes_lvl2: Dict[NodeId, List[Route]] = field(default_factory=dict)

    # First echelon routes:
    # List of truck routes (depot -> satellites -> depot)
    routes_lvl1: List[Route] = field(default_factory=list)

    # Optional metadata (runtime, parameters, iteration info, etc.)
    meta: Dict[str, object] = field(default_factory=dict)

    def used_satellites(self) -> List[NodeId]:
        """Returns satellites that have at least one EV route."""
        return [sat for sat, routes in self.routes_lvl2.items() if routes]

    def all_arcs_lvl2(self) -> List[Tuple[NodeId, NodeId]]:
        """Returns all arcs used in second echelon."""
        arcs: List[Tuple[NodeId, NodeId]] = []
        for routes in self.routes_lvl2.values():
            for route in routes:
                arcs.extend(route.arcs())
        return arcs
