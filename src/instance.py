from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from pathlib import Path


@dataclass(frozen=True) # crée une classe dont les données sont immuables
class Node:
    """Represents a node in the 2E-EVRP instance with fixed attributes such as location and demand."""
    id: int
    id: int
    x: float
    y: float
    demand: float = 0.0


@dataclass
class Instance2E:
    """
    Encapsulates all data defining a 2E-EVRP-TW instance, including network 
    structure, vehicle constraints, and time-related parameters.
    """
    # Core network structure: depot, satellites, charging stations, and clients
    depot: Node
    satellites: List[Node]
    stations: List[Node]
    clients: List[Node]
    
    # Vehicle and energy parameters for both echelons
    Q1: float   # capacity of 1st echelon vehicles
    Q2: float   # capacity of 2nd echelon vehicles
    BCe: float  # battery capacity for EVs
    h: float    # consumption rate per distance unit
    ge: float   # charging time
    
    # Fleet size limits for both echelons
    nv1: int    # max number of 1st echelon vehicles 
    nv2 : int   # idem for 2nd echelon
    
    # Time windows / service times (by node id)
    service_time_by_id: Dict[int, float] 
    tw_early_by_id: Dict[int, float] 
    tw_late_by_id: Dict[int, float] 

    # Function summary: service time.
    def service_time(self, node_id: int) -> float:
        return float(self.service_time_by_id.get(node_id, 0.0))

    # Function summary: tw early.
    def tw_early(self, node_id: int) -> float:
        return float(self.tw_early_by_id.get(node_id, 0.0))

    # Function summary: tw late.
    def tw_late(self, node_id: int) -> float:
        return float(self.tw_late_by_id.get(node_id, float("inf")))

    # Function summary: demand.
    def demand(self, node_id: int) -> float:
        node = self.node_by_id.get(node_id)
        return float(getattr(node, "demand", 0.0)) if node is not None else 0.0

    # Function summary: node.
    def node(self, node_id: int) -> Node:
        node = self.node_by_id.get(node_id)
        if node is None:
            raise KeyError(f"Unknown node id: {node_id}")
        return node
    
    # --- computed / cached ---
    id_to_idx: Dict[int, int] = field(init=False, default_factory=dict)
    idx_to_id: List[int] = field(init=False, default_factory=list)
    nodes_by_idx: List[Node] = field(init=False, default_factory=list)
    demand_by_idx: List[float] = field(init=False, default_factory=list)
    dist: Optional[List[List[float]]] = field(init=False, default=None)
    

    @property
    def depot_id(self) -> int:
        return self.depot.id  # expected 0

    @property
    def satellite_ids(self) -> Set[int]:
        return {n.id for n in self.satellites}

    @property
    def station_ids(self) -> Set[int]:
        return {n.id for n in self.stations}

    @property
    def client_ids(self) -> Set[int]:
        return {n.id for n in self.clients}

    @property
    def all_nodes(self) -> List[Node]:
        return [self.depot] + self.satellites + self.stations + self.clients

    @property
    def n_nodes(self) -> int:
        return len(self.all_nodes)
    
    @property
    def node_by_id(self) -> Dict[int, Node]:
        # useful to debug 
        return {n.id: n for n in self.all_nodes}
    
    def build_index(self) -> None:
        """
        Builds:
          - idx_to_id : list of node_ids in the chosen global order
          - id_to_idx : reverse mapping
          - nodes_by_idx : nodes ordered by idx
          - demand_by_idx : demand aligned with idx (0 for non-clients)
        """
        nodes = self.all_nodes
        idx_to_id = [n.id for n in nodes]

        if len(set(idx_to_id)) != len(idx_to_id):
            raise ValueError(f"Duplicate node ids found: {idx_to_id}")

        self.nodes_by_idx = nodes
        self.idx_to_id = idx_to_id
        self.id_to_idx = {nid: i for i, nid in enumerate(idx_to_id)}

        # Demand aligned with idx (clients only)
        self.demand_by_idx = [0.0] * len(nodes)
        for c in self.clients:
            self.demand_by_idx[self.id_to_idx[c.id]] = float(c.demand)

    def idx(self, node_id: int) -> int:
        return self.id_to_idx[node_id]

    def nid(self, idx: int) -> int:
        return self.idx_to_id[idx]

    def require_dist(self) -> List[List[float]]:
        if self.dist is None:
            raise RuntimeError("Distance matrix not attached. Call attach_distance_matrix(inst) first.")
        return self.dist


def load_instance(path: str) -> Instance2E:
    """
    Loads a 2E-EVRP-TW benchmark instance from a .txt file and delegates 
    parsing to the internal loader.
    """
    # Convert input path to a Path object and validate file extension
    p = Path(path)
    if p.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt 2E-EVRP-TW instance, got: {p.suffix}")

    # Delegate the full parsing and construction of the instance
    return _load_instance_2e_evrp_tw_txt(p)


def _load_instance_2e_evrp_tw_txt(path: Path) -> Instance2E:
    """Loads a 2E-EVRP-TW instance from a text file and constructs the corresponding Instance2E object."""
    # Read all numeric rows from the file
    rows = _read_numeric_rows(path)

    if not rows:
        raise ValueError(f"Empty instance file: {path}")
        
    # Parse header defining instance dimensions
    header = rows[0]
    if len(header) < 6:
        raise ValueError(f"Invalid header in {path.name}: {header}")

    nv1 = int(header[0])
    nv2 = int(header[1])
    nd = int(header[2])
    ns = int(header[3])
    nr = int(header[4])
    nc = int(header[5])

    if nd != 1:
        raise NotImplementedError(
            f"Current Instance2E expects one depot, but {path.name} has nd={nd}."
        )

    idx = 1
    
    # Split rows into structured blocks (vehicles, depot, satellites, stations, clients)
    v1_rows = rows[idx: idx + nv1]
    idx += nv1
    v2_rows = rows[idx: idx + nv2]
    idx += nv2
    depot_rows = rows[idx: idx + nd]
    idx += nd
    satellite_rows = rows[idx: idx + ns]
    idx += ns
    station_rows = rows[idx: idx + nr]
    idx += nr
    client_rows = rows[idx: idx + nc]
    idx += nc
    
    # Validate block sizes
    if len(v1_rows) != nv1:
        raise ValueError(f"{path.name}: expected {nv1} lvl1 vehicle rows, got {len(v1_rows)}")
    if len(v2_rows) != nv2:
        raise ValueError(f"{path.name}: expected {nv2} lvl2 vehicle rows, got {len(v2_rows)}")
    if len(depot_rows) != nd:
        raise ValueError(f"{path.name}: expected {nd} depot rows, got {len(depot_rows)}")
    if len(satellite_rows) != ns:
        raise ValueError(f"{path.name}: expected {ns} satellite rows, got {len(satellite_rows)}")
    if len(station_rows) != nr:
        raise ValueError(f"{path.name}: expected {nr} station rows, got {len(station_rows)}")
    if len(client_rows) != nc:
        raise ValueError(f"{path.name}: expected {nc} client rows, got {len(client_rows)}")
        
    # Extract vehicle and EV parameters (assumed constant across rows)
    Q1 = _constant_or_first(v1_rows, col=0, name="Q1")
    Q2 = _constant_or_first(v2_rows, col=0, name="Q2")
    BCe = _constant_or_first(v2_rows, col=3, name="BCe")
    ge = _constant_or_first(v2_rows, col=4, name="ge")
    h = _constant_or_first(v2_rows, col=5, name="h")

    next_id = 0
    
    # Build depot node
    depot = Node(
        id=next_id,
        x=float(depot_rows[0][0]),
        y=float(depot_rows[0][1]),
    )
    next_id += 1
    
    # Build satellite nodes
    satellites: List[Node] = []
    for r in satellite_rows:
        satellites.append(
            Node(
                id=next_id,
                x=float(r[0]),
                y=float(r[1]),
            )
        )
        next_id += 1
        
    # Build charging station nodes
    stations: List[Node] = []
    for r in station_rows:
        stations.append(
            Node(
                id=next_id,
                x=float(r[0]),
                y=float(r[1]),
            )
        )
        next_id += 1
        
    # Build client nodes and associated attributes
    clients: List[Node] = []
    service_time_by_id: Dict[int, float] = {}
    tw_early_by_id: Dict[int, float] = {}
    tw_late_by_id: Dict[int, float] = {}

    for r in client_rows:
        node_id = next_id
        next_id += 1

        x = float(r[0])
        y = float(r[1])
        demand = float(r[2])
        tw_early = float(r[5]) if len(r) > 5 else 0.0
        tw_late = float(r[6]) if len(r) > 6 else float("inf")
        service = float(r[7]) if len(r) > 7 else 0.0

        clients.append(
            Node(
                id=node_id,
                x=x,
                y=y,
                demand=demand,
            )
        )

        service_time_by_id[node_id] = service
        tw_early_by_id[node_id] = tw_early
        tw_late_by_id[node_id] = tw_late
        
    # Build the final instance object and initialize its internal structures
    inst = Instance2E(
        depot=depot,
        satellites=satellites,
        stations=stations,
        clients=clients,
        Q1=float(Q1),
        Q2=float(Q2),
        BCe=float(BCe),
        h=float(h),
        ge=float(ge),
        nv1=int(nv1),
        nv2=int(nv2),
        service_time_by_id=service_time_by_id,
        tw_early_by_id=tw_early_by_id,
        tw_late_by_id=tw_late_by_id,
    )

    inst.name = path.stem
    inst.source_path = str(path)
    inst.build_index()
    return inst


def _read_numeric_rows(path: Path) -> List[List[float]]:
    """Reads a text file and converts each non-empty line into a list of numeric values."""
    rows: List[List[float]] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            
            # Convert each line into floats and detect invalid formats early
            try:
                row = [float(x) for x in parts]
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric line in {path.name} at line {lineno}: {line}"
                ) from e

            rows.append(row)

    return rows


def _constant_or_first(rows: List[List[float]], col: int, name: str, tol: float = 1e-9) -> float:
    """Checks that a given column is constant across rows and returns its value."""
    if not rows:
        raise ValueError(f"No rows available for {name}")

    if col >= len(rows[0]):
        raise ValueError(f"Column {col} missing for {name}")

    val = float(rows[0][col])
    
    # Ensure consistency across all rows for this parameter
    for i, r in enumerate(rows[1:], start=2):
        if col >= len(r):
            raise ValueError(f"Column {col} missing for {name} on row {i}")
        if abs(float(r[col]) - val) > tol:
            raise ValueError(
                f"{name} is not constant across vehicle rows: first={val}, row{i}={float(r[col])}"
            )

    return val
