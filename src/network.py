import math
from typing import List

from .instance import Instance2E


def _euclidean(a, b) -> float:
    """
    Computes the euclidean distance between two nodes.
    """
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def build_distance_matrix(inst: Instance2E) -> List[List[float]]:
    """
    Build dist[idx_i][idx_j] where idx are indices in inst.nodes_by_idx
    (which is defined by inst.build_index()).
    """
    nodes = inst.nodes_by_idx
    n = len(nodes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = _euclidean(nodes[i], nodes[j])
    return dist


def attach_distance_matrix(inst: Instance2E) -> None:
    """
    Computes and stores dist matrix in inst.dist
    """
    inst.dist = build_distance_matrix(inst)
