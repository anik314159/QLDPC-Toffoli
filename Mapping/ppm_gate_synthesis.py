from __future__ import annotations

from Applications.MCTdecomp.binary_tree_decomposition import extract_toffoli_hyperedges, create_circuit
from linear_mapping import linear_mapping_tree_baseline
from itertools import combinations
from pprint import pprint
from collections.abc import Callable, Sequence
from typing import Any, Hashable
import networkx as nx


ToffoliEdge = tuple[int, int, int]
LogicalQubit = Hashable
PBCInstruction = dict[str, Any]
CCZAssignment = dict[str, Any]

def _metric_mst_cost(
    terminals,
    shortest_distances,
):
    """
    Return the MST cost connecting a set of terminal nodes in the
    shortest-path metric of the architecture graph.

    Parameters
    ----------
    terminals
        Factory node plus the modules containing the Toffoli qubits.

    shortest_distances
        Output of:

        dict(nx.all_pairs_dijkstra_path_length(
            architecture,
            weight="weight",
        ))

    Returns
    -------
    float
        Total weight of the terminal metric-closure MST.
    """

    terminals = set(terminals)

    if len(terminals) <= 1:
        return 0.0

    metric_graph = nx.Graph()
    metric_graph.add_nodes_from(terminals)

    # Build a complete graph between all terminal nodes.
    # Each edge weight is their shortest-path distance in the
    # original architecture.
    for left, right in combinations(terminals, 2):
        if (
            left not in shortest_distances
            or right not in shortest_distances[left]
        ):
            raise nx.NetworkXNoPath(
                f"No path exists between {left!r} and {right!r}."
            )

        metric_graph.add_edge(
            left,
            right,
            weight=shortest_distances[left][right],
        )

    mst = nx.minimum_spanning_tree(
        metric_graph,
        weight="weight",
    )

    return float(
        mst.size(weight="weight")
    )

def build_ccz_instruction_sequence(
    architecture,
    sequential_toffolis,
    weight="weight",
):
    qubit_to_module = {}

    for node, data in architecture.nodes(data=True):
        if data.get("kind") != "module":
            continue

        for qubit in data.get("placed_qubits", []):
            qubit_to_module[qubit] = node

    factories = [
        node
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "factory"
    ]

    distances = dict(
        nx.all_pairs_dijkstra_path_length(
            architecture,
            weight=weight,
        )
    )

    sequence = []

    for operation_index, (c1, c2, target) in enumerate(
        sequential_toffolis
    ):
        operand_modules = {
            qubit_to_module[c1],
            qubit_to_module[c2],
            qubit_to_module[target],
        }

        factory_candidates = []

        for factory in factories:
            terminals = operand_modules | {factory}

            mst_cost = _metric_mst_cost(
                terminals,
                distances,
            )

            factory_candidates.append(
                (mst_cost, str(factory), factory)
            )

        mst_cost, _, selected_factory = min(
            factory_candidates
        )

        r0, r1, r2 = (
            f"{selected_factory}_r0",
            f"{selected_factory}_r1",
            f"{selected_factory}_r2",
        )

        sequence.append(
            {
                "operation_index": operation_index,
                "toffoli": (c1, c2, target),
                "data_modules": {
                    c1: qubit_to_module[c1],
                    c2: qubit_to_module[c2],
                    target: qubit_to_module[target],
                },
                "factory": selected_factory,
                "factory_qubits": (r0, r1, r2),
                "factory_mst_cost": mst_cost,
            }
        )

    return sequence

toffoli_hyperedges,toffoli_sequenced_hyperedge = extract_toffoli_hyperedges(create_circuit(12))
architecture = linear_mapping_tree_baseline(
    control_count=12,
    toffoli_hyperedge_dict=toffoli_hyperedges,
    controls_per_module=4,
    module_capacity=11,
    factory_period=0,
)
sequence = build_ccz_instruction_sequence(architecture,toffoli_sequenced_hyperedge)
pprint(sequence, sort_dicts=False, width=100)