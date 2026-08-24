"""
ccz_error_model.py

Logical error estimator for an MCT decomposed into Toffoli gates,
where every Toffoli consumes one injected |CCZ> resource state.

Expected placement-graph format
-------------------------------
Computation module:
    kind="module"
    placed_qubits=[0, 1, 2, ...]

Factory:
    kind="factory"

Expected hyperedge format
-------------------------
{
    (control_1, control_2, target): frequency,
    ...
}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Hashable

import networkx as nx


@dataclass(frozen=True)
class CCZErrorRates:
    """
    Logical error rates used by the model.

    All values must be probabilities in the range [0, 1).
    """

    # ERROR SOURCE 1:
    # The factory produces an incorrect |CCZ> resource state.
    p_factory: float

    # ERROR SOURCE 2:
    # One logical inter-module communication primitive fails.
    p_link: float

    # ERROR SOURCE 3:
    # The CCZ-state injection or teleportation gadget fails.
    #
    # Do not include communication faults here if they are already
    # represented by p_link and the routing cost.
    p_injection: float = 0.0

    # ERROR SOURCE 4:
    # A local logical operation inside a computation module fails.
    p_local: float = 0.0

    # ERROR SOURCE 5:
    # Logical Hadamard error. A Toffoli implemented as H-CCZ-H
    # normally uses two Hadamards on the target.
    p_hadamard: float = 0.0

    # ERROR SOURCE 6:
    # Memory or idle error during one relevant idle interval.
    p_idle: float = 0.0


@dataclass(frozen=True)
class CCZOperationCounts:
    """
    Non-routing logical-operation counts for one injected Toffoli.
    """

    # Number of local logical operations used by the injection gadget.
    local_operations_per_toffoli: int = 0

    # H-CCZ-H gives two Hadamards.
    hadamards_per_toffoli: int = 2

    # Relevant idle intervals per Toffoli.
    idle_intervals_per_toffoli: int = 0


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(
            f"{name} must satisfy 0 <= {name} < 1; "
            f"received {value}."
        )


def _validate_inputs(
    rates: CCZErrorRates,
    counts: CCZOperationCounts,
) -> None:
    for name, value in vars(rates).items():
        _validate_probability(name, value)

    for name, value in vars(counts).items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer; "
                f"received {value!r}."
            )


def _extract_qubit_placement(
    architecture: nx.Graph,
) -> dict[Hashable, Hashable]:
    """
    Return:
        logical qubit -> computation module
    """

    qubit_to_module = {}

    for node, data in architecture.nodes(data=True):
        if data.get("kind") != "module":
            continue

        placed_qubits = data.get("placed_qubits", [])

        if not isinstance(placed_qubits, (list, tuple, set)):
            raise TypeError(
                f"Node {node!r} has an invalid placed_qubits "
                f"attribute: {placed_qubits!r}."
            )

        for qubit in placed_qubits:
            if qubit in qubit_to_module:
                previous_module = qubit_to_module[qubit]

                raise ValueError(
                    f"Logical qubit {qubit!r} is placed in both "
                    f"{previous_module!r} and {node!r}."
                )

            qubit_to_module[qubit] = node

    if not qubit_to_module:
        raise ValueError(
            "No logical qubits were found in module "
            "placed_qubits attributes."
        )

    return qubit_to_module


def _extract_factories(
    architecture: nx.Graph,
) -> list[Hashable]:
    factories = [
        node
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "factory"
    ]

    if not factories:
        raise ValueError(
            "The architecture graph contains no factory nodes."
        )

    return factories


def _metric_mst_cost(
    terminals: set[Hashable],
    shortest_distances: dict,
) -> float:
    """
    Calculate the MST cost on the terminal metric closure.

    This matches the routing-volume abstraction used previously:
    pairwise terminal distances are shortest-path distances in the
    architecture, and an MST is constructed over those terminals.
    """

    terminals = set(terminals)

    if len(terminals) <= 1:
        return 0.0

    metric_graph = nx.Graph()
    metric_graph.add_nodes_from(terminals)

    for left, right in combinations(terminals, 2):
        try:
            distance = shortest_distances[left][right]
        except KeyError as error:
            raise nx.NetworkXNoPath(
                f"No architecture path exists between "
                f"{left!r} and {right!r}."
            ) from error

        metric_graph.add_edge(
            left,
            right,
            weight=distance,
        )

    mst = nx.minimum_spanning_tree(
        metric_graph,
        weight="weight",
    )

    return float(mst.size(weight="weight"))


def _one_toffoli_log_success(
    routing_cost: float,
    rates: CCZErrorRates,
    counts: CCZOperationCounts,
) -> float:
    """
    Return log(success probability) for one injected Toffoli.
    """

    # ERROR SOURCE 1: one consumed |CCZ> state.
    log_factory_success = math.log1p(-rates.p_factory)

    # ERROR SOURCE 2: routing/inter-module communication.
    log_routing_success = (
        routing_cost * math.log1p(-rates.p_link)
    )

    # ERROR SOURCE 3: injection gadget.
    log_injection_success = math.log1p(
        -rates.p_injection
    )

    # ERROR SOURCE 4: local logical operations.
    log_local_success = (
        counts.local_operations_per_toffoli
        * math.log1p(-rates.p_local)
    )

    # ERROR SOURCE 5: H gates used to turn CCZ into Toffoli.
    log_hadamard_success = (
        counts.hadamards_per_toffoli
        * math.log1p(-rates.p_hadamard)
    )

    # ERROR SOURCE 6: memory and idle intervals.
    log_idle_success = (
        counts.idle_intervals_per_toffoli
        * math.log1p(-rates.p_idle)
    )

    return (
        log_factory_success
        + log_routing_success
        + log_injection_success
        + log_local_success
        + log_hadamard_success
        + log_idle_success
    )


def estimate_ccz_error(
    architecture: nx.Graph,
    toffoli_hyperedges: dict[tuple, int],
    rates: CCZErrorRates,
    counts: CCZOperationCounts | None = None,
    edge_weight: str | None = "weight",
    require_integer_routing_cost: bool = True,
) -> dict:
    """
    Estimate logical failure probability from a completed placement.

    Parameters
    ----------
    architecture
        NetworkX placement graph.

    toffoli_hyperedges
        Dictionary:
            (control_1, control_2, target) -> frequency

    rates
        Logical error probabilities.

    counts
        Local-operation counts for one injected Toffoli.

    edge_weight
        Edge attribute used for shortest paths.

        Use "weight" when graph weights count logical communication
        primitives.

        Use None to count every architecture edge as one hop.

    require_integer_routing_cost
        When True, reject fractional routing costs. This is useful
        when p_link means error per logical communication operation.

    Returns
    -------
    dict
        Summary and per-hyperedge results.
    """

    if counts is None:
        counts = CCZOperationCounts()

    _validate_inputs(rates, counts)

    if not toffoli_hyperedges:
        raise ValueError(
            "The Toffoli hyperedge dictionary is empty."
        )

    qubit_to_module = _extract_qubit_placement(
        architecture
    )

    factories = _extract_factories(architecture)

    shortest_distances = dict(
        nx.all_pairs_dijkstra_path_length(
            architecture,
            weight=edge_weight,
        )
    )

    total_ccz_count = 0
    routing_volume = 0.0
    total_log_success = 0.0
    per_hyperedge = []

    for hyperedge, frequency in toffoli_hyperedges.items():
        if len(hyperedge) != 3:
            raise ValueError(
                "Every Toffoli hyperedge must contain exactly "
                f"three qubits; received {hyperedge!r}."
            )

        if not isinstance(frequency, int) or frequency <= 0:
            raise ValueError(
                f"Frequency for {hyperedge!r} must be a "
                f"positive integer; received {frequency!r}."
            )

        missing_qubits = [
            qubit
            for qubit in hyperedge
            if qubit not in qubit_to_module
        ]

        if missing_qubits:
            raise ValueError(
                f"Hyperedge {hyperedge!r} contains unplaced "
                f"qubits: {missing_qubits}."
            )

        operand_modules = {
            qubit_to_module[qubit]
            for qubit in hyperedge
        }

        factory_candidates = []

        for factory in factories:
            terminals = operand_modules | {factory}

            routing_cost = _metric_mst_cost(
                terminals,
                shortest_distances,
            )

            if (
                require_integer_routing_cost
                and not math.isclose(
                    routing_cost,
                    round(routing_cost),
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    f"Routing cost {routing_cost} for hyperedge "
                    f"{hyperedge!r} is fractional. A fractional "
                    "cost should not be used as the exponent of "
                    "(1 - p_link) unless p_link is calibrated per "
                    "unit of weighted distance."
                )

            log_success = _one_toffoli_log_success(
                routing_cost,
                rates,
                counts,
            )

            failure_probability = -math.expm1(
                log_success
            )

            factory_candidates.append(
                {
                    "factory": factory,
                    "routing_cost": routing_cost,
                    "log_success": log_success,
                    "failure_probability":
                        failure_probability,
                }
            )

        # Select the factory giving the lowest predicted error.
        # With identical factories and uniform p_link, this is also
        # the factory with minimum routing cost.
        selected = min(
            factory_candidates,
            key=lambda candidate: (
                candidate["failure_probability"],
                str(candidate["factory"]),
            ),
        )

        routing_cost = selected["routing_cost"]
        log_success_one = selected["log_success"]
        failure_one = selected["failure_probability"]

        weighted_log_success = (
            frequency * log_success_one
        )

        total_ccz_count += frequency
        routing_volume += frequency * routing_cost
        total_log_success += weighted_log_success

        # First-order contributions are useful for explaining
        # which error source dominates. They are approximations
        # and are not exactly additive at larger error rates.
        factory_component = rates.p_factory
        routing_component = routing_cost * rates.p_link
        injection_component = rates.p_injection

        local_component = (
            counts.local_operations_per_toffoli
            * rates.p_local
        )

        hadamard_component = (
            counts.hadamards_per_toffoli
            * rates.p_hadamard
        )

        idle_component = (
            counts.idle_intervals_per_toffoli
            * rates.p_idle
        )

        per_hyperedge.append(
            {
                "hyperedge": hyperedge,
                "frequency": frequency,
                "operand_modules":
                    sorted(operand_modules, key=str),
                "selected_factory":
                    selected["factory"],
                "routing_cost": routing_cost,
                "failure_one_execution": failure_one,
                "failure_all_executions": (
                    -math.expm1(weighted_log_success)
                ),

                # Highlighted first-order error sources:
                "factory_error_component":
                    factory_component,
                "routing_error_component":
                    routing_component,
                "injection_error_component":
                    injection_component,
                "local_error_component":
                    local_component,
                "hadamard_error_component":
                    hadamard_component,
                "idle_error_component":
                    idle_component,

                "weighted_routing_cost":
                    frequency * routing_cost,
            }
        )

    exact_failure_probability = -math.expm1(
        total_log_success
    )

    # First-order summary terms.
    factory_error_component = (
        total_ccz_count * rates.p_factory
    )

    routing_error_component = (
        routing_volume * rates.p_link
    )

    injection_error_component = (
        total_ccz_count * rates.p_injection
    )

    local_error_component = (
        total_ccz_count
        * counts.local_operations_per_toffoli
        * rates.p_local
    )

    hadamard_error_component = (
        total_ccz_count
        * counts.hadamards_per_toffoli
        * rates.p_hadamard
    )

    idle_error_component = (
        total_ccz_count
        * counts.idle_intervals_per_toffoli
        * rates.p_idle
    )

    first_order_failure_probability = (
        factory_error_component
        + routing_error_component
        + injection_error_component
        + local_error_component
        + hadamard_error_component
        + idle_error_component
    )

    return {
        "summary": {
            "ccz_count": total_ccz_count,
            "routing_volume": routing_volume,
            "exact_failure_probability":
                exact_failure_probability,
            "exact_success_probability":
                1.0 - exact_failure_probability,
            "first_order_failure_probability":
                first_order_failure_probability,

            # Error-source breakdown for review:
            "factory_error_component":
                factory_error_component,
            "routing_error_component":
                routing_error_component,
            "injection_error_component":
                injection_error_component,
            "local_error_component":
                local_error_component,
            "hadamard_error_component":
                hadamard_error_component,
            "idle_error_component":
                idle_error_component,
        },
        "per_hyperedge": per_hyperedge,
    }