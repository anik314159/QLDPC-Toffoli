"""
square_grid_mapping.py

One call that builds a square grid, sites its factories, places the
circuit onto it, and returns a graph ready for
calculate_routing_volume.

    architecture = square_grid_mapping(
        control_count=127,
        toffoli_hyperedge_dict=hyperedges,
        factory_period=3,
        controls_per_module=4,
    )
    volume = calculate_routing_volume(
        architecture_graph=architecture,
        toffoli_hyperedge_dict=hyperedges,
    )

Shape
-----
    M       = ceil(control_count / controls_per_module)
    columns = round(sqrt(M))
    rows    = ceil(M / columns)

The shape depends on M only, NOT on the factory count. An earlier
version used round(sqrt(M * f)) to make each factory's territory
square, but that collapses to a 2-row strip once f is comparable to
M -- reproducing a long grid in exactly the regime it was meant to
improve on.

Factory count
-------------
    f = ceil(M / factory_period)

the same count a linear chain of M modules carries at that period, so
the two topologies can be compared at matched factory density. Since
routing cost is governed by M/(2f), an unmatched count would swamp any
topological effect.

Factory siting
--------------
Greedy k-medoids followed by swap local search, minimising the summed
module-to-nearest-factory distance. Factories are attached as LEAF
NODES to the chosen modules; they do not occupy grid positions and are
never used as routing waypoints.

Two caveats this implies, both of which understate cost:

  * A factory consumes no module capacity here, whereas a real one is
    a physical module with its own qubits (f + a' including the
    code-factory adapter). Add K * (f + a') by hand for footprint
    numbers -- it matters most at small factory_period, where f is
    large.

  * The factory-to-module attachment edge has weight 1, so an operand
    inside a factory's own module still costs 1 hop rather than 0.
    Uniform across configurations, so comparisons hold, but absolute
    routing volumes carry a +3 x num_toffolis offset.

Placement
---------
Three stages, as in the linear baseline, with grid-appropriate
scoring:

  1. Control groups fill modules COLUMN-MAJOR, keeping each subtree
     within one column.
  2. Subtrees whose leaves all sit in one group go to that module.
  3. Everything else goes to the module minimising true graph distance
     to its two children, tie-broken by distance to the nearest
     factory.
"""

from __future__ import annotations

import math
from functools import lru_cache

import networkx as nx


# ---------------------------------------------------------------
# Factory siting
# ---------------------------------------------------------------

def _kmedoid_sites(modules, hop, factory_count):
    """Greedy seed, then swap local search, on summed distance."""

    def total(sites):
        return sum(min(hop[m][s] for s in sites) for m in modules)

    sites = []
    for _ in range(min(factory_count, len(modules))):
        best_site, best_cost = None, math.inf
        for candidate in modules:
            if candidate in sites:
                continue
            cost = total(sites + [candidate])
            if cost < best_cost:
                best_site, best_cost = candidate, cost
        sites.append(best_site)

    current = total(sites)
    improved = True
    while improved:
        improved = False
        for index in range(len(sites)):
            for candidate in modules:
                if candidate in sites:
                    continue
                trial = list(sites)
                trial[index] = candidate
                cost = total(trial)
                if cost < current - 1e-12:
                    sites, current = trial, cost
                    improved = True
                    break
            if improved:
                break

    return sites


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------

def square_grid_mapping(
    control_count,
    toffoli_hyperedge_dict,
    factory_period=3,
    controls_per_module=4,
    module_capacity=11,
    column_major=True,
    uncompute=0,
):
    """
    Build a square grid, site its factories, place the circuit, and
    return the mapped architecture.

    Parameters
    ----------
    control_count : int
        Number of MCT controls.

    toffoli_hyperedge_dict : dict
        {(left_child, right_child, target): frequency}

    factory_period : int
        Factory density, matched to a linear chain at the same period.
        f = ceil(M / factory_period). Use 0 or 1 for one factory per
        module.

    controls_per_module : int
        The k of a k-MCT group. Peak occupancy is 2k-1, so k <= 6 for
        an 11-qubit module, and k = 6 additionally needs uncompute=1.

    module_capacity : int
        Usable logical qubits per module. 11 for the gross code.

    column_major : bool
        Fill each column before moving to the next, so a subtree and
        the factory serving it stay in the same column.

    uncompute : int
        1 releases module-internal ancilla slots once the module's
        output exists.

    Returns
    -------
    nx.Graph
        Mapped architecture with graph["qubit_to_module"] populated,
        ready for calculate_routing_volume.
    """

    if control_count <= 0:
        raise ValueError("control_count must be positive.")
    if controls_per_module <= 0:
        raise ValueError("controls_per_module must be positive.")
    if module_capacity <= 0:
        raise ValueError("module_capacity must be positive.")

    modules_required = math.ceil(control_count / controls_per_module)

    factory_count = (
        modules_required
        if factory_period <= 1
        else math.ceil(modules_required / factory_period)
    )
    factory_count = max(1, min(factory_count, modules_required))

    # ---- grid ----
    columns = max(1, min(round(math.sqrt(modules_required)), modules_required))
    rows = math.ceil(modules_required / columns)

    graph = nx.Graph()
    module_at = {}

    index = 0
    for row in range(rows):
        for col in range(columns):
            if index >= modules_required:
                break
            module = f"M{row}_{col}"
            graph.add_node(
                module,
                kind="module",
                index=index,
                row=row,
                col=col,
                placed_qubits=[],
            )
            module_at[(row, col)] = module
            index += 1

    for (row, col), module in module_at.items():
        if (row, col + 1) in module_at:
            graph.add_edge(module, module_at[(row, col + 1)], weight=1.0)
        if (row + 1, col) in module_at:
            graph.add_edge(module, module_at[(row + 1, col)], weight=1.0)

    all_modules = list(module_at.values())
    hop = dict(nx.all_pairs_shortest_path_length(graph.subgraph(all_modules)))

    # ---- factories ----
    sites = _kmedoid_sites(all_modules, hop, factory_count)

    for factory_index, module in enumerate(sites):
        factory = f"F{factory_index}"
        graph.add_node(
            factory,
            kind="factory",
            index=factory_index,
            attached_module=module,
        )
        graph.add_edge(factory, module, weight=1.0)

    factory_distance = {
        module: min(hop[module][s] for s in sites) for module in all_modules
    }

    # ---- fill order ----
    def position(module):
        data = graph.nodes[module]
        return (data["row"], data["col"])

    if column_major:
        fill_order = sorted(
            all_modules, key=lambda m: (position(m)[1], position(m)[0])
        )
    else:
        fill_order = sorted(
            all_modules, key=lambda m: (position(m)[0], position(m)[1])
        )

    # ---- tree structure ----
    original_controls = set(range(control_count))

    control_groups = [
        list(range(start, min(start + controls_per_module, control_count)))
        for start in range(0, control_count, controls_per_module)
    ]

    control_to_group = {
        control: group_index
        for group_index, group in enumerate(control_groups)
        for control in group
    }

    producer = {}
    for edge in toffoli_hyperedge_dict:
        if len(edge) != 3:
            raise ValueError(f"Expected a three-qubit hyperedge, got {edge!r}.")
        left_child, right_child, target = edge
        previous = producer.get(target)
        if previous is not None:
            if frozenset(previous) != frozenset((left_child, right_child)):
                raise ValueError(
                    f"Qubit {target} has multiple distinct producers."
                )
            continue
        producer[target] = (left_child, right_child)

    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(original_controls)
    for target, (left_child, right_child) in producer.items():
        dependency_graph.add_edge(left_child, target)
        dependency_graph.add_edge(right_child, target)

    if not nx.is_directed_acyclic_graph(dependency_graph):
        raise ValueError(
            "The supplied hyperedges do not form an acyclic binary tree."
        )

    topological_targets = [
        qubit
        for qubit in nx.topological_sort(dependency_graph)
        if qubit in producer
    ]

    @lru_cache(maxsize=None)
    def descendant_controls(qubit):
        if qubit in original_controls:
            return frozenset({qubit})
        if qubit not in producer:
            raise ValueError(f"Qubit {qubit} is neither control nor target.")
        left_child, right_child = producer[qubit]
        return descendant_controls(left_child) | descendant_controls(right_child)

    # ---- placement ----
    qubit_to_module = {}
    local_ancillas = {module: [] for module in all_modules}
    freed_slots = {module: 0 for module in all_modules}

    def effective_occupancy(module):
        return len(graph.nodes[module]["placed_qubits"]) - freed_slots[module]

    def place(module, qubits):
        placed = graph.nodes[module]["placed_qubits"]
        new_qubits = [q for q in qubits if q not in qubit_to_module]
        if effective_occupancy(module) + len(new_qubits) > module_capacity:
            raise RuntimeError(
                f"Placing {new_qubits} in {module} would exceed capacity "
                f"{module_capacity}."
            )
        placed.extend(new_qubits)
        for qubit in new_qubits:
            qubit_to_module[qubit] = module

    for group_index, controls in enumerate(control_groups):
        place(fill_order[group_index], controls)

    for target in topological_targets:
        group_ids = {
            control_to_group[c] for c in descendant_controls(target)
        }
        if len(group_ids) != 1:
            continue
        module = fill_order[next(iter(group_ids))]
        place(module, [target])
        local_ancillas[module].append(target)

    if uncompute == 1:

        def group_of(qubit):
            if qubit in control_to_group:
                return control_to_group[qubit]
            hit = {control_to_group[c] for c in descendant_controls(qubit)}
            return next(iter(hit)) if len(hit) == 1 else None

        must_stay_live = set()
        for target, (left_child, right_child) in producer.items():
            spans = {
                group_of(left_child),
                group_of(right_child),
                group_of(target),
            }
            if None in spans or len(spans) > 1:
                must_stay_live.update((left_child, right_child))

        for module in all_modules:
            freed_slots[module] = sum(
                1
                for ancilla in local_ancillas[module]
                if ancilla not in must_stay_live
            )

    for target in topological_targets:
        if target in qubit_to_module:
            continue

        left_child, right_child = producer[target]
        for child in (left_child, right_child):
            if child not in qubit_to_module:
                raise RuntimeError(f"Child {child} of {target} not placed.")

        left_module = qubit_to_module[left_child]
        right_module = qubit_to_module[right_child]

        candidates = [
            module
            for module in all_modules
            if effective_occupancy(module) < module_capacity
        ]
        if not candidates:
            raise RuntimeError(f"No module has space for {target}.")

        def score(module):
            return (
                hop[module][left_module] + hop[module][right_module],
                factory_distance[module],
                effective_occupancy(module),
                graph.nodes[module]["index"],
            )

        place(min(candidates, key=score), [target])

    distances = list(factory_distance.values())

    graph.graph["qubit_to_module"] = qubit_to_module
    graph.graph["control_groups"] = control_groups
    graph.graph["freed_slots"] = dict(freed_slots)
    graph.graph["rows"] = rows
    graph.graph["columns"] = columns
    graph.graph["required_modules"] = modules_required
    graph.graph["factory_count"] = len(sites)
    graph.graph["factory_modules"] = list(sites)
    graph.graph["factory_period"] = factory_period
    graph.graph["mean_factory_distance"] = sum(distances) / len(distances)
    graph.graph["max_factory_distance"] = max(distances)
    graph.graph["controls_per_module"] = controls_per_module
    graph.graph["placement_baseline"] = "square_grid_kmedoid"
    graph.graph["topology"] = "square_grid"

    return graph