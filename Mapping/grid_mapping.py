"""
grid_tree_baseline.py

Tree-aware placement for a Sethi-style long grid -- the grid
equivalent of linear_mapping_tree_baseline.

Three differences from the linear version, each forced by the
topology:

1. Stage 1 assigns control groups COLUMN-MAJOR, not row-major.
   On a long grid each column carries its own factory, so filling a
   column before moving to the next keeps a subtree and the factory it
   routes to inside one column. Row-major would spread consecutive
   groups across columns, so every subtree would straddle several
   factory territories.

2. Stage 3 scores candidates by TRUE GRAPH DISTANCE over the
   module-only subgraph, not by |index difference|. On a row-major
   indexed grid, M0_0 and M1_0 are adjacent but their indices differ
   by `columns`, so index arithmetic misreports distance badly.

3. The tiebreak is DISTANCE TO THE NEAREST FACTORY rather than
   distance to the architecture centre. On a grid with per-column
   factories, factory distance is the term that actually drives
   inter-module measurement count; the centre of the grid has no
   physical meaning.

Factories are read from the graph, so the same function works for
both "edge" and "distributed" factory placements.
"""

from __future__ import annotations

import math
from functools import lru_cache

import networkx as nx

from Architectures.arch import sethi_long_grid

def grid_tree_baseline(
    control_count,
    toffoli_hyperedge_dict,    
    controls_per_module=4,
    module_capacity=11,
    column_major=True,
    uncompute=0,
):
    """
    Place a binary-tree Toffoli decomposition onto a grid.

    Parameters
    ----------
    control_count : int
        Number of original MCT controls.

    toffoli_hyperedge_dict : dict
        Keys are (left_child, right_child, target) triples.

    architecture : nx.Graph
        A grid from sethi_long_grid() (or any graph following the same
        node-attribute contract). Modified in place and returned.

    controls_per_module : int
        Original controls assigned to each module in Stage 1.

    module_capacity : int
        Maximum simultaneous qubits per module.

    column_major : bool
        True (default) fills each column top to bottom before moving
        to the next, keeping subtrees inside a single column. False
        reverts to row-major for comparison.

    uncompute : int
        1 releases the slots of module-internal ancillas once the
        module's output exists. See linear_mapping_tree_baseline for
        the reasoning; the accounting here is identical.

    Returns
    -------
    nx.Graph
        The architecture with each module's "placed_qubits" filled in
        and graph["qubit_to_module"] populated.
    """

    if control_count <= 0:
        raise ValueError("control_count must be positive.")
    if controls_per_module <= 0:
        raise ValueError("controls_per_module must be positive.")
    if module_capacity <= 0:
        raise ValueError("module_capacity must be positive.")

    module_count = math.ceil(control_count / controls_per_module)
    factory_count = module_count//2

    architecture = sethi_long_grid(control_count=control_count,controls_per_module=controls_per_module,factory_count=factory_count)

    all_modules = [
        node
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "module"
    ]

    if len(all_modules) < module_count:
        raise ValueError(
            f"Grid has {len(all_modules)} modules but {module_count} are "
            "required. Increase the grid size or controls_per_module."
        )
    # ------------------------------------------------------------------
    # Stage-1 fill order.
    # ------------------------------------------------------------------
    def position(module):
        data = architecture.nodes[module]
        return (data.get("row", 0), data.get("col", 0))

    if column_major:
        module_nodes = sorted(
            all_modules, key=lambda m: (position(m)[1], position(m)[0])
        )
    else:
        module_nodes = sorted(
            all_modules, key=lambda m: (position(m)[0], position(m)[1])
        )
    # ------------------------------------------------------------------
    # Distances over MODULES ONLY: factories are leaves and must never
    # be used as routing waypoints.
    # ------------------------------------------------------------------
    module_subgraph = architecture.subgraph(all_modules)
    hop_distance = dict(nx.all_pairs_shortest_path_length(module_subgraph))

    factory_modules = [
        data["attached_module"]
        for _, data in architecture.nodes(data=True)
        if data.get("kind") == "factory"
    ]

    if not factory_modules:
        raise ValueError("The architecture has no factory nodes.")

    def distance(left, right):
        try:
            return hop_distance[left][right]
        except KeyError as exc:
            raise RuntimeError(
                f"No module-only path between {left!r} and {right!r}."
            ) from exc

    factory_distance = {
        module: min(distance(module, f) for f in factory_modules)
        for module in all_modules
    }

    # ------------------------------------------------------------------
    # Tree structure.
    # ------------------------------------------------------------------
    original_controls = set(range(control_count))

    control_groups = [
        list(range(start, min(start + controls_per_module, control_count)))
        for start in range(0, control_count, controls_per_module)
    ]

    control_to_group = {
        control: index
        for index, group in enumerate(control_groups)
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
            raise ValueError(
                f"Qubit {qubit} is neither a control nor a hyperedge target."
            )
        left_child, right_child = producer[qubit]
        return descendant_controls(left_child) | descendant_controls(right_child)

    # ------------------------------------------------------------------
    # Placement.
    # ------------------------------------------------------------------
    qubit_to_module = {}
    local_ancillas = {module: [] for module in all_modules}
    freed_slots = {module: 0 for module in all_modules}

    def effective_occupancy(module):
        return (
            len(architecture.nodes[module]["placed_qubits"])
            - freed_slots[module]
        )

    def place_qubits(module, qubits):
        placed = architecture.nodes[module]["placed_qubits"]
        new_qubits = [q for q in qubits if q not in qubit_to_module]

        if effective_occupancy(module) + len(new_qubits) > module_capacity:
            raise RuntimeError(
                f"Placing {new_qubits} in {module} would exceed capacity "
                f"{module_capacity}."
            )

        placed.extend(new_qubits)
        for qubit in new_qubits:
            qubit_to_module[qubit] = module

    # Stage 1: control groups, in fill order.
    for group_index, controls in enumerate(control_groups):
        place_qubits(module_nodes[group_index], controls)

    # Stage 2: fully local subtrees.
    for target in topological_targets:
        group_ids = {
            control_to_group[control]
            for control in descendant_controls(target)
        }
        if len(group_ids) != 1:
            continue

        module = module_nodes[next(iter(group_ids))]
        place_qubits(module, [target])
        local_ancillas[module].append(target)

    # Stage 3: higher-level outputs, scored by graph distance with a
    # factory-distance tiebreak.
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

        def placement_score(module):
            child_distance = (
                distance(module, left_module) + distance(module, right_module)
            )
            return (
                child_distance,
                factory_distance[module],
                effective_occupancy(module),
                architecture.nodes[module]["index"],
            )

        place_qubits(min(candidates, key=placement_score), [target])

    architecture.graph["qubit_to_module"] = qubit_to_module
    architecture.graph["control_groups"] = control_groups
    architecture.graph["placement_baseline"] = (
        "grid_tree_column_major" if column_major else "grid_tree_row_major"
    )
    architecture.graph["freed_slots"] = dict(freed_slots)

    return architecture

"""
joint_placement.py

Joint optimisation of FACTORY placement and QUBIT placement.

The two are coupled. Where the factories go determines the routing
cost of a given qubit placement; where the qubits go determines which
modules carry traffic and therefore where factories ought to be. Fixing
either one first gives a suboptimal answer to the other.

This module alternates between them, in the style of Lloyd's algorithm:

    1. Place qubits, given the current factory positions.
    2. Measure each module's DEMAND -- the frequency-weighted number of
       Toffolis touching it.
    3. Re-site the factories at demand-weighted k-medoids: choose f
       modules minimising sum over modules of demand x distance to the
       nearest chosen module.
    4. Repeat until the factory set stops moving.

Step 3 is the part that a fixed "distributed" arrangement misses.
Geometric spreading assumes uniform demand; real placements are not
uniform, because the tree's higher-level ancillas concentrate traffic
in whichever modules end up hosting them.

Both steps only reduce the objective, and the factory set is drawn
from a finite pool, so the loop terminates. It finds a local optimum,
not a global one -- k-medoids is NP-hard.

The floor still applies: with f factories and M modules, the furthest
module is about M/(2f) hops from one, whatever the arrangement. This
heuristic recovers waste against that bound; it does not beat it.
"""


import math

import networkx as nx


def module_demand(architecture, toffoli_hyperedge_dict):
    """
    Frequency-weighted count of Toffolis touching each module.

    A Toffoli whose three operands span several modules contributes its
    frequency to each of them, since each will have to be reached.
    """
    qubit_to_module = architecture.graph["qubit_to_module"]

    demand = {
        node: 0.0
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "module"
    }

    for hyperedge, frequency in toffoli_hyperedge_dict.items():
        touched = {
            qubit_to_module[qubit]
            for qubit in hyperedge
            if qubit in qubit_to_module
        }
        for module in touched:
            demand[module] += frequency

    return demand


def weighted_cost(candidate_sites, demand, hop):
    """Sum over modules of demand x distance to the nearest factory site."""
    total = 0.0
    for module, weight in demand.items():
        if weight == 0:
            continue
        total += weight * min(hop[module][site] for site in candidate_sites)
    return total


def choose_factory_sites(architecture, demand, factory_count, seed_sites=None):
    """
    Demand-weighted k-medoids over module nodes.

    Greedy seeding (each new site chosen to reduce the objective most)
    followed by swap-based local search. Returns the chosen modules.
    """
    modules = [
        node
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "module"
    ]

    hop = dict(
        nx.all_pairs_shortest_path_length(architecture.subgraph(modules))
    )

    factory_count = min(factory_count, len(modules))

    # --- seed ---
    if seed_sites:
        sites = list(seed_sites)[:factory_count]
    else:
        sites = []

    while len(sites) < factory_count:
        best_site, best_cost = None, math.inf
        for candidate in modules:
            if candidate in sites:
                continue
            cost = weighted_cost(sites + [candidate], demand, hop)
            if cost < best_cost:
                best_site, best_cost = candidate, cost
        sites.append(best_site)

    # --- swap-based local search ---
    current_cost = weighted_cost(sites, demand, hop)
    improved = True

    while improved:
        improved = False
        for index in range(len(sites)):
            for candidate in modules:
                if candidate in sites:
                    continue
                trial = list(sites)
                trial[index] = candidate
                cost = weighted_cost(trial, demand, hop)
                if cost < current_cost - 1e-12:
                    sites, current_cost = trial, cost
                    improved = True
                    break
            if improved:
                break

    return sites, current_cost


def attach_factories(architecture, sites):
    """Replace all factory nodes so they attach to `sites`."""
    for node in [
        n for n, d in list(architecture.nodes(data=True))
        if d.get("kind") == "factory"
    ]:
        architecture.remove_node(node)

    for index, module in enumerate(sites):
        factory = f"F{index}"
        architecture.add_node(
            factory,
            kind="factory",
            index=index,
            attached_module=module,
        )
        architecture.add_edge(factory, module, weight=1.0)

    return architecture


def joint_place(
    build_architecture,
    place_qubits,
    toffoli_hyperedge_dict,
    factory_count,
    max_rounds=5,
    verbose=False,
):
    """
    Alternate qubit placement and factory siting until the factory set
    stabilises.

    Parameters
    ----------
    build_architecture : callable() -> nx.Graph
        Returns a FRESH architecture with empty `placed_qubits`. Called
        once per round, since placement mutates the graph.

    place_qubits : callable(architecture) -> nx.Graph
        Runs a placement pass on the given architecture. It may consult
        the factory positions already attached to it.

    toffoli_hyperedge_dict : dict
        Used to compute module demand.

    factory_count : int
        Number of factories, f.

    max_rounds : int
        Cap on alternations. Convergence is usually 2-3 rounds.

    Returns
    -------
    (nx.Graph, list, list[float])
        The final placed architecture, the chosen factory modules, and
        the weighted cost after each round.
    """
    architecture = place_qubits(build_architecture())
    history = []
    previous_sites = None

    for round_index in range(max_rounds):
        demand = module_demand(architecture, toffoli_hyperedge_dict)
        sites, cost = choose_factory_sites(
            architecture, demand, factory_count, seed_sites=previous_sites
        )
        history.append(cost)

        if verbose:
            print(f"  round {round_index}: cost={cost:.1f} sites={sites}")

        if previous_sites is not None and set(sites) == set(previous_sites):
            break

        previous_sites = sites

        # Rebuild and re-place with the new factory positions, so the
        # placement pass can exploit them.
        fresh = attach_factories(build_architecture(), sites)
        architecture = place_qubits(fresh)

    architecture = attach_factories(architecture, previous_sites)
    return architecture, previous_sites, history