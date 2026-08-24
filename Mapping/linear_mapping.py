from Architectures.arch import linear_chain
from Applications.MCTdecomp.binary_tree_decomposition import extract_toffoli_hyperedges, create_circuit,create_parallel_k_grouped_mct
from qiskit import QuantumCircuit
import math
import networkx as nx
from itertools import combinations, product
from collections import defaultdict
from functools import lru_cache
from grid_mapping import grid_tree_baseline
from square_grid_mapping import square_grid_mapping
import math


def linear_mapping_baseline1(
    control_count,
    toffoli_hyperedge_dict,
    controls_per_module=4,
    module_capacity=11,
    factory_period=0,
):
    """
    Sequential capacity-only placement baseline.

    Placement policy
    ----------------
    1. Create ceil(control_count / controls_per_module) modules.

    2. Process Toffoli hyperedges in dictionary insertion order.

    3. Place all previously unplaced qubits from the current hyperedge
       into the current module.

    4. When the current module cannot fit those qubits, advance to the
       next module.

    5. Placement does not consider:
         - the MCT tree structure,
         - factory position,
         - communication distance,
         - routing volume.

    Parameters
    ----------
    control_count : int
        Number of original MCT controls.

    toffoli_hyperedge_dict : dict
        Hyperedge-frequency dictionary:

            {
                (control_1, control_2, target): frequency,
                ...
            }

        Qubits must use integer circuit indices.

    controls_per_module : int
        Used to determine the number of architecture modules:

            ceil(control_count / controls_per_module)

    module_capacity : int
        Maximum number of logical qubits placed in one module.

    factory_period : int
        Passed to linear_chain(). When zero, one factory is placed
        to the left of M0.

    Returns
    -------
    nx.Graph
        Linear architecture with placement stored in each module's
        "placed_qubits" attribute.
    """

    if control_count <= 0:
        raise ValueError("control_count must be positive.")

    if controls_per_module <= 0:
        raise ValueError(
            "controls_per_module must be positive."
        )

    if module_capacity <= 0:
        raise ValueError(
            "module_capacity must be positive."
        )

    if not toffoli_hyperedge_dict:
        raise ValueError(
            "No Toffoli hyperedges were provided."
        )

    module_count = math.ceil(
        control_count / controls_per_module
    )

    architecture = linear_chain(
        modules=module_count,
        factory_period=factory_period,
    )

    module_nodes = sorted(
        (
            node
            for node, data in architecture.nodes(data=True)
            if data.get("kind") == "module"
        ),
        key=lambda node: architecture.nodes[node]["index"],
    )

    # Dictionary order should correspond to circuit order.
    hyperedges = list(
        toffoli_hyperedge_dict.keys()
    )

    for hyperedge in hyperedges:
        if len(hyperedge) != 3:
            raise ValueError(
                f"Expected a three-qubit hyperedge, "
                f"received {hyperedge!r}."
            )

    # Each logical qubit has exactly one permanent module location.
    qubit_to_module = {}

    current_module_index = 0

    for hyperedge in hyperedges:

        # Qubits already placed by earlier hyperedges remain in their
        # original modules.
        new_qubits = [
            qubit
            for qubit in hyperedge
            if qubit not in qubit_to_module
        ]

        # This hyperedge introduces no new logical qubits.
        if not new_qubits:
            continue

        if len(new_qubits) > module_capacity:
            raise RuntimeError(
                f"Hyperedge {hyperedge} introduces "
                f"{len(new_qubits)} unplaced qubits, exceeding "
                f"the module capacity of {module_capacity}."
            )

        # Move sequentially along the line until the new qubits fit.
        while current_module_index < len(module_nodes):
            module = module_nodes[current_module_index]

            placed_qubits = architecture.nodes[module][
                "placed_qubits"
            ]

            available_capacity = (
                module_capacity - len(placed_qubits)
            )

            if len(new_qubits) <= available_capacity:
                break

            current_module_index += 1

        if current_module_index >= len(module_nodes):
            unplaced_hyperedges = hyperedges[
                hyperedges.index(hyperedge):
            ]

            raise RuntimeError(
                "The architecture does not have enough module "
                "capacity for the sequential placement.\n"
                f"Failed at hyperedge: {hyperedge}\n"
                f"Remaining hyperedges: {unplaced_hyperedges}"
            )

        selected_module = module_nodes[
            current_module_index
        ]

        architecture.nodes[selected_module][
            "placed_qubits"
        ].extend(new_qubits)

        for qubit in new_qubits:
            qubit_to_module[qubit] = selected_module

    architecture.graph["qubit_to_module"] = (
        qubit_to_module
    )
    architecture.graph["placement_baseline"] = (
        "sequential_capacity_only"
    )
    architecture.graph["controls_per_module"] = (
        controls_per_module
    )
    architecture.graph["module_capacity"] = (
        module_capacity
    )

    return architecture


# def linear_mapping_tree_baseline(
#     control_count,
#     toffoli_hyperedge_dict,
#     controls_per_module=4,
#     module_capacity=11,
#     factory_period=0,
#     uncompute = 0
# ):
#     """
#     Natural binary-tree placement baseline for any number of controls.

#     Placement policy
#     ----------------
#     1. Split the original controls into consecutive groups containing
#        at most `controls_per_module` controls.

#     2. Assign one module to each group.

#     3. Place all tree ancillas whose descendant controls are entirely
#        inside one group into that group's module.

#     4. Place higher-level outputs near the modules containing their
#        two child qubits.

#     Assumptions
#     -----------
#     - Hyperedges have the form:

#           (control_1, control_2, target)

#     - Qubits are represented by integer circuit indices.

#     - Original controls are indexed:

#           0, 1, ..., control_count - 1

#     - The Toffoli decomposition forms a binary computation tree.

#     Returns
#     -------
#     nx.Graph
#         The linear architecture with each module's placement stored
#         in its "placed_qubits" node attribute.
#     """

#     if control_count <= 0:
#         raise ValueError("control_count must be positive.")

#     if controls_per_module <= 0:
#         raise ValueError(
#             "controls_per_module must be positive."
#         )

#     if module_capacity <= 0:
#         raise ValueError("module_capacity must be positive.")

#     # Ceil allows the final module to contain fewer than four controls.
#     module_count = math.ceil(
#         control_count / controls_per_module
#     )

#     architecture = linear_chain(
#         modules=module_count,
#         factory_period=factory_period,
#     )

#     module_nodes = sorted(
#         (
#             node
#             for node, data in architecture.nodes(data=True)
#             if data.get("kind") == "module"
#         ),
#         key=lambda node: architecture.nodes[node]["index"],
#     )

#     original_controls = set(range(control_count))

#     # ---------------------------------------------------------
#     # Divide the controls into consecutive groups of up to four.
#     # ---------------------------------------------------------

#     control_groups = []

#     for group_index in range(module_count):
#         start = group_index * controls_per_module
#         stop = min(
#             start + controls_per_module,
#             control_count,
#         )

#         control_groups.append(
#             list(range(start, stop))
#         )

#     control_to_group = {
#         control: group_index
#         for group_index, group in enumerate(control_groups)
#         for control in group
#     }

#     # ---------------------------------------------------------
#     # Build the logical binary-tree producer relationship.
#     #
#     # producer[target] = (left_child, right_child)
#     # ---------------------------------------------------------

#     producer = {}

#     for edge in toffoli_hyperedge_dict:
#         if len(edge) != 3:
#             raise ValueError(
#                 f"Expected a three-qubit hyperedge, got {edge!r}."
#             )

#         left_child, right_child, target = edge

#         previous = producer.get(target)

#         if previous is not None:
#             previous_children = frozenset(previous)
#             current_children = frozenset(
#                 (left_child, right_child)
#             )

#             if previous_children != current_children:
#                 raise ValueError(
#                     f"Qubit {target} has multiple distinct "
#                     f"producers: {previous} and "
#                     f"{(left_child, right_child)}."
#                 )

#             continue

#         producer[target] = (
#             left_child,
#             right_child,
#         )

#     # ---------------------------------------------------------
#     # Build a dependency DAG to guarantee children are processed
#     # before their parent output.
#     # ---------------------------------------------------------

#     dependency_graph = nx.DiGraph()

#     dependency_graph.add_nodes_from(original_controls)

#     for target, children in producer.items():
#         left_child, right_child = children

#         dependency_graph.add_edge(
#             left_child,
#             target,
#         )
#         dependency_graph.add_edge(
#             right_child,
#             target,
#         )

#     if not nx.is_directed_acyclic_graph(dependency_graph):
#         raise ValueError(
#             "The supplied hyperedges do not form an acyclic "
#             "binary-tree decomposition."
#         )

#     topological_targets = [
#         qubit
#         for qubit in nx.topological_sort(dependency_graph)
#         if qubit in producer
#     ]

#     # ---------------------------------------------------------
#     # Determine which original controls lie below each tree node.
#     # ---------------------------------------------------------

#     @lru_cache(maxsize=None)
#     def descendant_controls(qubit):
#         if qubit in original_controls:
#             return frozenset({qubit})

#         if qubit not in producer:
#             raise ValueError(
#                 f"Qubit {qubit} appears as a tree input but is "
#                 "neither an original control nor the target of "
#                 "another hyperedge."
#             )

#         left_child, right_child = producer[qubit]

#         return (
#             descendant_controls(left_child)
#             | descendant_controls(right_child)
#         )

#     # ---------------------------------------------------------
#     # Placement helper.
#     # ---------------------------------------------------------

#     qubit_to_module = {}

#     def place_qubits(module, qubits):
#         placed_qubits = architecture.nodes[module][
#             "placed_qubits"
#         ]

#         new_qubits = [
#             qubit
#             for qubit in qubits
#             if qubit not in qubit_to_module
#         ]

#         required_occupancy = (
#             len(placed_qubits) + len(new_qubits)
#         )

#         if required_occupancy > module_capacity:
#             raise RuntimeError(
#                 f"Placing {new_qubits} in {module} would produce "
#                 f"occupancy {required_occupancy}, but the module "
#                 f"capacity is {module_capacity}."
#             )

#         placed_qubits.extend(new_qubits)

#         for qubit in new_qubits:
#             qubit_to_module[qubit] = module

#     # ---------------------------------------------------------
#     # Stage 1: place the original control groups.
#     # ---------------------------------------------------------

#     for group_index, controls in enumerate(control_groups):
#         module = module_nodes[group_index]

#         place_qubits(
#             module=module,
#             qubits=controls,
#         )

#     # ---------------------------------------------------------
#     # Stage 2: place every completely local subtree.
#     #
#     # A tree output is local when all of its original descendant
#     # controls belong to the same four-control group.
#     # ---------------------------------------------------------

#     for target in topological_targets:
#         leaves = descendant_controls(target)

#         group_ids = {
#             control_to_group[control]
#             for control in leaves
#         }

#         if len(group_ids) != 1:
#             continue

#         group_index = next(iter(group_ids))
#         module = module_nodes[group_index]

#         place_qubits(
#             module=module,
#             qubits=[target],
#         )
#     # uncompute = 1
#     # ---------------------------------------------------------
#     # Stage 3: place higher-level outputs.
#     #
#     # Factory locations are deliberately not used in the score.
#     # This keeps this baseline tree-aware but not factory-aware.
#     # ---------------------------------------------------------

#     architecture_center = (module_count - 1) / 2

#     def module_index(module):
#         return architecture.nodes[module]["index"]

#     if uncompute == 1:
#         module_capacity += len(architecture.nodes[module]["placed_qubits"])
            

#     for target in topological_targets:
#         if target in qubit_to_module:
#             continue

#         left_child, right_child = producer[target]

#         if left_child not in qubit_to_module:
#             raise RuntimeError(
#                 f"Child {left_child} of target {target} has not "
#                 "been placed."
#             )

#         if right_child not in qubit_to_module:
#             raise RuntimeError(
#                 f"Child {right_child} of target {target} has not "
#                 "been placed."
#             )

#         left_module = qubit_to_module[left_child]
#         right_module = qubit_to_module[right_child]

#         left_index = module_index(left_module)
#         right_index = module_index(right_module)

#         candidate_modules = [
#             module
#             for module in module_nodes
#             if len(
#                 architecture.nodes[module]["placed_qubits"]
#             ) < module_capacity
#         ]

#         if not candidate_modules:
#             raise RuntimeError(
#                 f"No module has space for higher-level output "
#                 f"{target}."
#             )

#         def placement_score(module):
#             candidate_index = module_index(module)

#             child_distance = (
#                 abs(candidate_index - left_index)
#                 + abs(candidate_index - right_index)
#             )

#             center_distance = abs(
#                 candidate_index - architecture_center
#             )

#             current_occupancy = len(
#                 architecture.nodes[module]["placed_qubits"]
#             )

#             return (
#                 child_distance,
#                 center_distance,
#                 current_occupancy,
#                 candidate_index,
#             )

#         selected_module = min(
#             candidate_modules,
#             key=placement_score,
#         )

#         place_qubits(
#             module=selected_module,
#             qubits=[target],
#         )

#     # Save useful placement metadata.
#     architecture.graph["qubit_to_module"] = qubit_to_module
#     architecture.graph["control_groups"] = control_groups
#     architecture.graph["placement_baseline"] = (
#         "natural_tree"
#     )

#     return architecture

def linear_mapping_tree_baseline(
    control_count,
    toffoli_hyperedge_dict,
    controls_per_module=4,
    module_capacity=11,
    factory_period=0,
    uncompute=0,
):
    """
    Natural binary-tree placement baseline for any number of controls.
 
    Placement policy
    ----------------
    1. Split the original controls into consecutive groups containing
       at most `controls_per_module` controls.
 
    2. Assign one module to each group.
 
    3. Place all tree ancillas whose descendant controls are entirely
       inside one group into that group's module.
 
    4. Place higher-level outputs near the modules containing their
       two child qubits.
 
    uncompute : int
        0 (default):
            Standard LIFO schedule. Every qubit placed in Stages 1-2
            occupies its slot for the whole computation.
 
        1:
            Early uncomputation. Once a module's own subtree output
            exists, that module's INTERNAL ancillas are uncomputed
            top-down and their slots released for Stage-3 placement.
 
            Freed per module = (Stage-2 ancillas in that module)
                               - (that module's output)
 
            The module output is NOT freed: it is consumed by a
            cross-module hyperedge and must stay live. The original
            controls are never freed either -- they are program
            qubits, and keeping them live is exactly what makes this
            uncomputation recomputation-free (a node can be
            uncomputed while its children hold their values; here the
            children are controls).
 
            Peak occupancy during the module's own subtree
            computation is unchanged at 2k-1, so the k <= 6 bound for
            an 11-qubit module still holds. What changes is the
            occupancy AFTERWARDS, which is when Stage 3 runs.
 
    Returns
    -------
    nx.Graph
        The linear architecture with each module's placement stored
        in its "placed_qubits" node attribute.
    """
 
    if control_count <= 0:
        raise ValueError("control_count must be positive.")
 
    if controls_per_module <= 0:
        raise ValueError(
            "controls_per_module must be positive."
        )
 
    if module_capacity <= 0:
        raise ValueError("module_capacity must be positive.")
 
    # Ceil allows the final module to contain fewer than four controls.
    module_count = math.ceil(
        control_count / controls_per_module
    )
 
    architecture = linear_chain(
        modules=module_count,
        factory_period=factory_period,
    )
 
    module_nodes = sorted(
        (
            node
            for node, data in architecture.nodes(data=True)
            if data.get("kind") == "module"
        ),
        key=lambda node: architecture.nodes[node]["index"],
    )
 
    original_controls = set(range(control_count))
 
    # ---------------------------------------------------------
    # Divide the controls into consecutive groups of up to four.
    # ---------------------------------------------------------
 
    control_groups = []
 
    for group_index in range(module_count):
        start = group_index * controls_per_module
        stop = min(
            start + controls_per_module,
            control_count,
        )
 
        control_groups.append(
            list(range(start, stop))
        )
 
    control_to_group = {
        control: group_index
        for group_index, group in enumerate(control_groups)
        for control in group
    }
 
    # ---------------------------------------------------------
    # Build the logical binary-tree producer relationship.
    #
    # producer[target] = (left_child, right_child)
    # ---------------------------------------------------------
 
    producer = {}
 
    for edge in toffoli_hyperedge_dict:
        if len(edge) != 3:
            raise ValueError(
                f"Expected a three-qubit hyperedge, got {edge!r}."
            )
 
        left_child, right_child, target = edge
 
        previous = producer.get(target)
 
        if previous is not None:
            previous_children = frozenset(previous)
            current_children = frozenset(
                (left_child, right_child)
            )
 
            if previous_children != current_children:
                raise ValueError(
                    f"Qubit {target} has multiple distinct "
                    f"producers: {previous} and "
                    f"{(left_child, right_child)}."
                )
 
            continue
 
        producer[target] = (
            left_child,
            right_child,
        )
 
    # ---------------------------------------------------------
    # Build a dependency DAG to guarantee children are processed
    # before their parent output.
    # ---------------------------------------------------------
 
    dependency_graph = nx.DiGraph()
 
    dependency_graph.add_nodes_from(original_controls)
 
    for target, children in producer.items():
        left_child, right_child = children
 
        dependency_graph.add_edge(
            left_child,
            target,
        )
        dependency_graph.add_edge(
            right_child,
            target,
        )
 
    if not nx.is_directed_acyclic_graph(dependency_graph):
        raise ValueError(
            "The supplied hyperedges do not form an acyclic "
            "binary-tree decomposition."
        )
 
    topological_targets = [
        qubit
        for qubit in nx.topological_sort(dependency_graph)
        if qubit in producer
    ]
 
    # ---------------------------------------------------------
    # Determine which original controls lie below each tree node.
    # ---------------------------------------------------------
 
    @lru_cache(maxsize=None)
    def descendant_controls(qubit):
        if qubit in original_controls:
            return frozenset({qubit})
 
        if qubit not in producer:
            raise ValueError(
                f"Qubit {qubit} appears as a tree input but is "
                "neither an original control nor the target of "
                "another hyperedge."
            )
 
        left_child, right_child = producer[qubit]
 
        return (
            descendant_controls(left_child)
            | descendant_controls(right_child)
        )
 
    # ---------------------------------------------------------
    # Placement helper.
    # ---------------------------------------------------------
 
    qubit_to_module = {}
 
    # Stage-2 ancillas per module, recorded so that the freeable set
    # can be worked out once Stage 2 finishes.
    local_ancillas = {module: [] for module in module_nodes}
 
    # Slots released per module by early uncomputation.
    freed_slots = {module: 0 for module in module_nodes}
 
    def effective_occupancy(module):
        """Occupancy Stage 3 sees, after any early uncomputation."""
        return (
            len(architecture.nodes[module]["placed_qubits"])
            - freed_slots[module]
        )
 
    def place_qubits(module, qubits):
        placed_qubits = architecture.nodes[module][
            "placed_qubits"
        ]
 
        new_qubits = [
            qubit
            for qubit in qubits
            if qubit not in qubit_to_module
        ]
 
        required_occupancy = (
            effective_occupancy(module) + len(new_qubits)
        )
 
        if required_occupancy > module_capacity:
            raise RuntimeError(
                f"Placing {new_qubits} in {module} would produce "
                f"occupancy {required_occupancy}, but the module "
                f"capacity is {module_capacity}."
            )
 
        placed_qubits.extend(new_qubits)
 
        for qubit in new_qubits:
            qubit_to_module[qubit] = module
 
    # ---------------------------------------------------------
    # Stage 1: place the original control groups.
    # ---------------------------------------------------------
 
    for group_index, controls in enumerate(control_groups):
        module = module_nodes[group_index]
 
        place_qubits(
            module=module,
            qubits=controls,
        )
 
    # ---------------------------------------------------------
    # Stage 2: place every completely local subtree.
    #
    # A tree output is local when all of its original descendant
    # controls belong to the same four-control group.
    # ---------------------------------------------------------
 
    for target in topological_targets:
        leaves = descendant_controls(target)
 
        group_ids = {
            control_to_group[control]
            for control in leaves
        }
 
        if len(group_ids) != 1:
            continue
 
        group_index = next(iter(group_ids))
        module = module_nodes[group_index]
 
        place_qubits(
            module=module,
            qubits=[target],
        )
 
        local_ancillas[module].append(target)
 
    # ---------------------------------------------------------
    # Early uncomputation: release the slots of module-internal
    # ancillas.
    #
    # A Stage-2 ancilla is the module's OUTPUT if some other target
    # consumes it as a child; that one must stay live. Everything
    # else in the module is internal and can be uncomputed top-down
    # as soon as the output exists.
    # ---------------------------------------------------------
 
    if uncompute == 1:
        consumed_as_child = set()
        for left_child, right_child in producer.values():
            consumed_as_child.add(left_child)
            consumed_as_child.add(right_child)
 
        for module in module_nodes:
            internal = [
                ancilla
                for ancilla in local_ancillas[module]
                if ancilla not in consumed_as_child
                or all(
                    qubit_to_module.get(parent) == module
                    for parent, children in producer.items()
                    if ancilla in children
                )
            ]
 
            # Never free the module's output: it is consumed by a
            # target that lives elsewhere.
            outputs = [
                ancilla
                for ancilla in local_ancillas[module]
                if ancilla not in internal
            ]
 
            # Guard: a module with no output has nothing downstream,
            # so its whole local subtree except the root is freeable.
            if not outputs and local_ancillas[module]:
                internal = local_ancillas[module][:-1]
 
            freed_slots[module] = len(internal)
 
    # ---------------------------------------------------------
    # Stage 3: place higher-level outputs.
    #
    # Factory locations are deliberately not used in the score.
    # This keeps this baseline tree-aware but not factory-aware.
    # ---------------------------------------------------------
 
    architecture_center = (module_count - 1) / 2
 
    def module_index(module):
        return architecture.nodes[module]["index"]
 
    for target in topological_targets:
        if target in qubit_to_module:
            continue
 
        left_child, right_child = producer[target]
 
        if left_child not in qubit_to_module:
            raise RuntimeError(
                f"Child {left_child} of target {target} has not "
                "been placed."
            )
 
        if right_child not in qubit_to_module:
            raise RuntimeError(
                f"Child {right_child} of target {target} has not "
                "been placed."
            )
 
        left_module = qubit_to_module[left_child]
        right_module = qubit_to_module[right_child]
 
        left_index = module_index(left_module)
        right_index = module_index(right_module)
 
        candidate_modules = [
            module
            for module in module_nodes
            if effective_occupancy(module) < module_capacity
        ]
 
        if not candidate_modules:
            raise RuntimeError(
                f"No module has space for higher-level output "
                f"{target}."
            )
 
        def placement_score(module):
            candidate_index = module_index(module)
 
            child_distance = (
                abs(candidate_index - left_index)
                + abs(candidate_index - right_index)
            )
 
            center_distance = abs(
                candidate_index - architecture_center
            )
 
            current_occupancy = effective_occupancy(module)
 
            return (
                child_distance,
                center_distance,
                current_occupancy,
                candidate_index,
            )
 
        selected_module = min(
            candidate_modules,
            key=placement_score,
        )
 
        place_qubits(
            module=selected_module,
            qubits=[target],
        )
 
    # Save useful placement metadata.
    architecture.graph["qubit_to_module"] = qubit_to_module
    architecture.graph["control_groups"] = control_groups
    architecture.graph["placement_baseline"] = (
        "natural_tree_uncompute" if uncompute == 1 else "natural_tree"
    )
    architecture.graph["freed_slots"] = dict(freed_slots)
 
    return architecture
 


import math
import networkx as nx
from collections import defaultdict
from itertools import combinations, product


def calculate_routing_volume(
    architecture_graph,
    toffoli_hyperedge_dict,
    weight="weight",
    routing_model="ccz_paths",
):
    """
    Calculate frequency-weighted routing volume.

    For each Toffoli hyperedge:
      1. Find the modules containing its qubits.
      2. Consider every factory in the architecture graph.
      3. Compute the routing cost under `routing_model`.
      4. Select the minimum-cost factory.
      5. Multiply the cost by the hyperedge frequency.

    Routing models
    --------------
    "ccz_paths" (default)
        Cost = sum of the factory-to-operand shortest-path distances,
        counted SEPARATELY for each of the three operands.

        This matches CCZ-injection compilation. Consuming a |CCZ>
        state is three independent Pauli product measurements --
        Z(c1)(x)Z(r0), Z(c2)(x)Z(r1), X(t)(x)Z(r2) -- each touching
        one data module and the magic module. Each gets its own GHZ,
        prepared and uncomputed separately, so no corridor is shared
        between them and the three paths must be added, not merged.

        Operands sharing a module are still counted separately: the
        three measurements are distinct operations regardless of
        where their data sits. 

    "mst"
        Cost = minimum spanning tree over {operand modules} u
        {factory} in the shortest-path metric.

        This matches a single WIDE Pauli product rotation whose
        support spans several modules at once -- the per-T-gate
        Litinski route -- where one GHZ covers all participating
        modules and shared corridor segments are paid once. It is the
        formulation used by Tour de Gross Sec. 3.4 and Sethi et al.
        Sec. III-C2.

        MST cost <= sum-of-paths cost always, so this model
        systematically UNDERCOUNTS CCZ injection.

    Neither model is an error prediction: both count inter-module
    hops and charge module-local operations their full
    distance-to-factory, whereas the circuit-level noise model makes
    in-module measurements ~181x cheaper than inter-module ones at
    p = 1e-3. Use routing volume as a topology metric and report
    instruction counts separately for error claims.

    Factory nodes are identified by:

        kind="factory"

    Module nodes are identified by:

        kind="module"

    Parameters
    ----------
    architecture_graph : nx.Graph
        Architecture produced by linear_chain().

    toffoli_hyperedge_dict : dict
        Mapping from a Toffoli hyperedge to its frequency:

            {
                (q0, q1, q2): 4,
                (q1, q3, q4): 2,
            }

    weight : str
        Graph-edge attribute used as routing cost.
        If the attribute is absent, each edge has cost 1.

    routing_model : str
        "ccz_paths" or "mst". See above.

    Returns
    -------
    float
        Total frequency-weighted routing volume.
    """

    if routing_model not in ("ccz_paths", "mst"):
        raise ValueError(
            "routing_model must be 'ccz_paths' or 'mst', got "
            f"{routing_model!r}."
        )

    factory_nodes = [
        node
        for node, data in architecture_graph.nodes(data=True)
        if data.get("kind") == "factory"
    ]

    if not factory_nodes:
        raise ValueError(
            "The architecture graph does not contain a factory node."
        )

    # A qubit may appear in more than one module in your current baseline.
    qubit_locations = defaultdict(list)

    for node, data in architecture_graph.nodes(data=True):
        if data.get("kind") != "module":
            continue

        for qubit in data.get("placed_qubits", []):
            qubit_locations[qubit].append(node)

    # Precompute all physical shortest-path distances.
    shortest_distances = dict(
        nx.all_pairs_dijkstra_path_length(
            architecture_graph,
            weight=weight,
        )
    )

    mst_cost_cache = {}

    def calculate_terminal_mst_cost(terminals):
        """
        Construct the terminal metric graph and return its MST cost.
        """

        # A frozenset removes repeated terminal locations.
        terminal_set = frozenset(terminals)

        if len(terminal_set) <= 1:
            return 0

        if terminal_set in mst_cost_cache:
            return mst_cost_cache[terminal_set]

        terminals = list(terminal_set)
        metric_graph = nx.Graph()
        metric_graph.add_nodes_from(terminals)

        for source, destination in combinations(terminals, 2):
            try:
                distance = shortest_distances[source][destination]
            except KeyError as error:
                raise nx.NetworkXNoPath(
                    f"No route exists between {source} and {destination}."
                ) from error

            metric_graph.add_edge(
                source,
                destination,
                weight=distance,
            )

        mst = nx.minimum_spanning_tree(
            metric_graph,
            weight="weight",
        )

        cost = mst.size(weight="weight")
        mst_cost_cache[terminal_set] = cost

        return cost

    def calculate_ccz_path_cost(qubit_modules, factory):
        """
        Sum of the factory-to-operand distances, one per Pauli product
        measurement. Duplicates are NOT collapsed: the three
        measurements are separate operations.
        """
        total = 0

        for module in qubit_modules:
            try:
                total += shortest_distances[factory][module]
            except KeyError as error:
                raise nx.NetworkXNoPath(
                    f"No route exists between {factory} and {module}."
                ) from error

        return total

    total_routing_volume = 0

    for hyperedge, frequency in toffoli_hyperedge_dict.items():
        location_options = []

        for qubit in hyperedge:
            locations = qubit_locations.get(qubit, [])

            if not locations:
                raise ValueError(
                    f"Qubit {qubit!r} in hyperedge {hyperedge!r} "
                    "has not been placed in any module."
                )

            location_options.append(locations)

        best_cost = math.inf

        # Try all possible locations if a qubit appears in multiple modules.
        for qubit_modules in product(*location_options):
            for factory in factory_nodes:
                if routing_model == "mst":
                    terminals = (*qubit_modules, factory)
                    cost = calculate_terminal_mst_cost(terminals)
                else:
                    cost = calculate_ccz_path_cost(
                        qubit_modules,
                        factory,
                    )

                best_cost = min(best_cost, cost)

        if math.isinf(best_cost):
            raise RuntimeError(
                f"Could not route hyperedge {hyperedge!r}."
            )

        total_routing_volume += frequency * best_cost

    return total_routing_volume


import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
 
from figplots import plot_routing_volume


control_size = 128
rv1 = []
rv2 = []
rv3 = []
rv4 = []
qc1 = create_circuit(48)
toffoli_hyperedges,_ = extract_toffoli_hyperedges(qc1)
from figplots import draw_architecture,draw_square_grid
mapped_graph3 = linear_mapping_tree_baseline(
    control_count=48,
    toffoli_hyperedge_dict=toffoli_hyperedges,
    controls_per_module=4,
    module_capacity=11,
    factory_period=4,
    uncompute = 0
)
draw_architecture(mapped_graph3)


mapped_graph4 = square_grid_mapping(
    control_count=48,
    toffoli_hyperedge_dict=toffoli_hyperedges,
    factory_period=4,
    controls_per_module=4,
)
draw_square_grid(mapped_graph4)
sum_k3 = 0
sum_k4 = 0
sum_k5 = 0
for num_controls in range(5,control_size,5):
    qc1 = create_circuit(num_controls)
    qc2 = create_parallel_k_grouped_mct(num_controls,6)

    toffoli_hyperedges,_ = extract_toffoli_hyperedges(qc1)
    toffoli_hyperedges_k,_ = extract_toffoli_hyperedges(qc2)

    fp = 4
    mapped_graph1 = linear_mapping_baseline1(
        control_count=num_controls,
        toffoli_hyperedge_dict=toffoli_hyperedges,
        controls_per_module=4,
        module_capacity=11,
        factory_period=fp
    )
   
    mapped_graph2 = linear_mapping_tree_baseline(
        control_count=num_controls,
        toffoli_hyperedge_dict=toffoli_hyperedges,
        controls_per_module=3,
        module_capacity=11,
        factory_period=fp,
        uncompute = 0
    )
    
    mapped_graph3 = linear_mapping_tree_baseline(
        control_count=num_controls,
        toffoli_hyperedge_dict=toffoli_hyperedges,
        controls_per_module=4,
        module_capacity=11,
        factory_period=fp,
        uncompute = 0
    )

    mapped_graph4 = linear_mapping_tree_baseline(
        control_count=num_controls,
        toffoli_hyperedge_dict=toffoli_hyperedges,
        controls_per_module=5,
        module_capacity=11,
        factory_period=fp
        )

    # mapped_graph4 = square_grid_mapping(
    #     control_count=num_controls,
    #     toffoli_hyperedge_dict=toffoli_hyperedges,
    #     factory_period=4,
    #     controls_per_module=4,
    # )
   
    # break
    # mapped_graph4 = grid_tree_baseline(control_count=num_controls,
    #                                    toffoli_hyperedge_dict=toffoli_hyperedges,
    #                                    controls_per_module=5,
    #                                    )
  
    
    routing_volume = calculate_routing_volume(
        architecture_graph=mapped_graph1,
        toffoli_hyperedge_dict=toffoli_hyperedges,
    )
    # print("Routing volume:", routing_volume)      
    rv1.append(routing_volume)
    routing_volume = calculate_routing_volume(
        architecture_graph=mapped_graph2,
        toffoli_hyperedge_dict=toffoli_hyperedges,
    )

    # print("Routing volume:", routing_volume)      
    rv2.append(routing_volume)

    routing_volume = calculate_routing_volume(
        architecture_graph=mapped_graph3,
        toffoli_hyperedge_dict=toffoli_hyperedges,
    )
    rv3.append(routing_volume)

    routing_volume = calculate_routing_volume(
            architecture_graph=mapped_graph4,
            toffoli_hyperedge_dict=toffoli_hyperedges,
        )
    rv4.append(routing_volume)
    # print(num_controls)
    # print("Routing volume:", routing_volume)      
def improvement_vs_baseline(baseline,improvement):
    sum = 0
    for i in range(len(baseline)):
        sum += (baseline[i] - improvement[i]) / baseline[i]

    return sum / len(baseline)


print(improvement_vs_baseline(rv1,rv2))
print(improvement_vs_baseline(rv1,rv3))
print(improvement_vs_baseline(rv1,rv4))
plot_routing_volume(
    list(range(5,control_size,5)),
    rv1,
    rv2,
    rv3,
    rv4,
    save_path=f"routing_volume_comparison_period_{fp}.pdf",
)


