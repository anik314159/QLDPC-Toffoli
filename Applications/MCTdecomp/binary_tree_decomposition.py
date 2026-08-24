from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag,dag_to_circuit
from qiskit.visualization import dag_drawer
from collections import Counter



 

def create_circuit(n):
    qc = QuantumCircuit(2*n -1)
    anc = n
    ctr = 0
    for i in range(0,n,2):
        qc.ccx(i,i+1,anc)
        anc+=1
        ctr = i

    ctr += 2
    while(anc != 2*n - 1):
        qc.ccx(ctr,ctr+1,anc)
        ctr += 2
        anc += 1
    qc_partial = QuantumCircuit(2*n -1)
    qc_total = QuantumCircuit(2*n -1)
    for inst in qc.data[:-1]:      # omit last gate
        qc_partial.append(inst.operation, inst.qubits, inst.clbits)

    qc_total.compose(qc, qubits=list(range(2*n-1)), inplace= True)
    
    qc_total.compose(qc_partial.inverse(), qubits=list(range(2*n-1)), inplace= True)
    return qc_total

def create_k_grouped_mct(n: int, k: int) -> QuantumCircuit:
    """
    Construct an n-controlled X operation using a k-grouped,
    level-by-level binary Toffoli tree.

    Qubit layout
    ------------
    controls:
        0, 1, ..., n - 1

    clean workspace ancillas:
        n, n + 1, ..., 2*n - 3

    final target:
        2*n - 2

    Total qubits:
        2*n - 1

    The circuit uses:
        n - 2 clean workspace ancillas
        2*n - 3 Toffoli gates

    Parameters
    ----------
    n:
        Number of controls. Must be at least 2.

    k:
        Maximum number of consecutive controls in each initial group.

    Returns
    -------
    QuantumCircuit
        The complete compute-target-uncompute MCT circuit.
    """

    if n < 2:
        raise ValueError("n must be at least 2.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    number_of_qubits = 2 * n - 1
    target = 2 * n - 2

    qc = QuantumCircuit(
        number_of_qubits,
        name=f"{n}_control_k{k}_grouped_mct",
    )

    # Next available workspace ancilla.
    next_ancilla = n

    # Store compute gates level by level so that they can later
    # be uncomputed in reverse level order.
    compute_levels: list[list[tuple[int, int, int]]] = []

    # ----------------------------------------------------------
    # Initial ordered k-grouping.
    # ----------------------------------------------------------
    groups = [
        list(range(start, min(start + k, n)))
        for start in range(0, n, k)
    ]

    # ----------------------------------------------------------
    # Stage 1: reduce every group independently.
    # ----------------------------------------------------------
    while any(len(group) > 1 for group in groups):
        level: list[tuple[int, int, int]] = []
        next_groups: list[list[int]] = []

        for group in groups:
            next_group: list[int] = []
            index = 0

            while index + 1 < len(group):
                left = group[index]
                right = group[index + 1]

                if next_ancilla >= target:
                    raise RuntimeError(
                        "Ancilla allocation reached the final target "
                        "before the reduction tree was complete."
                    )

                output = next_ancilla
                next_ancilla += 1

                level.append((left, right, output))
                next_group.append(output)

                index += 2

            # Carry an unmatched node to the next level.
            if index < len(group):
                next_group.append(group[index])

            next_groups.append(next_group)

        if level:
            compute_levels.append(level)

        groups = next_groups

    # One root remains from each initial group.
    roots = [group[0] for group in groups]

    # ----------------------------------------------------------
    # Stage 2: combine the group roots.
    #
    # Stop when two roots remain. Their final conjunction is
    # written directly onto the actual target qubit.
    # ----------------------------------------------------------
    while len(roots) > 2:
        level: list[tuple[int, int, int]] = []
        next_roots: list[int] = []

        index = 0

        while index + 1 < len(roots):
            left = roots[index]
            right = roots[index + 1]

            if next_ancilla >= target:
                raise RuntimeError(
                    "Ancilla allocation reached the final target "
                    "before the reduction tree was complete."
                )

            output = next_ancilla
            next_ancilla += 1

            level.append((left, right, output))
            next_roots.append(output)

            index += 2

        if index < len(roots):
            next_roots.append(roots[index])

        if level:
            compute_levels.append(level)

        roots = next_roots

    # Exactly two roots must remain for the target Toffoli.
    if len(roots) != 2:
        raise RuntimeError(
            f"Expected two final roots, but found {len(roots)}."
        )

    # ----------------------------------------------------------
    # Emit the compute tree level by level.
    # ----------------------------------------------------------
    for level in compute_levels:
        for control_1, control_2, output in level:
            qc.ccx(control_1, control_2, output)

        qc.barrier()

    # ----------------------------------------------------------
    # Apply the final Toffoli directly to the MCT target.
    # ----------------------------------------------------------
    qc.ccx(roots[0], roots[1], target)
    qc.barrier()

    # ----------------------------------------------------------
    # Uncompute all workspace ancillas.
    # ----------------------------------------------------------
    for level in reversed(compute_levels):
        for control_1, control_2, output in reversed(level):
            qc.ccx(control_1, control_2, output)

        qc.barrier()

    return qc

from math import ceil, log2
from qiskit import QuantumCircuit



from qiskit import QuantumCircuit


from qiskit import QuantumCircuit


def create_k_grouped_mct(
    n: int,
    k: int = 6,
    insert_barriers: bool = True,
) -> QuantumCircuit:
    """
    Build an n-controlled X using ordered groups of at most k controls.

    Behaviour
    ---------
    1. Controls are divided into consecutive groups of size at most k.
    2. All groups are reduced in parallel, level by level.
    3. The final root ancilla of each group remains dirty.
    4. All lower-level group ancillas are immediately uncomputed.
    5. The group roots are combined through an upper binary tree.
    6. The final target operation is applied.
    7. Only the upper-tree scratch ancillas are uncomputed.

    Final ancilla state
    -------------------
    Dirty:
        One persistent root ancilla for every non-singleton k-group.

    Clean:
        Every lower-level group scratch ancilla.
        Every upper-tree scratch ancilla.

    Qubit layout
    ------------
    controls:
        0 ... n-1

    reusable scratch pool:
        n ... n + scratch_count - 1

    persistent group roots:
        immediately after the scratch pool

    final target:
        final circuit qubit
    """

    if n < 2:
        raise ValueError("n must be at least 2.")

    if k < 2:
        raise ValueError("k must be at least 2.")

    # ----------------------------------------------------------
    # Divide controls into consecutive groups.
    # ----------------------------------------------------------
    groups = [
        list(range(start, min(start + k, n)))
        for start in range(0, n, k)
    ]

    number_of_groups = len(groups)

    # A group of size s >= 2 needs:
    #   s - 2 lower-level scratch ancillas
    #   1 persistent root ancilla
    total_group_scratch = sum(
        max(0, len(group) - 2)
        for group in groups
    )

    number_of_group_root_ancillas = sum(
        len(group) >= 2
        for group in groups
    )

    # Combining g group roots until two remain needs g - 2
    # temporary upper-tree ancillas.
    upper_scratch_count = max(
        0,
        number_of_groups - 2,
    )

    # Lower group scratch is clean before the upper tree begins,
    # so the upper tree can reuse the same scratch qubits.
    scratch_count = max(
        total_group_scratch,
        upper_scratch_count,
    )

    scratch_start = n
    root_start = scratch_start + scratch_count
    target = root_start + number_of_group_root_ancillas

    qc = QuantumCircuit(
        target + 1,
        name=f"{n}c_k{k}_grouped_mct",
    )

    scratch_pool = list(
        range(
            scratch_start,
            scratch_start + scratch_count,
        )
    )

    root_pool = list(
        range(
            root_start,
            root_start + number_of_group_root_ancillas,
        )
    )

    # Each plan contains all levels for one group.
    # Its final level contains the persistent root gate.
    group_plans = []
    group_roots = []

    next_local_scratch = 0
    next_group_root = 0

    # ==========================================================
    # Construct the logical tree for every group.
    # No gates are emitted yet.
    # ==========================================================
    for group in groups:
        # A singleton group uses its original control as its root.
        if len(group) == 1:
            group_plans.append(
                {
                    "group": group,
                    "levels": [],
                    "root": group[0],
                }
            )

            group_roots.append(group[0])
            continue

        local_scratch_count = len(group) - 2

        local_scratch = scratch_pool[
            next_local_scratch:
            next_local_scratch + local_scratch_count
        ]

        next_local_scratch += local_scratch_count

        persistent_root = root_pool[next_group_root]
        next_group_root += 1

        current_nodes = list(group)
        levels = []
        local_scratch_index = 0

        # Build all levels below the persistent group root.
        while len(current_nodes) > 2:
            level = []
            next_nodes = []

            index = 0

            while index + 1 < len(current_nodes):
                left = current_nodes[index]
                right = current_nodes[index + 1]

                output = local_scratch[local_scratch_index]
                local_scratch_index += 1

                level.append(
                    (left, right, output)
                )

                next_nodes.append(output)
                index += 2

            # Carry an unmatched node to the next level.
            if index < len(current_nodes):
                next_nodes.append(
                    current_nodes[index]
                )

            levels.append(level)
            current_nodes = next_nodes

        # Final group level writes into the persistent root.
        levels.append(
            [
                (
                    current_nodes[0],
                    current_nodes[1],
                    persistent_root,
                )
            ]
        )

        group_plans.append(
            {
                "group": group,
                "levels": levels,
                "root": persistent_root,
            }
        )

        group_roots.append(persistent_root)

    max_group_levels = max(
        (
            len(plan["levels"])
            for plan in group_plans
        ),
        default=0,
    )

    # ==========================================================
    # Stage 1: compute matching levels across all groups.
    #
    # For k=6:
    #   global depth 1: Level 1 from every group
    #   global depth 2: Level 2 from every group
    #   global depth 3: root level from every group
    # ==========================================================
    for level_index in range(max_group_levels):
        used = False

        for plan in group_plans:
            levels = plan["levels"]

            if level_index >= len(levels):
                continue

            for left, right, output in levels[level_index]:
                qc.ccx(left, right, output)
                used = True

        if insert_barriers and used:
            qc.barrier()

    # ==========================================================
    # Stage 2: uncompute only levels below each group root.
    #
    # The final level of each plan is deliberately skipped.
    # Therefore, one root per non-singleton group remains dirty.
    # ==========================================================
    for level_index in reversed(range(max_group_levels - 1)):
        used = False

        for plan in group_plans:
            levels = plan["levels"]

            # The last level is the persistent root level.
            if level_index >= len(levels) - 1:
                continue

            for left, right, output in reversed(
                levels[level_index]
            ):
                qc.ccx(left, right, output)
                used = True

        if insert_barriers and used:
            qc.barrier()

    # At this point:
    #
    #   group-local scratch = clean
    #   persistent roots    = dirty
    #
    # The scratch pool can now be reused by the upper tree.

    # ==========================================================
    # Stage 3: compute the upper tree over the group roots.
    # ==========================================================
    current_nodes = list(group_roots)
    upper_levels = []
    upper_scratch_index = 0

    while len(current_nodes) > 2:
        level = []
        next_nodes = []

        index = 0

        while index + 1 < len(current_nodes):
            left = current_nodes[index]
            right = current_nodes[index + 1]

            if upper_scratch_index >= len(scratch_pool):
                raise RuntimeError(
                    "Insufficient scratch space for the upper tree."
                )

            output = scratch_pool[upper_scratch_index]
            upper_scratch_index += 1

            level.append(
                (left, right, output)
            )

            next_nodes.append(output)
            index += 2

        if index < len(current_nodes):
            next_nodes.append(
                current_nodes[index]
            )

        upper_levels.append(level)
        current_nodes = next_nodes

    for level in upper_levels:
        for left, right, output in level:
            qc.ccx(left, right, output)

        if insert_barriers and level:
            qc.barrier()

    # ==========================================================
    # Stage 4: apply the actual MCT target operation.
    # ==========================================================
    if len(current_nodes) == 2:
        qc.ccx(
            current_nodes[0],
            current_nodes[1],
            target,
        )

    elif len(current_nodes) == 1:
        # Only one group exists. Its root already stores the
        # conjunction of all n controls.
        qc.cx(
            current_nodes[0],
            target,
        )

    else:
        raise RuntimeError(
            "No roots remained for the final target operation."
        )

    if insert_barriers:
        qc.barrier()

    # ==========================================================
    # Stage 5: uncompute only the upper-tree scratch.
    #
    # This section is necessary because upper-tree ancillas are not
    # allowed to remain dirty.
    #
    # It does not touch the already-cleaned group-local subtrees.
    # ==========================================================
    for level in reversed(upper_levels):
        for left, right, output in reversed(level):
            qc.ccx(left, right, output)

        if insert_barriers and level:
            qc.barrier()

    # Persistent group roots deliberately remain dirty.
    dirty_group_roots = [
        plan["root"]
        for plan in group_plans
        if len(plan["group"]) >= 2
    ]

    qc.metadata = {
        "number_of_controls": n,
        "group_size": k,
        "groups": groups,
        "group_roots": group_roots,
        "dirty_group_roots": dirty_group_roots,
        "clean_scratch_qubits": scratch_pool,
        "target": target,
        "upper_levels": upper_levels,
    }

    return qc
# def create_circuit_k(n, k):
#     """
#     Build an n-control MCT AND-tree with k controls per group and
#     early uncomputation of module-internal ancillas.
 
#     Schedule
#     --------
#     For each group of k consecutive controls:
#         1. Compute the group's balanced AND-tree.
#         2. Immediately uncompute that group's INTERNAL ancillas,
#            top-down (parent before children). This is free of
#            recomputation because the children of the lowest-level
#            ancillas are the original controls, which are never
#            released.
#         3. Keep the group OUTPUT live -- the higher-level tree
#            consumes it.
 
#     Then combine the group outputs with a higher-level AND-tree. The
#     final root holds AND(all controls).
 
#     Group outputs are deliberately NOT restored. They occupy their
#     slots for the whole computation anyway (the higher tree needs
#     them), so restoring them would buy no space while costing a
#     recompute of every internal ancilla. The uncompute gates are
#     therefore MOVED EARLIER rather than added: total gate count is no
#     higher than the standard compute-then-reverse schedule, and the
#     internal slots are released partway through instead of at the end.
 
#     Qubit layout
#     ------------
#         0 .. n-1     original controls
#         n ..         ancillas, allocated in creation order
 
#     Returns
#     -------
#     QuantumCircuit
#         Circuit whose last-allocated qubit holds AND(controls).
#     """
#     if n < 2:
#         raise ValueError("n must be at least 2.")
#     if k < 2:
#         raise ValueError("k must be at least 2.")
 
#     gates = []
#     next_ancilla = n
 
#     group_outputs = []
 
#     # ---- per group: compute the local tree, then free internals ----
#     for start in range(0, n, k):
#         controls = list(range(start, min(start + k, n)))
 
#         if len(controls) == 1:
#             # A lone trailing control is already its own "output".
#             group_outputs.append(controls[0])
#             continue
 
#         group_gates = []
#         level = controls
 
#         while len(level) > 1:
#             next_level = []
#             for i in range(0, len(level), 2):
#                 if i + 1 < len(level):
#                     target = next_ancilla
#                     next_ancilla += 1
#                     group_gates.append((level[i], level[i + 1], target))
#                     next_level.append(target)
#                 else:
#                     next_level.append(level[i])
#             level = next_level
 
#         group_root = level[0]
#         group_outputs.append(group_root)
 
#         gates.extend(group_gates)
 
#         # Free every ancilla except the group output, top-down.
#         # Reversing the bottom-up build order gives parent-before-child,
#         # so each uncompute fires while its children still hold values.
#         for gate in reversed(group_gates):
#             if gate[2] != group_root:
#                 gates.append(gate)
 
#     # ---- combine the group outputs ----
#     level = group_outputs
 
#     while len(level) > 1:
#         next_level = []
#         for i in range(0, len(level), 2):
#             if i + 1 < len(level):
#                 target = next_ancilla
#                 next_ancilla += 1
#                 gates.append((level[i], level[i + 1], target))
#                 next_level.append(target)
#             else:
#                 next_level.append(level[i])
#         level = next_level
 
#     circuit = QuantumCircuit(next_ancilla)
#     for a, b, target in gates:
#         circuit.ccx(a, b, target)
 
#     return circuit

def transform_to_ccz_h(qc):
    dag = circuit_to_dag(qc)
    dag_drawer(dag,filename="dag.png")
    replacement = QuantumCircuit(3)
    replacement.h(2)
    replacement.ccz(0,1,2)
    replacement.h(2)
    for node in dag.topological_op_nodes():
        if node.name =="ccx":
            dag.substitute_node_with_dag(
                node,
                circuit_to_dag(replacement)
            )
    new_qc = dag_to_circuit(dag)
    return new_qc

def extract_toffoli_hyperedges(qc, include_ccz=True):
    """
    Extract Toffoli/CCZ interactions from a Qiskit circuit.

    Returns
    -------
    hyperedges : list[tuple[int, int, int]]
        Ordered list of 3-qubit interactions.

    weights : Counter
        Weighted hyperedge counts.
    """

    hyperedges = []

    target_gate_names = {"ccx"}
    if include_ccz:
        target_gate_names.add("ccz")

    for inst in qc.data:
        op = inst.operation
        qargs = inst.qubits

        if op.name in target_gate_names:
            qids = tuple(qc.find_bit(q).index for q in qargs)

            if len(qids) != 3:
                raise ValueError(f"{op.name} does not have 3 qubits: {qids}")

            # For routing, CCZ/Toffoli hyperedge is unordered.
            edge = tuple(sorted(qids))

            hyperedges.append(edge)

    # print(hyperedges)
    weights = Counter(hyperedges)

    #hyperedges are sequential toffoli edges
    edge_dict = dict(weights)
    return edge_dict, hyperedges
from qiskit import QuantumCircuit


def create_parallel_k_grouped_mct(
    n: int,
    k: int = 6,
    insert_barriers: bool = True,
) -> QuantumCircuit:
    """
    Build an n-controlled X using ordered groups of at most k controls.

    For every group:
        1. Compute its binary reduction level by level.
        2. Retain the final group-root ancilla.
        3. Immediately uncompute all lower-level scratch ancillas.

    Equal levels from different groups are emitted together, allowing
    them to execute at the same logical depth.

    The retained group roots are then combined into an upper binary
    tree and applied to the final target.

    No computation is performed after the final target gate.

    Final ancilla state
    -------------------
    Clean:
        lower-level scratch ancillas inside each k-group

    Dirty:
        retained k-group roots
        upper-tree ancillas

    Qubit layout
    ------------
    controls:
        0 ... n-1

    group-local scratch:
        after controls

    persistent group roots:
        after group-local scratch

    upper-tree scratch:
        after group roots

    final target:
        last qubit
    """

    if n < 2:
        raise ValueError("n must be at least 2.")

    if k < 2:
        raise ValueError("k must be at least 2.")

    # Consecutive ordered groups.
    groups = [
        list(range(start, min(start + k, n)))
        for start in range(0, n, k)
    ]

    # A group of size s >= 2 needs:
    #
    #   s - 2 lower-level scratch ancillas
    #   1 persistent group-root ancilla
    #
    # A singleton group uses its original control directly.
    total_group_scratch = sum(
        max(0, len(group) - 2)
        for group in groups
    )

    number_of_group_roots = sum(
        len(group) >= 2
        for group in groups
    )

    number_of_effective_roots = len(groups)

    # To reduce g roots until two remain, g - 2 upper ancillas
    # are required.
    upper_scratch_count = max(
        0,
        number_of_effective_roots - 2,
    )

    group_scratch_start = n

    group_root_start = (
        group_scratch_start
        + total_group_scratch
    )

    upper_scratch_start = (
        group_root_start
        + number_of_group_roots
    )

    target = (
        upper_scratch_start
        + upper_scratch_count
    )

    qc = QuantumCircuit(
        target + 1,
        name=f"{n}c_parallel_k{k}_dirty_roots",
    )

    group_plans = []
    group_roots = []

    next_group_scratch = group_scratch_start
    next_group_root = group_root_start

    # ==========================================================
    # Build the reduction plan for each k-group.
    # ==========================================================
    for group in groups:
        # A singleton group needs no generated root.
        if len(group) == 1:
            group_plans.append(
                {
                    "group": group,
                    "levels": [],
                    "root_gate": None,
                    "root": group[0],
                }
            )

            group_roots.append(group[0])
            continue

        scratch_count = len(group) - 2

        group_scratch = list(
            range(
                next_group_scratch,
                next_group_scratch + scratch_count,
            )
        )

        next_group_scratch += scratch_count

        persistent_root = next_group_root
        next_group_root += 1

        current_nodes = list(group)
        lower_levels = []
        scratch_index = 0

        # Build all levels below the retained group root.
        while len(current_nodes) > 2:
            next_nodes = []
            level = []

            index = 0

            while index + 1 < len(current_nodes):
                left = current_nodes[index]
                right = current_nodes[index + 1]

                output = group_scratch[scratch_index]
                scratch_index += 1

                level.append(
                    (left, right, output)
                )

                next_nodes.append(output)
                index += 2

            # Carry an unmatched node to the next level.
            if index < len(current_nodes):
                next_nodes.append(
                    current_nodes[index]
                )

            lower_levels.append(level)
            current_nodes = next_nodes

        # The final pair writes into the persistent group root.
        root_gate = (
            current_nodes[0],
            current_nodes[1],
            persistent_root,
        )

        group_plans.append(
            {
                "group": group,
                "levels": lower_levels,
                "root_gate": root_gate,
                "root": persistent_root,
            }
        )

        group_roots.append(persistent_root)

    max_lower_levels = max(
        (
            len(plan["levels"])
            for plan in group_plans
        ),
        default=0,
    )

    # ==========================================================
    # Stage 1: compute matching levels across all groups.
    # ==========================================================
    for level_index in range(max_lower_levels):
        level_used = False

        for plan in group_plans:
            levels = plan["levels"]

            if level_index >= len(levels):
                continue

            for left, right, output in levels[level_index]:
                qc.ccx(left, right, output)
                level_used = True

        if insert_barriers and level_used:
            qc.barrier()

    # ==========================================================
    # Stage 2: compute every retained group root in parallel.
    # ==========================================================
    root_level_used = False

    for plan in group_plans:
        root_gate = plan["root_gate"]

        if root_gate is not None:
            qc.ccx(*root_gate)
            root_level_used = True

    if insert_barriers and root_level_used:
        qc.barrier()

    # ==========================================================
    # Stage 3: immediately clean all lower group levels.
    #
    # The retained group-root ancillas remain dirty.
    # ==========================================================
    for level_index in reversed(range(max_lower_levels)):
        level_used = False

        for plan in group_plans:
            levels = plan["levels"]

            if level_index >= len(levels):
                continue

            for left, right, output in reversed(
                levels[level_index]
            ):
                qc.ccx(left, right, output)
                level_used = True

        if insert_barriers and level_used:
            qc.barrier()

    # ==========================================================
    # Stage 4: combine retained group roots.
    # ==========================================================
    current_nodes = list(group_roots)
    upper_levels = []

    next_upper_scratch = upper_scratch_start

    while len(current_nodes) > 2:
        next_nodes = []
        level = []

        index = 0

        while index + 1 < len(current_nodes):
            left = current_nodes[index]
            right = current_nodes[index + 1]

            output = next_upper_scratch
            next_upper_scratch += 1

            level.append(
                (left, right, output)
            )

            next_nodes.append(output)
            index += 2

        if index < len(current_nodes):
            next_nodes.append(
                current_nodes[index]
            )

        upper_levels.append(level)
        current_nodes = next_nodes

    # Emit the upper tree level by level.
    for level in upper_levels:
        for left, right, output in level:
            qc.ccx(left, right, output)

        if insert_barriers and level:
            qc.barrier()

    # ==========================================================
    # Stage 5: apply the final MCT operation.
    #
    # Stop here. No mirrored uncomputation follows.
    # ==========================================================
    if len(current_nodes) == 2:
        qc.ccx(
            current_nodes[0],
            current_nodes[1],
            target,
        )

    elif len(current_nodes) == 1:
        # Only one group exists. Its root already stores the
        # conjunction of every control.
        qc.cx(
            current_nodes[0],
            target,
        )

    else:
        raise RuntimeError(
            "Upper reduction produced no final roots."
        )

    if insert_barriers:
        qc.barrier()

    qc.metadata = {
        "number_of_controls": n,
        "group_size": k,
        "groups": groups,
        "group_roots": group_roots,
        "group_scratch_count": total_group_scratch,
        "upper_scratch_count": upper_scratch_count,
        "upper_levels": upper_levels,
        "target": target,
        "clean_ancillas": "group-local lower-level scratch",
        "dirty_ancillas": (
            "persistent group roots and upper-tree scratch"
        ),
    }

    return qc

# qc = create_k_grouped_mct(8,6)
# qc.draw(output="mpl",filename="mct_12.pdf",
#     fold=-1)
# print(qc)
# # qc1 = create_circuit_k(12,6)
# print(extract_toffoli_hyperedges(qc))
# print(extract_toffoli_hyperedges(qc1))
# transform_to_ccz_h(qc)