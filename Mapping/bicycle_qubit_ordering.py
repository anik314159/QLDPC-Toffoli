"""
bicycle_qubit_ordering.py

Bridges your existing placement (linear_mapping_tree_baseline, run
with module_capacity=11) to the qubit ordering bicycle_compiler
implicitly assumes.

Why this is needed
-------------------
bicycle_compiler assigns qubits to gross-code blocks purely by
position in the PBC basis array: `basis.chunks_exact(11)` (see
crates/bicycle_compiler/src/compile.rs). Position 0-10 -> block 0,
11-21 -> block 1, etc. It has no placement logic of its own.

Your linear_mapping_tree_baseline already solves exactly this problem
one level up: it groups each 4-control MCT subtree (plus its local
tree ancillas) into one module, with module_capacity=11 -- already
sized to match a gross-code block. So no new placement algorithm is
needed here, just a thin adapter that:

  1. reads architecture.graph / per-module "placed_qubits" (module
     order == chain position, exactly like linear_chain's M0-M1-M2...
     path, which is the same topology bicycle_compiler assumes for
     its modules),
  2. pads each module's qubit list up to exactly `block_size` (11)
     with unused padding slots, so block boundaries in our qubit
     order line up with the compiler's fixed chunking (it only pads
     the *tail* of the whole basis, not per-module, so per-module
     padding has to happen on our side), and
  3. relabels the Toffoli triples / circuit onto that order before
     handing off to toffoli_to_bicycle_pbc.py.

As a side effect, this also tells you which Toffoli hyperedges span
more than one block -- those are the ones that cost a multi-hop GHZ /
JointMeasure chain down the module path.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import networkx as nx
from qiskit import QuantumCircuit

ToffoliTriple = tuple[Hashable, Hashable, Hashable]


def factory_ordered_modules(architecture: nx.Graph) -> list[Hashable]:
    """
    Order module nodes by graph distance from the architecture's
    factory, FARTHEST first, factory-attached module LAST.

    This matches bicycle_compiler's PathArchitecture exactly: "blocks
    plus one magic state factory at the end of the path"
    (architecture.rs) -- every rotation's magic-state injection
    always happens at the single last block (index n-1), regardless
    of where its Pauli support starts. So whichever module holds your
    factory has to be the one that lands in the last block.

    Only meaningful for a single-factory architecture (arch.py's
    linear_chain(factory_period=0), which attaches exactly one
    factory to M0). bicycle_compiler's PathArchitecture has no
    representation for multiple factories or a grid topology, so this
    raises rather than silently producing an order that doesn't
    correspond to anything the compiler actually does.
    """
    factory_nodes = [
        n for n, d in architecture.nodes(data=True) if d.get("kind") == "factory"
    ]
    if len(factory_nodes) != 1:
        raise ValueError(
            f"Found {len(factory_nodes)} factory node(s) in the architecture; "
            "bicycle_compiler's PathArchitecture models exactly one magic-state "
            "factory at the end of a single chain. Use linear_chain(factory_period=0) "
            "(one factory on M0) -- factory_period>0 or a grid architecture doesn't "
            "correspond to anything this compiler can target."
        )
    factory_node = factory_nodes[0]
    factory_module = architecture.nodes[factory_node]["attached_module"]

    module_nodes = [
        n for n, d in architecture.nodes(data=True) if d.get("kind") == "module"
    ]

    distances = nx.shortest_path_length(architecture, source=factory_node)
    missing = [m for m in module_nodes if m not in distances]
    if missing:
        raise ValueError(
            f"Module(s) {missing} are not connected to the factory in the "
            "architecture graph -- can't determine chain order."
        )

    # Farthest from factory first, factory's own module last.
    ordered = sorted(module_nodes, key=lambda m: -distances[m])
    assert ordered[-1] == factory_module, (
        f"Expected {factory_module} (factory's module) last, got {ordered[-1]}. "
        "This usually means the architecture isn't a simple path graph."
    )
    return ordered


def module_aligned_qubit_order(
    architecture: nx.Graph,
    block_size: int = 11,
) -> list[Hashable | None]:
    """
    Flatten architecture's per-module placement into a single ordered
    list, one contiguous run of `block_size` positions per module, in
    factory-aware chain order (see factory_ordered_modules -- NOT raw
    module-index order, since e.g. linear_chain(factory_period=0)
    puts the factory on M0, the *front* of the index order, while
    bicycle_compiler always wants its factory module at position
    n-1, the *end*). Padded with None where a module has fewer than
    `block_size` qubits placed.

    None entries become permanently-unused circuit qubits -- they'll
    show up as "I" in every Pauli term, exactly like bicycle_compiler's
    own tail-padding, just applied per module instead of once at the
    end.
    """
    module_nodes = factory_ordered_modules(architecture)

    order: list[Hashable | None] = []
    for module in module_nodes:
        qubits = list(architecture.nodes[module].get("placed_qubits", []))
        if len(qubits) > block_size:
            raise ValueError(
                f"{module} has {len(qubits)} placed qubits, exceeding "
                f"block_size={block_size}. Re-run placement with "
                f"module_capacity<={block_size}."
            )
        qubits = qubits + [None] * (block_size - len(qubits))
        order.extend(qubits)

    return order


def circuit_from_placement(
    architecture: nx.Graph,
    toffoli_sequence: Sequence[ToffoliTriple],
    block_size: int = 11,
) -> QuantumCircuit:
    """
    Build the QuantumCircuit to feed into
    toffoli_to_bicycle_pbc.toffoli_circuit_to_pbc_jsonl, with qubits
    laid out so that bicycle_compiler's chunks_exact(block_size)
    reproduces your linear_mapping_tree_baseline placement.
    """
    order = module_aligned_qubit_order(architecture, block_size)
    position = {qubit: i for i, qubit in enumerate(order) if qubit is not None}

    circuit = QuantumCircuit(len(order))
    for c1, c2, target in toffoli_sequence:
        for q in (c1, c2, target):
            if q not in position:
                raise ValueError(
                    f"Qubit {q!r} (from Toffoli {(c1, c2, target)!r}) has no "
                    "module placement -- run linear_mapping_tree_baseline "
                    "on the full hyperedge set first."
                )
        circuit.ccx(position[c1], position[c2], position[target])

    return circuit


def block_span_report(
    architecture: nx.Graph,
    toffoli_sequence: Sequence[ToffoliTriple],
    block_size: int = 11,
) -> dict:
    """
    Diagnostic: for each Toffoli, how many blocks (gross-code modules)
    its three qubits span, and the max chain distance between them.
    Cross-block Toffolis are the ones that cost a multi-hop GHZ /
    JointMeasure chain in bicycle_compiler; same-block ones are free
    of inter-module cost.
    """
    order = module_aligned_qubit_order(architecture, block_size)
    block_of = {qubit: i // block_size for i, qubit in enumerate(order) if qubit is not None}

    per_op = []
    for c1, c2, target in toffoli_sequence:
        blocks = {block_of[q] for q in (c1, c2, target)}
        per_op.append(
            {
                "toffoli": (c1, c2, target),
                "blocks": sorted(blocks),
                "num_blocks": len(blocks),
                "chain_span": max(blocks) - min(blocks),
            }
        )

    cross_block = [op for op in per_op if op["num_blocks"] > 1]
    return {
        "num_toffolis": len(per_op),
        "num_cross_block": len(cross_block),
        "max_chain_span": max((op["chain_span"] for op in per_op), default=0),
        "per_operation": per_op,
    }