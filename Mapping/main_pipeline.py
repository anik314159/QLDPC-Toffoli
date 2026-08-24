"""
run_ccz_pipeline.py

End-to-end driver: Toffoli circuit -> placed architecture -> CCZ-injection
bicycle instructions -> JSONL for bicycle_numerics.

Run:
    python run_ccz_pipeline.py                    # default: 6-MCT, 1 factory
    python run_ccz_pipeline.py --controls 12 --factory-period 2

Then follow the printed bicycle_numerics command.
"""

from __future__ import annotations

import argparse
import os
import sys

import networkx as nx

# Make the project root importable regardless of invocation directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Architectures.arch import linear_chain

try:
    from Applications.MCTdecomp.binary_tree_decomposition import (
        extract_toffoli_hyperedges,
        create_circuit,create_circuit_k
    )
    HAVE_REAL_DECOMPOSITION = True
except ImportError:  # pragma: no cover
    HAVE_REAL_DECOMPOSITION = False
from linear_mapping import (
    linear_mapping_tree_baseline,
    calculate_routing_volume,
)
from bicycle_emitter import (
    CCZParams,
    emit,
    validate_emitted,
    write_jsonl,
    summarize,
)


# ---------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------

def build_and_tree(control_count, ancilla_base=1000):
    """
    Balanced binary AND-tree over `control_count` controls.

    Returns
    -------
    hyperedges : dict[(a, b, target)] -> 1
        Hyperedge dict in the form linear_mapping_tree_baseline wants.
    sequence : list[(a, b, target)]
        Topologically ordered Toffoli execution sequence.

    STAND-IN ONLY. Used when Applications.MCTdecomp is unavailable.
    Emits the COMPUTE half of the tree with no uncomputation, so its
    Toffoli count is roughly half that of the project's real
    create_circuit(), which does include uncomputation.
    """
    if control_count < 2:
        raise ValueError("control_count must be at least 2.")

    next_ancilla = [ancilla_base]

    def fresh():
        qubit = next_ancilla[0]
        next_ancilla[0] += 1
        return qubit

    hyperedges = {}
    sequence = []

    level = list(range(control_count))

    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                target = fresh()
                edge = (level[i], level[i + 1], target)
                hyperedges[edge] = 1
                sequence.append(edge)
                next_level.append(target)
            else:
                # Odd one out is carried to the next level unchanged.
                next_level.append(level[i])
        level = next_level

    return hyperedges, sequence


# ---------------------------------------------------------------
# Architecture construction
# ---------------------------------------------------------------

def build_linear_architecture(
    control_count,
    hyperedges,
    controls_per_module=4,
    module_capacity=11,
    factory_period=0,
    uncompute = 0
):
    """Linear chain, placed with the tree-aware baseline."""
    return linear_mapping_tree_baseline(
        control_count=control_count,
        toffoli_hyperedge_dict=hyperedges,
        controls_per_module=controls_per_module,
        module_capacity=module_capacity,
        factory_period=factory_period,
        uncompute = uncompute
    )


# ---------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------

def report_placement(architecture):
    print("Placement")
    print("---------")
    modules = sorted(
        (
            (node, data)
            for node, data in architecture.nodes(data=True)
            if data.get("kind") == "module"
        ),
        key=lambda item: item[1].get("index", 0),
    )
    for module, data in modules:
        placed = data.get("placed_qubits", [])
        print(f"  {module:8s} {len(placed):2d}/11  {placed}")

    for node, data in architecture.nodes(data=True):
        if data.get("kind") == "factory":
            print(f"  {node} -> attached to {data.get('attached_module')}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Emit CCZ-injection bicycle instructions for bicycle_numerics."
    )
    parser.add_argument("--controls", type=int, default=6,
                        help="Number of MCT controls (default: 6).")
    parser.add_argument("--controls-per-module", type=int, default=4)
    parser.add_argument("--uncompute", type=int, default=0)
    parser.add_argument("--module-capacity", type=int, default=11)
    parser.add_argument("--factory-period", type=int, default=2,
                        help="0 = one factory on M0, "
                             "k>0 = a factory every k modules.")
    parser.add_argument("--aut-per-pauli", type=int, default=2,
                        help="Automorphisms to reach positions 1/7.")
    parser.add_argument("--synthetic-tree", action="store_true",
                        help="Force the built-in stand-in tree instead of "
                             "Applications.MCTdecomp (compute half only).")
    parser.add_argument("--output", default="circuit.bicycle.jsonl")
    parser.add_argument("--model", default="gross_1e-3",
                        choices=["gross_1e-3", "gross_1e-4",
                                 "two-gross_1e-3", "two-gross_1e-4",
                                 "fake_slow"])
    args = parser.parse_args()

    # 1. Circuit
    if HAVE_REAL_DECOMPOSITION and not args.synthetic_tree:
        # The project's own decomposition. This already includes
        # uncomputation of the tree ancillas.
        if args.uncompute == 1:
            hyperedges, sequence = extract_toffoli_hyperedges(
                create_circuit_k(args.controls,args.controls_per_module)
            )
        else:
             hyperedges, sequence = extract_toffoli_hyperedges(
                create_circuit(args.controls)
            )
        source = "binary_tree_decomposition (uncomputation included)"
     
    else:
        # Fallback stand-in used only when Applications.MCTdecomp is
        # not importable. COMPUTE HALF ONLY -- no uncomputation, so
        # counts are roughly half of a real MCT.
        hyperedges, sequence = build_and_tree(args.controls)
        source = "synthetic AND-tree (COMPUTE ONLY, no uncomputation)"

    print(f"{args.controls}-MCT -> {len(sequence)} Toffolis")
    print(f"  source: {source}\n")

    # 2. Architecture + placement
    architecture = build_linear_architecture(
        control_count=args.controls,
        hyperedges=hyperedges,
        controls_per_module=args.controls_per_module,
        module_capacity=args.module_capacity,
        factory_period=args.factory_period,
        uncompute=args.uncompute
    )

    report_placement(architecture)

    volume = calculate_routing_volume(
        architecture_graph=architecture,
        toffoli_hyperedge_dict=hyperedges,
    )
    print(f"Routing volume: {volume}\n")

    # 3. Emit bicycle instructions
    params = CCZParams(
        aut_per_pauli=args.aut_per_pauli,
    )
    result = emit(architecture, sequence, params)

    # 4. Validate before trusting anything
    validate_emitted(result, architecture)
    print("Adjacency validation: PASSED\n")

    print(summarize(result))
    print()

    print("Per-Toffoli detail")
    print("------------------")
    for entry in result.per_toffoli:
        kind = "CROSS" if entry["cross_module"] else "local"
        print(f"  {str(entry['toffoli']):24s} factory={entry['factory']:4s} "
              f"magic={str(entry['magic_module']):8s} hops={entry['relay_hops']:3d} "
              f"{kind}")
    print()

    # 5. Write
    write_jsonl(result, args.output)
    print(f"Wrote {args.output}\n")

    print("=" * 62)
    print("Run bicycle_numerics")
    print("=" * 62)
    print("From your bicycle-architecture-compiler checkout:")
    print()
    print("  cargo build --release")
    print()
    print(f"  ./target/release/bicycle_numerics {result.numerics_qubits} "
          f"{args.model} \\")
    print(f"      < {os.path.abspath(args.output)} \\")
    print("      > results.csv")
    print()
    print("Notes:")
    print(f"  * {result.numerics_qubits} = {result.num_blocks} blocks x 11 "
          "(PathArchitecture::for_qubits uses div_ceil(11)).")
    print("  * No measurement table needed -- bicycle_numerics reads the")
    print("    compiled instruction stream directly.")
    print(f"  * Output is CSV, one row per line of input "
          f"({len(result.lines)} rows = one per Toffoli),")
    print("    with total_error cumulative through that point.")
    print("  * Factory physical qubits (f + a' each) are NOT in the reported")
    print("    qubit count -- add them yourself for footprint numbers.")


if __name__ == "__main__":
    main()