"""
compile_to_bicycle.py

Glues your existing pipeline together and writes the PBC file to feed
bicycle_compiler.
"""
from Applications.MCTdecomp.binary_tree_decomposition import extract_toffoli_hyperedges, create_circuit


from linear_mapping import linear_mapping_tree_baseline

from bicycle_qubit_ordering import circuit_from_placement, block_span_report
from toff_to_bicycle_pbc import toffoli_circuit_to_pbc_jsonl, summarize_pbc


def main(control_count: int, out_path: str = "circuit.pbc.jsonl", code: str = "gross"):
    toffoli_hyperedges, toffoli_sequence = extract_toffoli_hyperedges(
        create_circuit(control_count)
    )

    architecture = linear_mapping_tree_baseline(
        control_count=control_count,
        toffoli_hyperedge_dict=toffoli_hyperedges,
        controls_per_module=4,
        module_capacity=11,   # must be <=11 to fit a gross-code block
        factory_period=0,     # exactly one factory -- required by bicycle_compiler
    )

    circuit = circuit_from_placement(architecture, toffoli_sequence, block_size=11)

    report = block_span_report(architecture, toffoli_sequence, block_size=11)
    print(f"{report['num_toffolis']} Toffolis, "
          f"{report['num_cross_block']} cross-block, "
          f"max chain span {report['max_chain_span']}")
    print(summarize_pbc(circuit))

    pbc_jsonl = toffoli_circuit_to_pbc_jsonl(circuit)
    with open(out_path, "w") as f:
        f.write(pbc_jsonl)
    print(f"Wrote {out_path} ({circuit.num_qubits} qubits, "
          f"{len(pbc_jsonl.splitlines())} PBC instructions)")

    print(
        "\nNext, in your bicycle-architecture-compiler checkout:\n"
        f"  ./target/release/bicycle_compiler {code} --measurement-table data/table_{code} "
        f"< {out_path} > circuit.isa.jsonl\n"
        f"  ./target/release/bicycle_numerics {circuit.num_qubits} {code}_1e-4 < circuit.isa.jsonl"
    )

    return pbc_jsonl


if __name__ == "__main__":
    main(control_count=12)