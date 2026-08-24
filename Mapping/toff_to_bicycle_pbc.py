"""
toffoli_to_bicycle_pbc.py

Converts a Toffoli/MCT circuit into a Pauli-Based Computation (PBC)
instruction stream in the exact JSON-lines format consumed by
qiskit-community/bicycle-architecture-compiler's `bicycle_compiler`
binary (Tour de Gross, arXiv:2506.03094).

This deliberately does NOT reimplement CCZ-state injection, factory
routing, or an error model by hand (that was the wrong layer to hand-
roll, see previous turn). `bicycle_compiler` already does the PBC ->
Gross-code-ISA compilation internally, using the exact measurement
tables from the paper, and `bicycle_numerics` already does the
circuit-noise error estimation. This script's only job is to produce
correct input for that pipeline, using the same officially-supported
route the repo's own scripts/qiskit_demo.py and
notebooks/custom_circuits.ipynb use:

    Toffoli circuit
      -> transpile to {Clifford, rz, t, tdg}
      -> qiskit.transpiler.passes.LitinskiTransformation
      -> PBC circuit (PauliEvolutionGate + PauliProductMeasurement)
      -> JSON lines: {"Rotation": {...}} / {"Measurement": {...}}

Requires: qiskit >= 2.3.0rc1 (for LitinskiTransformation).
Requires the bicycle-architecture-compiler repo checked out alongside
(for scripts/qiskit_parser.py) -- or copy iter_qiskit_pbc_circuit from
there; it's small and reproduced inline below to keep this file
self-contained.

Usage
-----
    from toffoli_to_bicycle_pbc import (
        circuit_from_toffoli_sequence,
        toffoli_circuit_to_pbc_jsonl,
    )

    # Option A: you already have a qiskit circuit (e.g. from
    # Applications.MCTdecomp.binary_tree_decomposition.create_circuit)
    pbc_jsonl = toffoli_circuit_to_pbc_jsonl(my_qiskit_circuit)

    # Option B: you only have the (c1, c2, target) triples, e.g. from
    # toffoli_sequenced_hyperedge in ppm_gate_synthesis.py
    circuit = circuit_from_toffoli_sequence(toffoli_sequenced_hyperedge, num_qubits=12)
    pbc_jsonl = toffoli_circuit_to_pbc_jsonl(circuit)

    with open("circuit.pbc.jsonl", "w") as f:
        f.write(pbc_jsonl)

Then, in the bicycle-architecture-compiler checkout (after `cargo
build --release` and generating a measurement table):

    ./target/release/bicycle_compiler gross \\
        --measurement-table data/table_gross \\
        < circuit.pbc.jsonl \\
        > circuit.bicycle_isa.jsonl

    ./target/release/bicycle_numerics 12 gross_1e-4 \\
        < circuit.bicycle_isa.jsonl
        # -> CSV of per-instruction / total error stats
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from Applications.MCTdecomp.binary_tree_decomposition import extract_toffoli_hyperedges, create_circuit
from linear_mapping import linear_mapping_tree_baseline
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.transpiler.passes import LitinskiTransformation

ToffoliTriple = tuple[int, int, int]

# Pauli lookup, reproduced verbatim from
# bicycle-architecture-compiler/scripts/qiskit_parser.py so this file
# has no dependency on that repo's Python path.
_PAULI_TABLE = {
    (True, True): "Y",
    (True, False): "Z",
    (False, True): "X",
    (False, False): "I",
}


def circuit_from_toffoli_sequence(
    toffoli_sequence: Sequence[ToffoliTriple],
    num_qubits: int,
) -> QuantumCircuit:
    """
    Build a QuantumCircuit of CCX gates from a (c1, c2, target) triple
    sequence, e.g. `toffoli_sequenced_hyperedge` from
    ppm_gate_synthesis.py / binary_tree_decomposition.py.

    Prefer feeding `bicycle_compiler` your actual
    `create_circuit(...)` output directly when you have it -- this
    reconstruction is a fallback for when only the flattened
    (c1, c2, target) triples are available.
    """
    circuit = QuantumCircuit(num_qubits)
    for c1, c2, target in toffoli_sequence:
        circuit.ccx(c1, c2, target)
    return circuit


def _iter_pbc_json(pbc: QuantumCircuit) -> Iterator[dict]:
    """
    Yield PBC instructions as plain dicts.

    Identical logic to
    bicycle-architecture-compiler/scripts/qiskit_parser.py::iter_qiskit_pbc_circuit
    (as_str=False case), reproduced here to keep this module
    standalone. `pbc` must contain only PauliEvolutionGate (single
    Pauli) and PauliProductMeasurement instructions, i.e. it must
    already be the output of LitinskiTransformation.
    """
    qubit_to_index = {qubit: index for index, qubit in enumerate(pbc.qubits)}

    for inst in pbc.data:
        if inst.name == "PauliEvolution":
            evo = inst.operation
            if isinstance(evo.operator, list):
                raise ValueError("Grouped operators in Pauli not supported.")

            op = evo.operator.to_sparse_list()
            if len(op) > 1:
                raise ValueError("PauliEvolution is not a single rotation.")
            paulis, indices, coeff = op[0]

            basis = ["I"] * pbc.num_qubits
            for pauli, i in zip(paulis, indices):
                basis[i] = pauli

            angle = evo.params[0] * np.real(coeff)
            yield {"Rotation": {"basis": basis, "angle": str(angle)}}

        elif inst.name == "pauli_product_measurement":
            ppm = inst.operation
            z, x, phase = ppm._to_pauli_data()

            basis = ["I"] * pbc.num_qubits
            for qubit, zq, xq in zip(inst.qubits, z, x):
                basis[qubit_to_index[qubit]] = _PAULI_TABLE[(zq, xq)]

            flipped = bool(phase == 2)
            yield {"Measurement": {"basis": basis, "flip_result": flipped}}

        else:
            raise ValueError(f"Unsupported instruction in PBC circuit: {inst.name}")


def toffoli_circuit_to_pbc_jsonl(
    circuit: QuantumCircuit,
    fix_clifford: bool = False,
) -> str:
    """
    Full conversion: arbitrary Toffoli/Clifford circuit -> PBC JSON
    lines, ready to pipe into `bicycle_compiler`.

    Mirrors scripts/qiskit_demo.py::compile_pbc exactly (same basis
    gate set, same LitinskiTransformation call).
    """
    basis_gates = ["rz", "t", "tdg"] + get_clifford_gate_names()
    transpiled = transpile(circuit, basis_gates=basis_gates)

    pbc = LitinskiTransformation(fix_clifford=fix_clifford)(transpiled)

    lines = (json.dumps(inst).replace(" ", "") for inst in _iter_pbc_json(pbc))
    return "\n".join(lines) + "\n"


def summarize_pbc(circuit: QuantumCircuit, fix_clifford: bool = False) -> dict:
    """Quick counts, useful before committing to a full compiler run."""
    basis_gates = ["rz", "t", "tdg"] + get_clifford_gate_names()
    transpiled = transpile(circuit, basis_gates=basis_gates)
    pbc = LitinskiTransformation(fix_clifford=fix_clifford)(transpiled)

    n_rotations = sum(1 for inst in pbc.data if inst.name == "PauliEvolution")
    n_measurements = sum(
        1 for inst in pbc.data if inst.name == "pauli_product_measurement"
    )
    return {
        "num_qubits": pbc.num_qubits,
        "num_rotations": n_rotations,
        "num_measurements": n_measurements,
        "num_pbc_instructions": n_rotations + n_measurements,
    }


if __name__ == "__main__":
    # Minimal smoke test: a single Toffoli, matching the standard
    # 7-T-gate / 6-CNOT decomposition.
    toffoli_hyperedges,toffoli_sequenced_hyperedge = extract_toffoli_hyperedges(create_circuit(12))
    # architecture = linear_mapping_tree_baseline(
    #     control_count=12,
    #     toffoli_hyperedge_dict=toffoli_hyperedges,
    #     controls_per_module=4,
    #     module_capacity=11,
    #     factory_period=0,
    # )
    # print(toffoli_hyperedges)
    demo_circuit = circuit_from_toffoli_sequence(toffoli_sequenced_hyperedge, num_qubits=23)
    print(summarize_pbc(demo_circuit))
    print(toffoli_circuit_to_pbc_jsonl(demo_circuit))