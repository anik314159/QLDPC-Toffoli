"""
ccz_bicycle_emitter.py

Emit bicycle instructions (Tour de Gross ISA) for a Toffoli circuit
compiled via CCZ-state injection on a MULTI-FACTORY BB-code
architecture, in the JSON format consumed by `bicycle_numerics`.

Design commitments
------------------
* Multiple factories are supported. Each Toffoli is assigned to the
  factory minimising the metric-closure MST over
  {operand modules} u {factory}, i.e. the same cost model the user's
  ppm_gate_synthesis.py already uses, and the same spanning-tree
  justification Tour de Gross Sec. 3.4 gives for generalising their
  scheme beyond a single path.

* Routing and adjacency are computed from the architecture graph via
  nx.shortest_path rather than block-index arithmetic. Block indices
  are just labels for run_numerics' per-block clock bookkeeping.

* CCZ injection, not per-T Litinski rotations. One |CCZ> state is
  assumed delivered by a Gidney-style factory into that factory's
  attached module (the "magic module"), then consumed by three
  Pauli-product measurements. The outcome-conditioned Clifford
  corrections are absorbed into the Pauli frame, not emitted (see
  step 4 in emit()).

Notes
-----
1. `bicycle_numerics` does NOT call PathArchitecture::validate_operation,
   so adjacency is our responsibility; `validate_emitted` below checks
   it against the architecture graph.

2. `PathArchitecture::for_qubits` uses `qubits.div_ceil(11)`, so the
   CLI qubit count to reproduce N blocks is N * 11. `emit()` returns
   this as `numerics_qubits`.

3. Physical qubit accounting for factories (f + a' per factory, Table
   1/3) is NOT included in run_numerics' reported `qubits`. Add
   K * (f + a') yourself when reporting total footprint.

Usage
-----
    from ccz_bicycle_emitter import emit, write_jsonl, validate_emitted

    result = emit(architecture, toffoli_sequence)
    validate_emitted(result, architecture)          # raises on illegal ops
    write_jsonl(result, "circuit.bicycle.jsonl")

    # then:
    #   bicycle_numerics <result.numerics_qubits> gross_1e-3 \\
    #       < circuit.bicycle.jsonl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Hashable, Sequence

import json
import networkx as nx

from math import ceil

Instruction = Any            # str | dict
BlockOp = tuple[int, Instruction]
Operation = list[BlockOp]    # joint ops carry 2 entries
Line = list[Operation]       # one run_numerics iteration


# ---------------------------------------------------------------
# Modeling parameters (knobs, not results)
# ---------------------------------------------------------------

@dataclass
class CCZParams:
    # NOTE ON FACTORY COST. Table 2's T-injection error is compound:
    #   P_T = P_factory + P_C
    # where P_factory is the distillation cost internal to the
    # factory, and P_C is the INTER-MODULE MEASUREMENT that delivers
    # the state into a data block.
    #
    # By DEFAULT this emitter drops P_factory (p_factory = 0.0
    # below). It is independent of placement and identical across the
    # mappings being compared, so it cancels in any comparison -- but
    # it means the default figure is a PLACEMENT-ATTRIBUTABLE error,
    # not an end-to-end one. Set p_factory to get the end-to-end
    # figure. (A CCZ factory's P_factory differs from a T factory's,
    # so Table 2's values do not transfer directly -- see the anchor
    # points on p_factory.)
    #
    # The delivery cost is NOT dropped: it is exactly the
    # cross-module traffic under study, and appears below as the
    # relayed joint Pauli product measurements.

    # Automorphisms needed to bring one Pauli's operands onto
    # positions 1/7. nr_generators() is in {0,1,2}; 2 is the
    # pessimistic bound.
    aut_per_pauli: int = 2

    # Basis for the three entangling measurements, in operand order
    # (control, control, target).
    #
    # The Hadamards of the Toffoli = H(t) . CCZ . H(t) identity are
    # NOT emitted as separate instructions: BicycleISA has no
    # Hadamard, and in the measurement-based setting a Hadamard is a
    # basis change absorbed into the Pauli being measured. Conjugating
    # the target's entangling measurement by H turns its Z into X,
    # which is exactly what this default encodes.
    entangling_bases: tuple[str, str, str] = ("Z", "Z", "X")

    # Conditional Clifford corrections after the three entangling
    # measurements (CZ-type, pairwise among the three operands).
    corrections: int = 3

    # ---------------------------------------------------------------
    # ANALYTIC ERROR TERMS  (not expressible in BicycleISA)
    # ---------------------------------------------------------------
    # Error contributions that are real but cannot be emitted as
    # instructions, because run_numerics only charges error for
    # TGate/Automorphism/Measure/JointMeasure. They accumulate into
    # EmitResult.analytic_error and must be added to run_numerics'
    # `total_error` column afterwards -- see end_to_end_error().

    # P_factory: logical error of ONE |CCZ> state at the factory's own
    # output, BEFORE delivery into a module. Delivery is already
    # covered by the relayed JointMeasures below, so setting this does
    # NOT double-count.
    #
    # Leave at 0.0 to report a PLACEMENT-ATTRIBUTABLE figure (the
    # original behaviour of this emitter); set it for an end-to-end
    # one. Anchor points, both at the factory output:
    #
    #   Gidney-Fowler CCZ factory,     p=1e-3   ~5e-11
    #   2|T>->|CCZ> from cultivation,  p=1e-3   ~1e-9
    #
    # P_factory is a DESIGN VARIABLE, not a constant: the usual sizing
    # rule is P_factory <= 0.01 * eps_target / n_toffoli. Treat this
    # as a sweep parameter, not a number to look up once.
    p_factory: float = 0.0

    # |CCZ> states consumed per Toffoli. 1 for CCZ injection; 4 if you
    # ever switch to the 4-T decomposition.
    states_per_toffoli: int = 1

    # P_C: error of one inter-module JointMeasure. Used ONLY to price
    # the probabilistic Clifford corrections below. Must match the
    # `intermodule` value of whichever bicycle_numerics model you run
    # against, or the two halves of the total are inconsistent.
    #
    #   gross_1e-3   2.01e-3      two_gross_1e-3   1e-9
    #   gross_1e-4   4.81e-8      two_gross_1e-4   1e-18
    p_intermodule: float = 0.0

    # Charge the outcome-conditioned CZ corrections at their EXPECTED
    # cost (each fires with probability 1/2). Off by default: the
    # Litinski argument is that these commute to the end of the
    # circuit and are absorbed, so charging them is a pessimistic
    # upper bound rather than the baseline.
    count_corrections: bool = False





# ---------------------------------------------------------------
# Instruction constructors (exact serde shapes)
# ---------------------------------------------------------------

def init_t() -> Instruction:
    return "InitT"


def t_gate(basis: str = "Z", primed: bool = False, adjoint: bool = False) -> Instruction:
    if basis == "I":
        raise ValueError("TGate basis cannot be I.")
    return {"TGate": {"basis": basis, "primed": primed, "adjoint": adjoint}}


def automorphism(x: int, y: int) -> Instruction:
    return {"Automorphism": {"x": x % 6, "y": y % 6}}


def measure(p1: str = "Z", p7: str = "I") -> Instruction:
    if p1 == "I" and p7 == "I":
        raise ValueError("Measure requires at least one non-identity basis.")
    return {"Measure": {"p1": p1, "p7": p7}}


def joint_measure(p1: str = "Z", p7: str = "I") -> Instruction:
    if p1 == "I" and p7 == "I":
        raise ValueError("JointMeasure requires at least one non-identity basis.")
    return {"JointMeasure": {"p1": p1, "p7": p7}}


# ---------------------------------------------------------------
# Architecture introspection
# ---------------------------------------------------------------

@dataclass
class ArchIndex:
    modules: list[Hashable]
    block_of: dict[Hashable, int]
    factories: list[Hashable]
    magic_module: dict[Hashable, Hashable]   # factory -> attached module
    qubit_to_module: dict[Hashable, Hashable]
    distances: dict[Hashable, dict[Hashable, float]]

    def num_blocks(self) -> int:
        return len(self.modules)


def index_architecture(architecture: nx.Graph) -> ArchIndex:
    modules = [
        n for n, d in architecture.nodes(data=True) if d.get("kind") == "module"
    ]
    if not modules:
        raise ValueError("Architecture contains no module nodes.")

    # Stable ordering: by 'index' attribute when present, else by name.
    def sort_key(node):
        data = architecture.nodes[node]
        return (data.get("index", 0), str(node))

    modules = sorted(modules, key=sort_key)
    block_of = {module: i for i, module in enumerate(modules)}

    factories = [
        n for n, d in architecture.nodes(data=True) if d.get("kind") == "factory"
    ]
    if not factories:
        raise ValueError(
            "Architecture contains no factory nodes; CCZ injection needs "
            "at least one."
        )

    magic_module = {}
    for factory in factories:
        attached = architecture.nodes[factory].get("attached_module")
        if attached is None:
            neighbors = [
                n for n in architecture.neighbors(factory)
                if architecture.nodes[n].get("kind") == "module"
            ]
            if not neighbors:
                raise ValueError(f"Factory {factory!r} is not attached to a module.")
            attached = neighbors[0]
        magic_module[factory] = attached

    qubit_to_module = architecture.graph.get("qubit_to_module")
    if not qubit_to_module:
        qubit_to_module = {}
        for module in modules:
            for qubit in architecture.nodes[module].get("placed_qubits", []):
                qubit_to_module[qubit] = module
    if not qubit_to_module:
        raise ValueError(
            "No qubit placement found. Run a placement pass before emitting."
        )

    distances = dict(
        nx.all_pairs_dijkstra_path_length(architecture, weight="weight")
    )

    return ArchIndex(
        modules=modules,
        block_of=block_of,
        factories=factories,
        magic_module=magic_module,
        qubit_to_module=qubit_to_module,
        distances=distances,
    )


def _mst_cost(terminals, distances) -> float:
    terminals = set(terminals)
    if len(terminals) <= 1:
        return 0.0

    metric = nx.Graph()
    metric.add_nodes_from(terminals)
    for left, right in combinations(terminals, 2):
        metric.add_edge(left, right, weight=distances[left][right])

    return float(nx.minimum_spanning_tree(metric, weight="weight").size(weight="weight"))


def select_factory(operand_modules, arch: ArchIndex) -> Hashable:
    """
    Nearest factory by metric-closure MST over
    {operand modules} u {factory}. Ties broken by name for determinism.
    """
    candidates = []
    for factory in arch.factories:
        cost = _mst_cost(set(operand_modules) | {factory}, arch.distances)
        candidates.append((cost, str(factory), factory))
    return min(candidates)[2]


def module_path(source, target, architecture: nx.Graph) -> list[Hashable]:
    """
    Shortest path between two modules through MODULE nodes only
    (factories are leaves and must not be used as relays).
    """
    if source == target:
        return [source]

    module_only = architecture.subgraph(
        [
            n for n, d in architecture.nodes(data=True)
            if d.get("kind") == "module"
        ]
    )
    return nx.shortest_path(module_only, source, target, weight="weight")


# ---------------------------------------------------------------
# Emission
# ---------------------------------------------------------------

@dataclass
class EmitResult:
    lines: list[Line] = field(default_factory=list)
    numerics_qubits: int = 0
    num_blocks: int = 0

    # Integer INSTRUCTION counts. Everything here is emitted into the
    # JSONL and will therefore be re-counted by run_numerics.
    stats: dict[str, Any] = field(default_factory=dict)

    # Float ERROR contributions that are NOT emitted and NOT seen by
    # run_numerics. Kept strictly separate from `stats` so the two are
    # never silently mixed. Add these to run_numerics' `total_error`
    # via end_to_end_error().
    analytic_error: dict[str, float] = field(default_factory=dict)

    # Cumulative analytic error after Toffoli i, aligned with
    # run_numerics' one-row-per-line CSV output.
    analytic_error_cumulative: list[float] = field(default_factory=list)

    # |CCZ> states drawn from each factory, for throughput checks.
    states_per_factory: dict[str, int] = field(default_factory=dict)

    per_toffoli: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------
# Relay operator bookkeeping (item 4)
# ---------------------------------------------------------------

def conjugate_basis(basis: str, hop_index: int) -> str:
    """
    Basis carried by hop `hop_index` of a relay transporting a
    logical operator of type `basis`.

    Modelling choice, stated explicitly rather than left implicit:
    the two ENDPOINT hops carry the operator's own type, while
    INTERIOR hops of the chain carry the complementary type. This
    reflects that interior joint measurements must not measure the
    logical operator being transported (that would collapse it);
    they stitch the chain in the conjugate basis instead.

    This captures the operator bookkeeping at the granularity that
    matters for run_numerics -- which basis each JointMeasure
    carries -- without simulating the full Pauli frame. It is not a
    stabiliser-exact propagation.
    """
    complement = {"Z": "X", "X": "Z", "Y": "Y", "I": "I"}
    if hop_index == 0:
        return basis
    return complement[basis]


def _relay_ops(
    path: list[Hashable],
    arch: ArchIndex,
    basis: str = "Z",
    p7: str = "I",
) -> list[Operation]:
    """
    Emit one JointMeasure per adjacent hop along `path`, with the
    per-hop basis resolved by conjugate_basis (item 4).
    """
    ops: list[Operation] = []
    hops = list(zip(path, path[1:]))

    for hop_index, (left, right) in enumerate(hops):
        hop_basis = conjugate_basis(basis, hop_index)
        # Final hop returns to the operator's own basis.
        if hop_index == len(hops) - 1:
            hop_basis = basis

        ops.append(
            [
                (arch.block_of[left], joint_measure(hop_basis, p7)),
                (arch.block_of[right], joint_measure(hop_basis, p7)),
            ]
        )
    return ops


def emit(
    architecture: nx.Graph,
    toffoli_sequence: Sequence[tuple[Hashable, Hashable, Hashable]],
    params: CCZParams | None = None,
) -> EmitResult:
    """
    Emit the full bicycle instruction stream for `toffoli_sequence`.

    Protocol per Toffoli (CCZ gate teleportation):

      1. |CCZ> assumed present on magic qubits r0, r1, r2 in the
         magic module (black-box factory, not costed).
      2. CNOT each data operand with its OWN magic qubit r_i
         (item 1: r0/r1/r2 are tracked individually).
      3. Destructively measure r0, r1, r2 -- these are the outcomes
         the corrections are conditioned on (item 2).
      4. Pairwise CZ-type corrections on the data qubits.
      5. Syndrome cycle.

    One output LINE per Toffoli, so run_numerics reports one CSV row
    per Toffoli.
    """
    params = params or CCZParams()
    arch = index_architecture(architecture)

    result = EmitResult(
        num_blocks=arch.num_blocks(),
        numerics_qubits=arch.num_blocks() * 11,   # PathArchitecture::for_qubits
    )

    totals = {
        "automorphism": 0,
        "measure": 0,
        # Counts OPERATIONS, matching run_numerics' own
        # `joint_measurements` column (IsaCounter::add fires once per
        # op, on op[0], not once per touched block).
        "joint_measure": 0,
        # Subset of the above attributable to Clifford corrections.
        "correction_joint_measure": 0,
        "clifford_absorbed": 0,
    }

    # Analytic (non-emitted) error terms. Floats, kept out of `totals`.
    # Corrections are no longer here -- they are emitted, so
    # run_numerics charges them directly.
    analytic = {
        "factory": 0.0,
    }
    running_analytic = 0.0

    for op_index, (c1, c2, target) in enumerate(toffoli_sequence):
        for qubit in (c1, c2, target):
            if qubit not in arch.qubit_to_module:
                raise ValueError(f"Qubit {qubit!r} has no module placement.")

        operand_modules = [
            arch.qubit_to_module[c1],
            arch.qubit_to_module[c2],
            arch.qubit_to_module[target],
        ]

        factory = select_factory(set(operand_modules), arch)
        magic = arch.magic_module[factory]
        magic_block = arch.block_of[magic]

        # Item 1: the three magic qubits are named and tracked. They
        # all live in `magic`, so their block is magic_block, but they
        # are distinct logical qubits and each pairs with exactly one
        # operand.
        magic_qubits = (
            f"{factory}_r0",
            f"{factory}_r1",
            f"{factory}_r2",
        )

        line: Line = []
        hops_this_toffoli = 0

        # --- 1. |CCZ> delivered onto (r0, r1, r2) by `factory`. ---
        #
        # ***** THIS IS WHERE THE FACTORY COST GOES. *****
        #
        # The state arrives already carrying P_factory. That error is
        # attached to the state, not to any instruction, so it cannot
        # be emitted -- run_numerics has no ISA variant that would
        # carry it. It is accumulated analytically instead and folded
        # in by end_to_end_error() after the numerics run.
        #
        # This is charged ONCE PER STATE, not per operand: the three
        # PPMs below all consume one and the same |CCZ>.
        #
        # No double-counting with delivery: published CCZ factory
        # error rates are quoted at the factory's own output, and the
        # factory->module hand-off is the relayed JointMeasures below.
        analytic["factory"] += params.states_per_toffoli * params.p_factory
        result.states_per_factory[str(factory)] = (
            result.states_per_factory.get(str(factory), 0)
            + params.states_per_toffoli
        )

        # --- 2. Consume the |CCZ> state: one Pauli product
        #        measurement per operand, joint between the data
        #        qubit and ITS magic qubit r_i.
        #
        # This is the PBC formulation: magic-state consumption IS a
        # Pauli product measurement. The CNOT-then-measure picture
        # decomposes that same PPM into Clifford rotations plus a
        # measurement -- and per Litinski Fig. 5, those rotations are
        # pi/4 CLIFFORD rotations, which commute to the end of the
        # circuit and are absorbed into the final measurements. They
        # are therefore NOT emitted as instructions; only the
        # measurement is real.
        # 2b. Consume the |CCZ> state: THREE SEPARATE Pauli product
        #     measurements, one per operand.
        #
        #         Z(c1) (x) Z(r0),  Z(c2) (x) Z(r1),  X(t) (x) Z(r2)
        #
        #     Each is its own P(phi) and therefore gets its OWN GHZ,
        #     prepared and uncomputed independently (Tour de Gross
        #     Sec. 2: "To execute P(phi), a GHZ state is prepared via
        #     inter-module measurements on the pivot qubit").
        #
        #     Each of these measurements is 2-QUBIT -- one data qubit
        #     and one magic qubit -- so its support touches at most
        #     two modules and its GHZ is a PATH, not a branching tree.
        #     Sethi et al.'s spanning-tree cost applies to a single
        #     n-qubit rotation whose Pauli spans many modules at once
        #     (the per-T-gate Litinski case); it does NOT let these
        #     three independent measurements share corridor segments.
        for operand, magic_qubit, basis in zip(
            (c1, c2, target), magic_qubits, params.entangling_bases
        ):
            operand_module = arch.qubit_to_module[operand]

            # Bring the operand onto an addressable position (1/7).
            for _ in range(params.aut_per_pauli):
                line.append(
                    [(arch.block_of[operand_module], automorphism(1, 0))]
                )
                totals["automorphism"] += 1

            # Bring the magic qubit r_i onto an addressable position.
            for _ in range(params.aut_per_pauli):
                line.append([(magic_block, automorphism(1, 0))])
                totals["automorphism"] += 1

            path = module_path(magic, operand_module, architecture)
            participating = {arch.block_of[module] for module in path}

            # ----------------------------------------------------------
            # Prepare the initial state, over the GHZ SPAN ONLY.
            #
            # compile_measurement writes this loop as `(0..n)` -- every
            # block in the architecture -- but carries the comment
            # "TODO: Prepare state only on qubits that are in the range
            # of the measurement". The blanket version is an unoptimised
            # shortcut in that prototype, not a physical requirement:
            # the GHZ spans only first_nontrivial..last_nontrivial, so
            # blocks outside that range are never entangled and need
            # neither preparation nor uncomputation.
            #
            # A purely in-module PPM builds no GHZ at all, so it skips
            # both loops entirely.
            # ----------------------------------------------------------
            if len(path) > 1:
                for block in participating:
                    line.append([(block, measure("X", "I"))])
                    totals["measure"] += 1

            # ----------------------------------------------------------
            # GHZ preparation across the participating range.
            #
            # For a MEASUREMENT (unlike a rotation) the GHZ spans only
            # first_nontrivial..last_nontrivial, so a 2-qubit PPM costs
            # exactly one JointMeasure per adjacent hop between the
            # magic module and the operand's module.
            # ----------------------------------------------------------
            if len(path) > 1:
                relay = _relay_ops(path, arch, basis=basis)
                line.extend(relay)
                totals["joint_measure"] += len(relay)
                hops_this_toffoli += len(relay)

            # ----------------------------------------------------------
            # Uncompute the GHZ, over the span only.
            #
            # Y on the two blocks carrying the non-trivial Pauli (the
            # magic module and the operand's module), X on the blocks
            # the GHZ merely relayed through.
            # ----------------------------------------------------------
            if len(path) > 1:
                endpoints = {magic_block, arch.block_of[operand_module]}
                for block in participating:
                    pauli = "Y" if block in endpoints else "X"
                    line.append([(block, measure(pauli, "I"))])
                    totals["measure"] += 1
            else:
                # In-module PPM: a single native measurement, no GHZ.
                line.append([(magic_block, measure(basis, "Z"))])
                totals["measure"] += 1

        # --- 3. Readout of r0, r1, r2. ---
        # These outcomes drive the Clifford corrections. Basis follows
        # the entangling basis: Z for the controls, X for the
        # H-absorbed target.
        #
        # NOTE: emitted as Measure, NOT DestructiveZ/DestructiveX.
        # bicycle_numerics' IsaCounter::add accepts only TGate,
        # Automorphism, Measure and JointMeasure; every other variant
        # hits unreachable!(). DestructiveZ (displayed "measZ") panics
        # the numerics binary even though it is a legal BicycleISA
        # value.
        for basis in params.entangling_bases:
            line.append([(magic_block, measure(basis, "I"))])
            totals["measure"] += 1

        # --- 4. Conditional Clifford corrections: NOT EMITTED. ---
        #
        # The outcome-conditioned corrections are CZ-type Clifford
        # operations (pi/4 Pauli rotations). Per Litinski Sec. 2,
        # Clifford rotations are commuted to the end of the circuit
        # and absorbed into the final Pauli product measurements, so
        # they contribute no runtime instructions.
        #
        # They are COUNTED here (clifford_absorbed) rather than
        # emitted, so the bookkeeping stays visible.
        #
        # Caveat, stated rather than hidden: absorption is not
        # literally free. Commuting a Clifford past an anticommuting
        # rotation transforms its axis (P'_phi -> (iPP')_phi), so the
        # cost migrates into the Pauli weight of downstream
        # measurements rather than vanishing. For the in-module vs
        # cross-module BREAK-EVEN RATIO this is second order and
        # affects both sides alike; do NOT rely on it for absolute
        # instruction counts or end-to-end error figures.
        totals["clifford_absorbed"] += params.corrections

        # If you want the pessimistic upper bound instead of the
        # absorbed baseline, EMIT the corrections so run_numerics
        # charges their error itself.
        #
        # Each correction fires with probability 1/2, so its expected
        # relay cost is 0.5 * hops. That is not an emittable quantity
        # -- half a JointMeasure does not exist -- so it is rounded UP
        # per correction pair:
        #
        #     emitted_hops = ceil(0.5 * hops)
        #
        # Rounding up, not to nearest, keeps this an upper bound. Note
        # the 1-hop case: ceil(0.5) = 1, so an adjacent-module
        # correction gets no discount at all from its 1/2 firing
        # probability. That is the price of an integer instruction
        # stream, and it makes short corrections relatively more
        # expensive than long ones (a 6-hop correction pays 3/6, a
        # 1-hop correction pays 1/1).
        #
        # The emitted hops are a PREFIX of the correction's path, so
        # every one is between genuinely adjacent modules and
        # validate_emitted still passes.
        expected_corr_hops = 0.0
        emitted_corr_hops = 0
        if params.count_corrections:
            correction_pairs = [
                (c2, target),   # CZ(c2,t)^{m0}
                (c1, target),   # CZ(c1,t)^{m1}
                (c1, c2),       # CZ(c1,c2)^{m2}
            ]
            for qa, qb in correction_pairs:
                ma = arch.qubit_to_module[qa]
                mb = arch.qubit_to_module[qb]
                if ma == mb:
                    continue

                corr_path = module_path(ma, mb, architecture)
                hops = len(corr_path) - 1
                expected_corr_hops += 0.5 * hops

                n_emit = ceil(0.5 * hops)
                emitted_corr_hops += n_emit

                # CZ-type correction: Z-basis relay along the path.
                for left, right in list(zip(corr_path, corr_path[1:]))[:n_emit]:
                    line.append(
                        [
                            (arch.block_of[left], joint_measure("Z", "I")),
                            (arch.block_of[right], joint_measure("Z", "I")),
                        ]
                    )
                    totals["joint_measure"] += 1

        totals["correction_joint_measure"] += emitted_corr_hops

        # --- 5. Syndrome cycles are NOT emitted. ---
        # SyndromeCycle ("sc") also hits IsaCounter::add's
        # unreachable!() arm. Idling between operations is accounted
        # for by run_numerics itself, via per-block clock skew and
        # model.idling_error(), so no explicit instruction is needed
        # or permitted.

        result.lines.append(line)

        # Cumulative analytic error after this Toffoli. run_numerics'
        # `total_error` column is likewise CUMULATIVE (total_error is a
        # running sum living outside the per-line closure), so these
        # two are directly addable row by row.
        running_analytic = analytic["factory"]
        result.analytic_error_cumulative.append(running_analytic)

        result.per_toffoli.append(
            {
                "operation_index": op_index,
                "toffoli": (c1, c2, target),
                "factory": factory,
                "magic_module": magic,
                "magic_qubits": magic_qubits,
                "operand_modules": sorted(set(map(str, operand_modules))),
                "cross_module": len(set(operand_modules)) > 1,
                "relay_hops": hops_this_toffoli,
                "expected_correction_hops": expected_corr_hops,
                "emitted_correction_hops": emitted_corr_hops,
                "operations": len(line),
            }
        )

    result.stats = totals
    result.analytic_error = dict(analytic)
    result.analytic_error["total"] = sum(analytic.values())
    return result


# ---------------------------------------------------------------
# Validation and output
# ---------------------------------------------------------------

# bicycle_numerics' IsaCounter::add (crates/bicycle_numerics/src/lib.rs)
# matches exactly these four variants and calls unreachable!() on
# anything else. Emitting any other BicycleISA value -- even a legal
# one like DestructiveZ or SyndromeCycle -- panics the binary.
NUMERICS_ACCEPTED = {"TGate", "Automorphism", "Measure", "JointMeasure"}


def validate_emitted(result: EmitResult, architecture: nx.Graph) -> None:
    """
    Reimplements PathArchitecture::validate_operation's intent, but
    against the ARCHITECTURE GRAPH rather than block-index arithmetic,
    so it is correct for grids as well as chains.

    Raises ValueError on the first illegal Operation.
    """
    arch = index_architecture(architecture)
    block_to_module = {i: m for m, i in arch.block_of.items()}

    module_only = architecture.subgraph(
        [n for n, d in architecture.nodes(data=True) if d.get("kind") == "module"]
    )

    # Instruction-vocabulary check: catch anything run_numerics
    # would panic on, before it reaches the binary.
    for line_index, line in enumerate(result.lines):
        for op_index, op in enumerate(line):
            for _, instr in op:
                name = instr if isinstance(instr, str) else next(iter(instr))
                if name not in NUMERICS_ACCEPTED:
                    raise ValueError(
                        f"Line {line_index} op {op_index}: {name!r} is not "
                        "accepted by bicycle_numerics (IsaCounter::add only "
                        f"handles {sorted(NUMERICS_ACCEPTED)}; anything else "
                        "hits unreachable!())."
                    )

    for line_index, line in enumerate(result.lines):
        for op_index, op in enumerate(line):
            if len(op) == 1:
                continue

            if len(op) > 2:
                raise ValueError(
                    f"Line {line_index} op {op_index}: operations may touch at "
                    f"most 2 blocks (got {len(op)})."
                )

            if len(op) == 2:
                left_block, _ = op[0]
                right_block, _ = op[1]
                left_module = block_to_module[left_block]
                right_module = block_to_module[right_block]

                if not module_only.has_edge(left_module, right_module):
                    raise ValueError(
                        f"Line {line_index} op {op_index}: joint operation "
                        f"between non-adjacent modules {left_module!r} and "
                        f"{right_module!r}."
                    )


def write_jsonl(result: EmitResult, path: str) -> None:
    with open(path, "w") as handle:
        for line in result.lines:
            handle.write(json.dumps(line, separators=(",", ":")) + "\n")


def end_to_end_error(
    result: EmitResult,
    numerics_total_error: Sequence[float],
) -> list[float]:
    """
    Fold the analytic (non-emitted) error terms into run_numerics'
    reported `total_error` column.

    `numerics_total_error` is the `total_error` column of the CSV, in
    row order -- one row per Toffoli, cumulative. Returns the
    end-to-end cumulative error, same length.

    This is the ONLY place P_factory reaches a reported number. It
    cannot be done inside the JSONL: run_numerics charges error per
    INSTRUCTION, and a state that arrives pre-corrupted is not an
    instruction.

        rows = list(csv.DictReader(open("out.csv")))
        e2e = end_to_end_error(
            result, [float(r["total_error"]) for r in rows]
        )
    """
    if len(numerics_total_error) != len(result.analytic_error_cumulative):
        raise ValueError(
            f"Row count mismatch: numerics gave "
            f"{len(numerics_total_error)} rows, emitter recorded "
            f"{len(result.analytic_error_cumulative)}. The CSV must come "
            "from exactly this EmitResult."
        )
    return [
        numeric + analytic
        for numeric, analytic in zip(
            numerics_total_error, result.analytic_error_cumulative
        )
    ]


def summarize(result: EmitResult) -> str:
    lines = [
        f"blocks:            {result.num_blocks}",
        f"numerics qubits:   {result.numerics_qubits}  "
        f"(pass this to bicycle_numerics)",
        f"output lines:      {len(result.lines)}  (one per Toffoli)",
        "",
        "Instruction counts (emitted; run_numerics recounts these)",
    ]
    for key, value in result.stats.items():
        lines.append(f"  {key:18s} {value:6d}")

    cross = sum(1 for t in result.per_toffoli if t["cross_module"])
    hops = sum(t["relay_hops"] for t in result.per_toffoli)
    lines += [
        "",
        f"cross-module Toffolis: {cross}/{len(result.per_toffoli)}",
        f"total relay hops:      {hops}",
    ]

    if result.states_per_factory:
        lines += ["", "Magic states drawn per factory"]
        for name, count in sorted(result.states_per_factory.items()):
            lines.append(f"  {name:18s} {count:6d}")

    if result.analytic_error:
        lines += [
            "",
            "Analytic error (NOT emitted; add via end_to_end_error)",
        ]
        for key, value in result.analytic_error.items():
            lines.append(f"  {key:18s} {value:.3e}")
        if result.analytic_error.get("total", 0.0) == 0.0:
            lines.append(
                "  -> p_factory=0: this is a PLACEMENT-ATTRIBUTABLE run."
            )
    return "\n".join(lines)