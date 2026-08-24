"""
sweep_numerics.py

Sweep n = 13..127 and k = 3,4,5, emit bicycle instructions for each
configuration, run them through bicycle_numerics, and write two
tables -- one per noise model.

Usage
-----
    python sweep_numerics.py --numerics /path/to/bicycle_numerics

    # dry run: emit and count instructions, but don't invoke the binary
    python sweep_numerics.py --analytic-only

Outputs
-------
    results_gross_1e-3.csv
    results_gross_1e-4.csv
    results_raw.csv          (one row per n/k/model, all fields)

Each table has n as rows, k as columns, holding total_error.

Notes
-----
bicycle_numerics writes a CSV to stdout with one row per input line
(one per Toffoli here). The LAST row's total_error is the whole-circuit
figure, which is what gets tabulated.

total_error is a LINEAR SUM of per-instruction errors -- the expected
number of failures -- not a probability. It exceeds 1 for large n at
p=1e-3. The script also records

    p_fail = 1 - exp(-total_error)

which is the interpretable form (accurate to <0.1% versus the exact
product, since every individual p_i is tiny).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Applications.MCTdecomp.binary_tree_decomposition import (
    create_circuit,
    extract_toffoli_hyperedges,
)
from linear_mapping import linear_mapping_tree_baseline
from bicycle_emitter import emit, validate_emitted, write_jsonl


MODELS = ("gross_1e-3", "gross_1e-4")

# Table 2 rates, used for the analytic cross-check.
ANALYTIC = {
    "gross_1e-3": dict(jm=2.01e-3, m=1.11e-5, aut=2 * 4.01e-7),
    "gross_1e-4": dict(jm=4.81e-8, m=1.01e-9, aut=2 * 6.07e-14),
}


def build(n, k, module_capacity=11, factory_period=0, uncompute=0):
    """Place and emit one configuration. Returns (result, architecture)."""
    hyperedges, sequence = extract_toffoli_hyperedges(create_circuit(n))

    architecture = linear_mapping_tree_baseline(
        control_count=n,
        toffoli_hyperedge_dict=hyperedges,
        controls_per_module=k,
        module_capacity=module_capacity,
        factory_period=factory_period,
    )

    result = emit(architecture, sequence)
    validate_emitted(result, architecture)

    return result, architecture


def analytic_total_error(stats, model):
    rates = ANALYTIC[model]
    return (
        stats["joint_measure"] * rates["jm"]
        + stats["measure"] * rates["m"]
        + stats["automorphism"] * rates["aut"]
    )


def run_numerics(binary, jsonl_path, qubits, model):
    """
    Invoke bicycle_numerics and return the LAST row's total_error.

    Returns None if the binary fails, so one bad configuration does not
    abort the sweep.
    """
    with open(jsonl_path) as handle:
        try:
            completed = subprocess.run(
                [binary, str(qubits), model],
                stdin=handle,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"      numerics failed: {exc}", file=sys.stderr)
            return None

    if completed.returncode != 0:
        message = completed.stderr.strip().splitlines()
        tail = message[-1] if message else "(no stderr)"
        print(f"      numerics exited {completed.returncode}: {tail}",
              file=sys.stderr)
        return None

    rows = list(csv.DictReader(completed.stdout.splitlines()))
    if not rows:
        print("      numerics produced no rows", file=sys.stderr)
        return None

    return float(rows[-1]["total_error"])


def sweep(
    control_counts,
    k_values,
    numerics_binary=None,
    module_capacity=11,
    factory_period=0,
    uncompute=0,
):
    records = []

    for n in control_counts:
        for k in k_values:
            print(f"n={n:4d} k={k}", end="", flush=True)

            try:
                result, _ = build(
                    n, k, module_capacity, factory_period, uncompute
                )
            except Exception as exc:
                print(f"  PLACEMENT FAILED: {type(exc).__name__}")
                for model in MODELS:
                    records.append(
                        dict(n=n, k=k, model=model, status="placement_failed")
                    )
                continue

            stats = result.stats
            print(
                f"  blocks={result.num_blocks:3d}"
                f"  jm={stats['joint_measure']:6d}"
                f"  m={stats['measure']:6d}",
                end="",
                flush=True,
            )

            for model in MODELS:
                record = dict(
                    n=n,
                    k=k,
                    model=model,
                    status="ok",
                    blocks=result.num_blocks,
                    qubits=result.numerics_qubits,
                    toffolis=len(result.per_toffoli),
                    joint_measure=stats["joint_measure"],
                    measure=stats["measure"],
                    automorphism=stats["automorphism"],
                    cross_module=sum(
                        1 for t in result.per_toffoli if t["cross_module"]
                    ),
                    relay_hops=sum(
                        t["relay_hops"] for t in result.per_toffoli
                    ),
                    analytic_total_error=analytic_total_error(stats, model),
                )

                if numerics_binary is not None:
                    with tempfile.NamedTemporaryFile(
                        suffix=".jsonl", delete=False
                    ) as handle:
                        path = handle.name
                    try:
                        write_jsonl(result, path)
                        measured = run_numerics(
                            numerics_binary,
                            path,
                            result.numerics_qubits,
                            model,
                        )
                    finally:
                        os.unlink(path)

                    record["total_error"] = measured
                    if measured is None:
                        record["status"] = "numerics_failed"
                else:
                    record["total_error"] = record["analytic_total_error"]
                    record["status"] = "analytic"

                error = record["total_error"]
                record["p_fail"] = (
                    1 - math.exp(-error) if error is not None else None
                )

                records.append(record)

            print()

    return records


def write_raw(records, path="results_raw.csv"):
    fields = [
        "n", "k", "model", "status", "blocks", "qubits", "toffolis",
        "joint_measure", "measure", "automorphism", "cross_module",
        "relay_hops", "analytic_total_error", "total_error", "p_fail",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({f: record.get(f) for f in fields})
    print(f"wrote {path}")


def write_table(records, model, k_values, factory_period, path=None):
    """One table per model: rows are n, columns are k."""
    path = path or f"results_{model}_{factory_period}.csv"

    by_n = {}
    for record in records:
        if record["model"] != model:
            continue
        by_n.setdefault(record["n"], {})[record["k"]] = record

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["n"]
        for k in k_values:
            header += [f"k={k}_total_error", f"k={k}_p_fail"]
        writer.writerow(header)

        for n in sorted(by_n):
            row = [n]
            for k in k_values:
                record = by_n[n].get(k)
                if record is None or record.get("total_error") is None:
                    row += ["", ""]
                else:
                    row += [
                        f"{record['total_error']:.6g}",
                        f"{record['p_fail']:.6g}",
                    ]
            writer.writerow(row)

    print(f"wrote {path}")


def print_table(records, model, k_values):
    by_n = {}
    for record in records:
        if record["model"] != model:
            continue
        by_n.setdefault(record["n"], {})[record["k"]] = record

    print()
    print(f"=== {model} : total_error (expected failures) ===")
    header = f"{'n':>5} |" + "".join(f"{'k=' + str(k):>13}" for k in k_values)
    print(header)
    print("-" * len(header))

    for n in sorted(by_n):
        row = f"{n:5d} |"
        for k in k_values:
            record = by_n[n].get(k)
            if record is None or record.get("total_error") is None:
                row += f"{'--':>13}"
            else:
                row += f"{record['total_error']:13.4g}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep bicycle_numerics over n and k."
    )
    parser.add_argument(
        "--numerics",
        help="Path to the bicycle_numerics binary. Omit to compute "
             "Table 2 values analytically instead of invoking it.",
    )
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--n-min", type=int, default=5)
    parser.add_argument("--n-max", type=int, default=450)
    parser.add_argument(
        "--n-step", type=int, default=10,
        help="Step between control counts. Use a larger step for a "
             "quick pass; 1 gives every n in the range.",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[3, 4, 5],
    )
    parser.add_argument("--module-capacity", type=int, default=11)
    parser.add_argument("--factory-period", type=int, default=2)
    parser.add_argument("--uncompute", type=int, default=0)
    args = parser.parse_args()

    binary = None if args.analytic_only else args.numerics
    if binary is not None and not os.path.isfile(binary):
        parser.error(f"bicycle_numerics not found at {binary!r}")
    if binary is None:
        print("No binary given -- computing Table 2 values analytically.\n")

    control_counts = list(range(args.n_min, args.n_max + 1, args.n_step))

    records = sweep(
        control_counts,
        args.k,
        numerics_binary=binary,
        module_capacity=args.module_capacity,
        factory_period=args.factory_period,
        uncompute=args.uncompute,
    )

    write_raw(records)
    # for model in MODELS[1]:
    model = MODELS[1]
    write_table(records, model, args.k,args.factory_period)
    print_table(records, model, args.k)


if __name__ == "__main__":
    main()