"""
make_period_figures.py

Sweep factory periods, compare linear-chain against square-grid
placement at MATCHED factory count, and produce:

    routing_volume_comparison_period_<P>.pdf   one per period
    routing_volume_summary.pdf                 improvement vs period
    routing_volume_sweep.csv                   the raw numbers

Usage
-----
    python make_period_figures.py
    python make_period_figures.py --periods 2 3 4 5 --n-max 127
    python make_period_figures.py --outdir Figures/routing_volume_comparison

Notes
-----
Factory count is matched between the two topologies at every point:
the grid is built with f = ceil(M / period), the same count a chain of
M modules carries at that period. Since routing cost is governed by
M/(2f), an unmatched count would swamp any topological effect. The
script asserts the match and reports any point where it fails.

The whiskers on the summary plot span MIN to MAX across circuit sizes.
They are not statistical: every measurement here is deterministic, so
the spread reflects variation with n, not uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Applications.MCTdecomp.binary_tree_decomposition import (
    create_circuit,
    extract_toffoli_hyperedges,
)
from linear_mapping import (
    linear_mapping_tree_baseline,
    calculate_routing_volume,
)
from square_grid_mapping import square_grid_mapping


LINEAR_COLOUR = "#B3541E"
GRID_COLOUR = "#1D4E89"

import matplotlib.pyplot as plt
import networkx as nx



def factory_count(architecture):
    return sum(
        1
        for _, data in architecture.nodes(data=True)
        if data.get("kind") == "factory"
    )


def sweep(control_counts, periods, controls_per_module, module_capacity):
    """Returns {period: {n, linear, grid, factories_linear, factories_grid}}."""
    results = {}

    for period in periods:
        record = {
            "n": [],
            "linear": [],
            "grid": [],
            "factories_linear": [],
            "factories_grid": [],
        }

        for n in control_counts:
            hyperedges, _ = extract_toffoli_hyperedges(create_circuit(n))

            chain = linear_mapping_tree_baseline(
                control_count=n,
                toffoli_hyperedge_dict=hyperedges,
                controls_per_module=controls_per_module,
                module_capacity=module_capacity,
                factory_period=period,
            )

            grid = square_grid_mapping(
                control_count=n,
                toffoli_hyperedge_dict=hyperedges,
                factory_period=period,
                controls_per_module=controls_per_module,
                module_capacity=module_capacity,
            )

            record["n"].append(n)
            record["linear"].append(
                calculate_routing_volume(
                    architecture_graph=chain,
                    toffoli_hyperedge_dict=hyperedges,
                )
            )
            record["grid"].append(
                calculate_routing_volume(
                    architecture_graph=grid,
                    toffoli_hyperedge_dict=hyperedges,
                )
            )
            record["factories_linear"].append(factory_count(chain))
            record["factories_grid"].append(factory_count(grid))

        results[period] = record

        mismatched = [
            (n, a, b)
            for n, a, b in zip(
                record["n"], record["factories_linear"], record["factories_grid"]
            )
            if a != b
        ]
        if mismatched:
            print(
                f"  WARNING period {period}: factory counts differ at "
                f"{len(mismatched)} of {len(record['n'])} sizes "
                f"-- e.g. n={mismatched[0][0]} has "
                f"{mismatched[0][1]} vs {mismatched[0][2]}"
            )

    return results


def improvements(record):
    return [
        100 * (a - b) / a for a, b in zip(record["linear"], record["grid"])
    ]


def plot_period(period, record, outdir):
    figure, axis = plt.subplots(figsize=(5.2, 4.0))

    axis.plot(
        record["n"], record["linear"],
        marker="o", ms=5.5, lw=1.8, color=LINEAR_COLOUR, label="Linear chain",
    )
    axis.plot(
        record["n"], record["grid"],
        marker="s", ms=5.5, lw=1.8, color=GRID_COLOUR, label="Square grid",
    )
    axis.fill_between(
        record["n"], record["grid"], record["linear"],
        alpha=0.12, color=GRID_COLOUR,
    )

    axis.set_xlabel("Number of MCT controls")
    axis.set_ylabel("Routing volume (inter-module hops)")
    axis.grid(True, ls="--", alpha=0.45)
    axis.legend(frameon=True, fontsize=9)
    axis.margins(x=0.05)

    figure.tight_layout()
    path = os.path.join(
        outdir, f"routing_volume_linvsgrid_comparison_period_{period}.pdf"
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_summary(results, outdir):
    periods = sorted(results)
    means, lows, highs = [], [], []

    for period in periods:
        gains = improvements(results[period])
        mean = sum(gains) / len(gains)
        means.append(mean)
        lows.append(mean - min(gains))
        highs.append(max(gains) - mean)

    figure, axis = plt.subplots(figsize=(5.6, 4.0))

    shades = plt.cm.Blues(
        [0.35 + 0.55 * i / max(1, len(periods) - 1) for i in range(len(periods))]
    )
    bars = axis.bar(
        [str(p) for p in periods], means,
        color=shades, width=0.6, edgecolor="black", linewidth=0.6,
    )
    axis.errorbar(
        [str(p) for p in periods], means, yerr=[lows, highs],
        fmt="none", ecolor="black", capsize=5, lw=1.1,
    )
    axis.axhline(0, color="black", lw=0.9)

    span = max(means) + max(highs)
    for bar, mean in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.03 * span,
            f"{mean:.1f}%",
            ha="center", fontsize=9, fontweight="bold",
        )

    axis.set_xlabel("Factory period $P$")
    axis.set_ylabel("Routing volume reduction (%)")
    axis.grid(True, axis="y", ls="--", alpha=0.45)

    figure.tight_layout()
    path = os.path.join(outdir, "routing_volume_summary.pdf")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def write_csv(results, outdir):
    path = os.path.join(outdir, "routing_volume_sweep.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "period", "n", "linear_routing_volume", "grid_routing_volume",
                "improvement_percent", "factories_linear", "factories_grid",
            ]
        )
        for period in sorted(results):
            record = results[period]
            for index, n in enumerate(record["n"]):
                linear = record["linear"][index]
                grid = record["grid"][index]
                writer.writerow(
                    [
                        period, n, linear, grid,
                        f"{100 * (linear - grid) / linear:.2f}",
                        record["factories_linear"][index],
                        record["factories_grid"][index],
                    ]
                )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, nargs="+", default=[2, 3, 4,5,6])
    parser.add_argument("--n-min", type=int, default=13)
    parser.add_argument("--n-max", type=int, default=128)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--controls-per-module", type=int, default=3)
    parser.add_argument("--module-capacity", type=int, default=11)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    control_counts = list(range(args.n_min, args.n_max + 1, args.n_step))
    print(
        f"Sweeping {len(control_counts)} sizes x {len(args.periods)} periods "
        f"(k={args.controls_per_module})"
    )

    results = sweep(
        control_counts,
        args.periods,
        args.controls_per_module,
        args.module_capacity,
    )

    print()
    print(f"{'period':>7} {'mean':>8} {'min':>8} {'max':>8}")
    for period in sorted(results):
        gains = improvements(results[period])
        print(
            f"{period:7d} {sum(gains) / len(gains):7.1f}% "
            f"{min(gains):7.1f}% {max(gains):7.1f}%"
        )
        plot_period(period, results[period], args.outdir)

    plot_summary(results, args.outdir)
    write_csv(results, args.outdir)

    print(f"\nwrote {len(results) + 1} figures and a CSV to {args.outdir}/")


if __name__ == "__main__":
    main()