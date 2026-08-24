#!/usr/bin/env python3
"""
plot_saturation.py

Saturation plot for a bicycle_numerics sweep: total error and failure
rate against circuit size n, one series per k.

Input CSV shape (as produced by the sweep):

    n,k=3_total_error,k=3_p_fail,k=4_total_error,k=4_p_fail,...

The `k=<K>_` prefix is discovered from the header, so any number of k
values works. `n` may be named `n`, `i`, or `qubits`.

WHY TWO FIGURES AND NOT ONE
--------------------------
total_error and p_fail are different scales with different meanings, so
they are written as two independent figures.

They also carry different information, which is the whole point of the
figure:

  * total_error is bicycle_numerics' UNION BOUND: a plain sum of
    per-instruction error rates (`total_error += instruction_error`).
    It grows without limit and crosses 1.0, past which it is no longer
    a probability and no longer bounds anything.

  * p_fail is that sum passed through the Poisson survival form
      p_fail = 1 - exp(-total_error)
    which saturates at 1. Verified against this dataset to < 2e-6.

So the left panel shows the accumulating cost and the right panel shows
what it means, and the interesting feature -- saturation -- is only
visible once you see them side by side.

Usage
-----
    python plot_saturation.py results_gross_1e3.csv
    python plot_saturation.py results_gross_1e3.csv -o fig/sat --smooth 5
    python plot_saturation.py results_gross_1e3.csv --check
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------
# Palette -- validated categorical slots 1-3 (all-pairs, light mode).
# Adding a 4th k needs a 4th slot; slots 4+ do NOT clear the
# all-pairs CVD floor, so SERIES falls back to distinguishing by
# linestyle as well as hue past three.
# ---------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
DASHES = [(None, None), (None, None), (None, None), (4, 2), (1, 1.6), (6, 2, 1, 2)]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# AXIS = "#097eb4"
AXIS =  "#e1e0d9"
AXIS_DOTTED = "#010507"
CRITICAL = "#d03b3b"


# ---------------------------------------------------------------
# Loading
# ---------------------------------------------------------------

N_ALIASES = ("n", "i", "qubits", "N")


def load(path: Path):
    """
    Returns (n_array, {k: {"total_error": arr, "p_fail": arr}}).
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows.")

    header = rows[0].keys()

    n_col = next((c for c in N_ALIASES if c in header), None)
    if n_col is None:
        raise ValueError(
            f"No n column found in {path}. Looked for {N_ALIASES}, "
            f"header is {list(header)}."
        )

    pattern = re.compile(r"^k=([\d.]+)_(total_error|p_fail)$")
    series: dict[str, dict[str, list[float]]] = {}
    for column in header:
        match = pattern.match(column)
        if match:
            k, quantity = match.groups()
            series.setdefault(k, {})[quantity] = [
                float(row[column]) for row in rows
            ]

    if not series:
        raise ValueError(
            f"No 'k=<K>_total_error' / 'k=<K>_p_fail' columns in {path}. "
            f"Header is {list(header)}."
        )

    n = np.array([float(row[n_col]) for row in rows])

    out = {}
    for k in sorted(series, key=float):
        block = series[k]
        if "total_error" not in block:
            raise ValueError(f"k={k} has p_fail but no total_error column.")
        entry = {"total_error": np.array(block["total_error"])}
        if "p_fail" in block:
            entry["p_fail"] = np.array(block["p_fail"])
        else:
            # Derive it if the sweep didn't write it.
            entry["p_fail"] = 1.0 - np.exp(-entry["total_error"])
            entry["derived"] = True
        out[k] = entry
    return n, out


def check_saturation_law(data) -> str:
    """
    Confirm p_fail == 1 - exp(-total_error) on this dataset, so the
    figure's claim about the two panels is checked rather than assumed.
    """
    lines = ["Saturation law check:  p_fail == 1 - exp(-total_error)"]
    worst_overall = 0.0
    for k, block in data.items():
        if block.get("derived"):
            lines.append(f"  k={k:<4} p_fail was DERIVED, not in the CSV")
            continue
        residual = np.abs(
            (1.0 - np.exp(-block["total_error"])) - block["p_fail"]
        )
        worst = float(residual.max())
        worst_overall = max(worst_overall, worst)
        lines.append(f"  k={k:<4} max |residual| = {worst:.3e}")
    lines.append(
        "  -> law HOLDS" if worst_overall < 1e-4
        else "  -> law DOES NOT HOLD; the right panel is not a transform "
             "of the left, relabel the figure"
    )
    return "\n".join(lines)


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    half = window // 2
    padded = np.pad(y, (half, half), mode="edge")
    return np.array(
        [np.median(padded[i:i + 2 * half + 1]) for i in range(len(y))]
    )


# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------

def style_axes(ax, xlabel, ylabel, title, subtitle):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=13)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=13)
    ax.set_title(title, color=INK, fontsize=17, loc="left", pad=18, weight="bold")
    ax.text(
        0.0, 1.015, subtitle, transform=ax.transAxes,
        color=MUTED, fontsize=9, va="bottom", ha="left",
    )



def _plot_series(ax, n, data, key, smooth: int, raw_alpha: float):
    """Plot all k-series for one quantity on a single axis."""
    for index, (k, block) in enumerate(data.items()):
        color = SERIES[index % len(SERIES)]
        dash = DASHES[index % len(DASHES)]
        label = f"k = {k}"
        y = block[key]

        if smooth > 1:
            ax.plot(
                n, y, color=color, linewidth=1.0, alpha=raw_alpha,
                zorder=2, solid_capstyle="round"
            )
            y_plot = rolling_median(y, smooth)
            ax.plot(
                n, y_plot, color=color, linewidth=2.0, zorder=3,
                label=label,
                dashes=dash if dash[0] else (None, None),
                solid_capstyle="round",
            )
        else:
            y_plot = y
            ax.plot(
                n, y_plot, color=color, linewidth=2.0, zorder=3,
                label=label,
                dashes=dash if dash[0] else (None, None),
                solid_capstyle="round",
            )

        ax.annotate(
            label,
            xy=(n[-1], y_plot[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9.5,
            va="center",
            ha="left",
            weight="bold",
            zorder=4,
        )


def _finalize_single_axis(fig, ax):
    """Apply legend, scientific-notation cleanup, and tight layout."""
    legend = ax.legend(
        frameon=False,
        fontsize=9.5,
        loc="lower right",
        labelcolor=INK_2,
        handlelength=1.8,
    )
    for label in legend.get_texts():
        label.set_color(INK_2)

    fig.canvas.draw()
    offset = ax.yaxis.get_offset_text()
    text = offset.get_text()
    if text:
        offset.set_visible(False)
        exponent = text.replace("\u2212", "-").split("e")[-1]
        ax.set_ylabel(
            f"{ax.get_ylabel()}   " rf"[$\times 10^{{{int(exponent)}}}$]",
            color=INK_2,
            fontsize=10,
        )

    fig.tight_layout()


def make_figures(
    n,
    data,
    smooth: int,
    raw_alpha: float,
    log: bool = False,
    headroom: float = 0.08,
):
    """Return separate total-error and failure-probability figures."""
    err_max = max(float(b["total_error"].max()) for b in data.values())
    fail_max = max(float(b["p_fail"].max()) for b in data.values())
    err_min = min(float(b["total_error"].min()) for b in data.values())
    fail_min = min(float(b["p_fail"].min()) for b in data.values())

    # Figure 1: total error
    fig_err, ax_err = plt.subplots(1, 1, figsize=(6.4, 5.0), facecolor=SURFACE)
    style_axes(
        ax_err,
        "# Toffoli controls",
        "Total error ",
        "",
        "run_numerics' running sum of per-instruction error;",
    )
    _plot_series(ax_err, n, data, "total_error", smooth, raw_alpha)

    if log:
        ax_err.set_yscale("log")
        ax_err.set_ylim(
            10 ** math.floor(math.log10(err_min)),
            10 ** math.ceil(math.log10(err_max)),
        )
    else:
        ax_err.set_ylim(0, err_max * (1 + headroom))

    if err_max >= 1.0:
        ax_err.axhline(1.0, color=AXIS_DOTTED, linewidth=1.4,
                       linestyle=(0, (5, 3)), zorder=1)
        ax_err.annotate(
            "E = 1 ",
            xy=(n[0], 1.0), xytext=(2, 5), textcoords="offset points",
            color=CRITICAL, fontsize=9, va="bottom", ha="left", zorder=4,
        )

        first_k = next(iter(data))
        crossing = np.nonzero(data[first_k]["total_error"] >= 1.0)[0]
        if crossing.size:
            n_cross = n[crossing[0]]
            ax_err.axvline(n_cross, color=AXIS_DOTTED, linewidth=1.0,
                           linestyle=(0, (2, 3)), zorder=1)
            
            n_cross = n[crossing[1]]
            ax_err.axvline(n_cross, color=AXIS_DOTTED, linewidth=1.0,
                            linestyle=(0, (2, 3)), zorder=1)
            
            n_cross = n[crossing[2]]
            ax_err.axvline(n_cross, color=AXIS_DOTTED, linewidth=1.0,
                            linestyle=(0, (2, 3)), zorder=1)
            ax_err.annotate(
                f"n = {n[crossing[0]]}, {n[crossing[1]]},{n[crossing[2]]}",
                xy=(n_cross, ax_err.get_ylim()[0]), xytext=(6, 6),
                textcoords="offset points", color=AXIS_DOTTED, fontsize=9,
                va="bottom", ha="left",
            )
    else:
        ax_err.annotate(
            f"max E = {err_max:.3g}  ·  far below the E = 1 breakdown",
            xy=(0.0, 1.0), xycoords="axes fraction", xytext=(2, -14),
            textcoords="offset points", color=MUTED, fontsize=9,
            va="top", ha="left", zorder=4,
        )

    ax_err.set_xlim(n[0], n[-1] + (n[-1] - n[0]) * 0.075)
    _finalize_single_axis(fig_err, ax_err)

    # Figure 2: failure probability
    fig_fail, ax_fail = plt.subplots(1, 1, figsize=(6.4, 5.0), facecolor=SURFACE)
    style_axes(
        ax_fail,
        "# Toffoli controls",
        "Failure probability",
        "",
        r"$p_{\mathrm{fail}} = 1 - e^{-E}$"
        if fail_max >= 0.5
        else r"$p_{\mathrm{fail}} = 1 - e^{-E}",
    )
    _plot_series(ax_fail, n, data, "p_fail", smooth, raw_alpha)

    if log:
        ax_fail.set_yscale("log")
        ax_fail.set_ylim(
            10 ** math.floor(math.log10(fail_min)),
            10 ** math.ceil(math.log10(fail_max)),
        )
    else:
        ax_fail.set_ylim(0, fail_max * (1 + headroom))

    # crossing_fails = np.nonzero(data[first_k]["p_fail"] >= 0.33)[0]

    if fail_max >= 0.5:
        ax_fail.axhline(1.0, color=AXIS_DOTTED, linewidth=1.4,
                               linestyle=(0, (5, 3)), zorder=1)
        ax_fail.axhline(0.333, color=AXIS_DOTTED, linewidth=1.4,
                        linestyle=(0, (5, 3)), zorder=1)
        ax_fail.annotate(
            "p = 1", xy=(n[0], 1.0), xytext=(2, 4),
            textcoords="offset points", color=AXIS_DOTTED, fontsize=9,
            va="bottom", ha="left",
        )
        # ax_fail.axvline(n[crossing_fails[0]], color=AXIS_DOTTED, linewidth=1.0,
        #                            linestyle=(0, (2, 3)), zorder=1)
        # ax_fail.axvline(n[crossing_fails[1]], color=AXIS_DOTTED, linewidth=1.0,
        #                                    linestyle=(0, (2, 3)), zorder=1)
        # ax_fail.axvline(n[crossing_fails[2]], color=AXIS_DOTTED, linewidth=1.0,
        #                                    linestyle=(0, (2, 3)), zorder=1)
        # ax_fail.annotate(
        #             f"n = {n[crossing_fails[0]]},{n[crossing_fails[1]]},{n[crossing_fails[2]]}", xy=(n[crossing_fails[0]], 0.1), xytext=(7, 4),
        #             textcoords="offset points", color=AXIS_DOTTED, fontsize=9,
        #             va="bottom", ha="left",
        #         )
        if not log:
            ax_fail.set_ylim(0, max(fail_max * (1 + headroom), 1.05))
    else:
        ax_fail.annotate(
            f"max p = {fail_max:.3g}  ·  p = 1 off-scale, no saturation",
            xy=(0.0, 1.0), xycoords="axes fraction", xytext=(2, -14),
            textcoords="offset points", color=MUTED, fontsize=9,
            va="top", ha="left", zorder=4,
        )

    ax_fail.set_xlim(n[0], n[-1] + (n[-1] - n[0]) * 0.075)
    _finalize_single_axis(fig_fail, ax_fail)

    return fig_err, fig_fail

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="sweep CSV")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output stem (default: alongside the CSV)")
    parser.add_argument("--smooth", type=int, default=1, metavar="W",
                        help="rolling-median window over n; raw stays as a "
                             "faint underlay (default 1 = off)")
    parser.add_argument("--raw-alpha", type=float, default=0.28,
                        help="opacity of the raw trace when --smooth is on")
    parser.add_argument("--formats", default="pdf",
                        help="comma-separated output formats")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--log", action="store_true",
                        help="log y-axis on both panels; use when the sweep "
                             "spans decades (p=1e-4, two-gross)")
    parser.add_argument("--headroom", type=float, default=0.08,
                        help="fractional padding above the data max on the "
                             "linear y-axis (default 0.08)")
    parser.add_argument("--check", action="store_true",
                        help="print the saturation-law residual check")
    args = parser.parse_args()

    n, data = load(args.csv)
    print(f"loaded {args.csv}: n = {n[0]:.0f}..{n[-1]:.0f} "
          f"({len(n)} rows), k = {', '.join(data)}")

    if args.check:
        print(check_saturation_law(data))

    fig_err, fig_fail = make_figures(
        n, data, args.smooth, args.raw_alpha,
        log=args.log, headroom=args.headroom
    )

    stem = args.out or args.csv.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats.split(","):
        fmt = fmt.strip()
        error_path = stem.parent / f"{stem.name}_total_error.{fmt}"
        fail_path = stem.parent / f"{stem.name}_failure_probability.{fmt}"

        fig_err.savefig(
            error_path, dpi=args.dpi, facecolor=SURFACE,
            bbox_inches="tight"
        )
        fig_fail.savefig(
            fail_path, dpi=args.dpi, facecolor=SURFACE,
            bbox_inches="tight"
        )

        print(f"wrote {error_path}")
        print(f"wrote {fail_path}")

    plt.close(fig_err)
    plt.close(fig_fail)


if __name__ == "__main__":
    main()