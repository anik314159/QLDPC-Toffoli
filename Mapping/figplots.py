import math
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.ticker import NullLocator


def draw_architecture(
    architecture,
    rows=None,
    cols=None,
    show_qubits=False,
    save_path=None,
):
    modules = [
        node
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "module"
    ]

    factories = [
        node
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "factory"
    ]

    modules = sorted(
        modules,
        key=lambda node: architecture.nodes[node].get("index", 0)
    )

    pos = {}

    # --------------------------------------------------
    # MODULE POSITIONS
    # --------------------------------------------------

    # Linear architecture
    if rows is None and cols is None:
        for i, module in enumerate(modules):
            pos[module] = (i, 0)

    # Grid architecture
    else:
        if cols is None:
            cols = math.ceil(len(modules) / rows)

        if rows is None:
            rows = math.ceil(len(modules) / cols)

        for i, module in enumerate(modules):
            row = i // cols
            col = i % cols

            # negative row so row 0 appears at the top
            pos[module] = (col, -row)

    # --------------------------------------------------
    # FACTORY POSITIONS
    # --------------------------------------------------

    for factory in factories:

        attached = architecture.nodes[factory].get(
            "attached_module"
        )

        if attached is None:
            neighbors = [
                n
                for n in architecture.neighbors(factory)
                if architecture.nodes[n].get("kind") == "module"
            ]

            if neighbors:
                attached = neighbors[0]

        if attached is not None and attached in pos:
            x, y = pos[attached]

            # Put factory slightly above its attached module
            pos[factory] = (x, y + 0.7)

    # --------------------------------------------------
    # LABELS
    # --------------------------------------------------

    labels = {}

    for node in modules:
        if show_qubits:
            qubits = architecture.nodes[node].get(
                "placed_qubits",
                []
            )

            labels[node] = (
                f"{node}\n"
                f"{qubits}"
            )
        else:
            labels[node] = str(node)

    for factory in factories:
        labels[factory] = str(factory)

    # --------------------------------------------------
    # DRAW
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    nx.draw_networkx_nodes(
        architecture,
        pos,
        nodelist=modules,
        node_size=1800,
    )

    nx.draw_networkx_nodes(
        architecture,
        pos,
        nodelist=factories,
        node_shape="s",
        node_size=1300,
    )

    nx.draw_networkx_edges(
        architecture,
        pos,
        width=2,
    )

    nx.draw_networkx_labels(
        architecture,
        pos,
        labels=labels,
        font_size=8,
    )

    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

    plt.show()


import matplotlib.pyplot as plt
import networkx as nx


def draw_square_grid(
    architecture,
    show_qubits=True,
    show_factories=True,
    save_path=None,
    figsize=(10, 7),
):
    """
    Draw a grid architecture produced by square_grid_mapping().

    Module positions are taken directly from the stored
    node attributes:
        row
        col

    Factories are drawn slightly above their attached modules.
    """

    modules = [
        node
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "module"
    ]

    factories = [
        node
        for node, data in architecture.nodes(data=True)
        if data.get("kind") == "factory"
    ]

    # ------------------------------------------------------
    # Positions
    # ------------------------------------------------------

    pos = {}

    for module in modules:
        data = architecture.nodes[module]

        row = data["row"]
        col = data["col"]

        # Negative row places row 0 at the top.
        pos[module] = (col, -row)

    if show_factories:
        for factory in factories:
            attached = architecture.nodes[factory][
                "attached_module"
            ]

            x, y = pos[attached]

            # Slight offset so factory is visibly separate
            # from its attached BB module.
            pos[factory] = (x + 0.22, y + 0.38)

    # ------------------------------------------------------
    # Labels
    # ------------------------------------------------------

    labels = {}

    for module in modules:
        data = architecture.nodes[module]

        if show_qubits:
            qubits = data.get("placed_qubits", [])

            labels[module] = (
                f"{module}\n"
                f"{qubits}"
            )
        else:
            labels[module] = module

    if show_factories:
        for factory in factories:
            labels[factory] = factory

    # ------------------------------------------------------
    # Separate module-module and factory-module edges
    # ------------------------------------------------------

    module_edges = []
    factory_edges = []

    for u, v in architecture.edges():

        u_kind = architecture.nodes[u].get("kind")
        v_kind = architecture.nodes[v].get("kind")

        if u_kind == "module" and v_kind == "module":
            module_edges.append((u, v))
        else:
            factory_edges.append((u, v))

    # ------------------------------------------------------
    # Draw
    # ------------------------------------------------------

    plt.figure(figsize=figsize)

    # Grid edges
    nx.draw_networkx_edges(
        architecture,
        pos,
        edgelist=module_edges,
        width=2,
    )

    # Factory attachment edges
    if show_factories:
        nx.draw_networkx_edges(
            architecture,
            pos,
            edgelist=factory_edges,
            width=1.5,
            style="dashed",
        )

    # BB modules
    nx.draw_networkx_nodes(
        architecture,
        pos,
        nodelist=modules,
        node_size=2200,
        node_shape="o",
    )

    # Factories
    if show_factories:
        nx.draw_networkx_nodes(
            architecture,
            pos,
            nodelist=factories,
            node_size=1100,
            node_shape="s",
        )

    nx.draw_networkx_labels(
        architecture,
        pos,
        labels=labels,
        font_size=8,
    )

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight",
        )

    plt.show()

def plot_routing_volume(
    control_counts,
    rv1,
    rv2,
    rv3,
    rv4,
    labels=(
        "Sequential first-fit",
        "3-MCT placement",
        "4-MCT placement",
        "5-MCT placement"
    ),
    log_x=False,
    save_path=None,
):
    """
    Plot routing volume versus number of MCT controls.
    """

    expected_length = len(control_counts)

    if not (
        len(rv1) == expected_length
        and len(rv2) == expected_length
        and len(rv3) == expected_length
        and len(rv4) == expected_length
    ):
        raise ValueError(
            "control_counts, rv1, rv2, rv3, and rv4 "
            "must have the same length."
        )

    figure, axis = plt.subplots(figsize=(10, 5))

    for values, marker, label in zip(
        (rv1, rv2, rv3, rv4),
        ("o", "o", "o", "o"),
        labels,
    ):
        axis.plot(
            control_counts,
            values,
            marker=marker,
            markersize=7,
            linewidth=2.0,
            label=label,
        )

    axis.set_xlabel("#-MCT controls", fontsize=17)
    axis.set_ylabel("# Inter-module connections", fontsize=17)
    axis.set_title(
        "Routing metric versus MCT size",
        fontsize=14
    )

    if log_x:
        axis.set_xscale("log", base=2)
        axis.xaxis.set_minor_locator(NullLocator())

    axis.set_xticks(control_counts[::2])
    axis.set_xticklabels(
        [str(count) for count in control_counts[::2]]
    )

    axis.tick_params(
        axis="both",
        labelsize=11
    )

    axis.margins(x=0.08)

    axis.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    # --------------------------------------------------------
    # More visible legend
    # --------------------------------------------------------
    legend = axis.legend(
        loc="upper left",
        fontsize=17,
        frameon=True,
        framealpha=1.0,
        fancybox=True,
        edgecolor="black",
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(1.2)

    figure.tight_layout()

    if save_path is not None:
        figure.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

    return figure, axis