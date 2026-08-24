import math
import networkx as nx

def linear_chain(
    modules=5,
    factory_period=0,
    factory_start=0,
):
    """
    Create a linear chain of computation modules with factories
    attached periodically as leaf nodes.

    Parameters
    ----------
    modules : int
        Number of computation modules.

    factory_period : int
        Factory attachment period.

        factory_period == 0:
            Attach one factory to M0.

        factory_period == 1:
            Attach one factory to every module.

        factory_period == 2:
            Attach factories to alternate modules.

        factory_period == 3:
            Attach factories to every third module.

    factory_start : int
        Index of the first module receiving a factory.

        For example, with factory_period=2:

            factory_start=0 -> M0, M2, M4, ...
            factory_start=1 -> M1, M3, M5, ...

    Returns
    -------
    nx.Graph
        Linear modular architecture.
    """

    if modules < 1:
        raise ValueError("modules must be at least 1.")

    if factory_period < 0:
        raise ValueError(
            "factory_period must be non-negative."
        )

    if not 0 <= factory_start < modules:
        raise ValueError(
            "factory_start must be a valid module index."
        )

    graph = nx.Graph()

    # Create computation-module nodes.
    for module_index in range(modules):
        module = f"M{module_index}"

        graph.add_node(
            module,
            kind="module",
            index=module_index,
            placed_qubits=[],
        )

    # Create the uninterrupted module chain.
    for module_index in range(modules - 1):
        left_module = f"M{module_index}"
        right_module = f"M{module_index + 1}"

        graph.add_edge(
            left_module,
            right_module,
            weight=1.0,
        )

    # Decide which modules receive factories.
    if factory_period == 0:
        attachment_indices = [0]
    else:
        attachment_indices = range(
            factory_start,
            modules,
            factory_period,
        )

    # Attach each factory to one computation module.
    for factory_index, module_index in enumerate(
        attachment_indices
    ):
        factory = f"F{factory_index}"
        module = f"M{module_index}"

        graph.add_node(
            factory,
            kind="factory",
            index=factory_index,
            attached_module=module,
        )

        graph.add_edge(
            factory,
            module,
            weight=1.0,
        )

    graph.graph["factory_period"] = factory_period
    graph.graph["factory_start"] = factory_start

    return graph
    
def sethi_long_grid(
    control_count,
    controls_per_module=4,
    factory_count=2,
    factory_placement="edge",
    exact_module_count=True,
):
   
    if control_count <= 0:
        raise ValueError("control_count must be positive.")

    if controls_per_module <= 0:
        raise ValueError("controls_per_module must be positive.")

    if factory_count <= 0:
        raise ValueError("factory_count must be positive.")

    if factory_placement not in {"edge", "distributed"}:
        raise ValueError(
            "factory_placement must be 'edge' or 'distributed'."
        )

    required_modules = math.ceil(control_count / controls_per_module)

    # The factory count fixes the short edge, giving the long grid.
    columns = min(factory_count, required_modules)
    rows = math.ceil(required_modules / columns)

    allocated_modules = (
        required_modules
        if exact_module_count
        else rows * columns
    )

    graph = nx.Graph()
    module_at = {}

    # Modules, row by row.
    for module_index in range(allocated_modules):
        row = module_index // columns
        col = module_index % columns

        module = f"M{row}_{col}"

        graph.add_node(
            module,
            kind="module",
            index=module_index,
            row=row,
            col=col,
            placed_qubits=[],
        )

        module_at[(row, col)] = module

    # Nearest-neighbour connections.
    for (row, col), module in module_at.items():
        right = (row, col + 1)
        below = (row + 1, col)

        if right in module_at:
            graph.add_edge(module, module_at[right], weight=1.0)

        if below in module_at:
            graph.add_edge(module, module_at[below], weight=1.0)

    # Choose the module each factory attaches to.
    attachment_points = []

    if factory_placement == "edge":
        for col in range(columns):
            if (0, col) in module_at:
                attachment_points.append((0, col))
    else:
        # One factory per column, centred in that column's occupied
        # span so its coverage radius is not wasted off the end.
        for col in range(columns):
            occupied = [
                row
                for row in range(rows)
                if (row, col) in module_at
            ]
            if not occupied:
                continue
            attachment_points.append((occupied[len(occupied) // 2], col))

    for factory_index, (row, col) in enumerate(attachment_points):
        factory = f"F{factory_index}"
        module = module_at[(row, col)]

        graph.add_node(
            factory,
            kind="factory",
            index=factory_index,
            attached_module=module,
        )

        graph.add_edge(factory, module, weight=1.0)

    graph.graph["required_modules"] = required_modules
    graph.graph["allocated_modules"] = allocated_modules
    graph.graph["rows"] = rows
    graph.graph["columns"] = columns
    graph.graph["factory_count"] = len(attachment_points)
    graph.graph["factory_placement"] = factory_placement
    graph.graph["controls_per_module"] = controls_per_module
    graph.graph["topology"] = "sethi_long_grid"

    return graph

# def grid_with_short_edge_factories(
#     num_controls,
#     factory_count=None,
#     exact_module_count=True,
# ):
#     """
#     Create a grid architecture and automatically attach factories
#     along its short edge.

#     Parameters
#     ----------
#     num_modules : int
#         Required number of computation modules.

#     factory_count : int | None
#         None:
#             Construct a compact near-square grid. The number of
#             factories equals the short-edge width.

#         Positive integer:
#             Construct a Sethi-style long grid with this many
#             factory columns.

#     exact_module_count : bool
#         True:
#             Create exactly num_modules computation modules.
#             The final row may be incomplete.

#         False:
#             Create the complete rows x columns rectangle.

#     Returns
#     -------
#     nx.Graph
#         Grid architecture with factories attached to the top edge.
#     """
#     num_modules = math.ceil(num_controls/4)
#     if num_modules <= 0:
#         raise ValueError("num_modules must be positive.")

#     if factory_count is None:
#         # Near-square grid whose vertical dimension is at least
#         # as large as its horizontal dimension.
#         columns = max(1, math.floor(math.sqrt(num_modules)))
#     else:
#         if factory_count <= 0:
#             raise ValueError("factory_count must be positive.")

#         # In the Sethi-style model, the factory count fixes
#         # the width of the short edge.
#         columns = min(factory_count, num_modules)

#     rows = math.ceil(num_modules / columns)

#     allocated_modules = (
#         num_modules
#         if exact_module_count
#         else rows * columns
#     )

#     graph = nx.Graph()
#     module_at = {}

#     # Create computation modules row by row.
#     for module_index in range(allocated_modules):
#         row = module_index // columns
#         col = module_index % columns

#         module = f"M{row}_{col}"

#         graph.add_node(
#             module,
#             kind="module",
#             index=module_index,
#             row=row,
#             col=col,
#             placed_qubits=[],
#         )

#         module_at[(row, col)] = module

#     # Add horizontal and vertical module connections.
#     for (row, col), module in module_at.items():
#         right = (row, col + 1)
#         below = (row + 1, col)

#         if right in module_at:
#             graph.add_edge(
#                 module,
#                 module_at[right],
#                 weight=1.0,
#             )

#         if below in module_at:
#             graph.add_edge(
#                 module,
#                 module_at[below],
#                 weight=1.0,
#             )

#     # The top row is the selected short boundary.
#     top_edge_modules = sorted(
#         (
#             module
#             for (row, col), module in module_at.items()
#             if row == 0
#         ),
#         key=lambda module: graph.nodes[module]["col"],
#     )

#     # Attach one factory to each top-edge module.
#     for factory_index, module in enumerate(top_edge_modules):
#         factory = f"F{factory_index}"

#         graph.add_node(
#             factory,
#             kind="factory",
#             index=factory_index,
#             attached_module=module,
#         )

#         graph.add_edge(
#             factory,
#             module,
#             weight=1.0,
#         )

#     graph.graph["required_modules"] = num_modules
#     graph.graph["allocated_modules"] = allocated_modules
#     graph.graph["rows"] = rows
#     graph.graph["columns"] = columns
#     graph.graph["factory_count"] = len(top_edge_modules)
#     graph.graph["topology"] = (
#         "compact_grid"
#         if factory_count is None
#         else "sethi_long_grid"
#     )

#     return graph



# def grid_with_edge_factories(rows=3, cols=4, factory_edge="left", factory_period=1):
#     """
#     Create a 2D grid of BB/Gross-code modules with factories attached
#     only along one edge of the grid.

#     Module nodes:
#         M0_0, M0_1, ..., M{rows-1}_{cols-1}

#     Factory nodes:
#         F0, F1, ...

#     Parameters
#     ----------
#     rows : int
#         Number of grid rows.

#     cols : int
#         Number of grid columns.

#     factory_edge : str
#         Which edge gets factories.
#         Options: "left", "right", "top", "bottom"

#     factory_period : int
#         If 0, no factories are added.
#         If 1, attach factory to every module on that edge.
#         If 2, attach factory to every second module on that edge, etc.

#     Returns
#     -------
#     G : nx.Graph
#         NetworkX graph with module and factory nodes.
#     """

#     if rows < 1 or cols < 1:
#         raise ValueError("rows and cols must be >= 1")

#     if factory_period < 0:
#         raise ValueError("factory_period must be >= 0")

#     if factory_edge not in {"left", "right", "top", "bottom"}:
#         raise ValueError("factory_edge must be one of: left, right, top, bottom")

#     G = nx.Graph()

#     # Add module nodes
#     for r in range(rows):
#         for c in range(cols):
#             node = f"M{r}_{c}"
#             G.add_node(
#                 node,
#                 kind="module",
#                 row=r,
#                 col=c
#             )

#     # Add grid edges between neighboring modules
#     for r in range(rows):
#         for c in range(cols):
#             node = f"M{r}_{c}"

#             # Horizontal edge
#             if c + 1 < cols:
#                 right = f"M{r}_{c+1}"
#                 G.add_edge(node, right)

#             # Vertical edge
#             if r + 1 < rows:
#                 down = f"M{r+1}_{c}"
#                 G.add_edge(node, down)

#     # No factories requested
#     if factory_period == 0:
#         return G

#     # Decide which edge modules receive factories
#     edge_modules = []

#     if factory_edge == "left":
#         for r in range(rows):
#             edge_modules.append(f"M{r}_0")

#     elif factory_edge == "right":
#         for r in range(rows):
#             edge_modules.append(f"M{r}_{cols-1}")

#     elif factory_edge == "top":
#         for c in range(cols):
#             edge_modules.append(f"M0_{c}")

#     elif factory_edge == "bottom":
#         for c in range(cols):
#             edge_modules.append(f"M{rows-1}_{c}")

#     # Attach factories along that edge
#     factory_count = 0

#     for idx, module in enumerate(edge_modules):
#         if idx % factory_period == 0:
#             factory = f"F{factory_count}"
#             factory_count += 1

#             G.add_node(
#                 factory,
#                 kind="factory",
#                 index=factory_count - 1,
#                 attached_to=module,
#                 edge=factory_edge
#             )

#             G.add_edge(factory, module)

#     return G


# G = linear_chain(modules=5, factory_period=1)
# print(list(G.nodes(data=True)))
# print(list(G.edges()))
    





        
