import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("output/figures")


def build_small_graph() -> rx.PyGraph:
    """Build and visualize the 5-node graph for the small-scale Max-Cut example."""
    n_small = 5

    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, n_small, 1))
    edge_list = [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 4, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]
    graph.add_edges_from(edge_list)
    draw_graph(graph, node_size=600, with_labels=True)
    plt.title("5-node graph — Max-Cut")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "small_graph.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Figure saved to: {output_path}")

    plt.show()

    return graph