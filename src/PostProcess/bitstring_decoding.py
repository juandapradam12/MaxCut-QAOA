import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
from pathlib import Path

from qiskit_addon_opt_mapper.applications import Maxcut

OUTPUT_DIR = Path("output/figures")


def to_bitstring(integer: int, num_bits: int) -> list[int]:
    """Convert an integer to a list of bits of fixed width.

    Args:
        integer:  Integer representation of the measured bitstring.
        num_bits: Total number of bits (number of qubits / graph nodes).

    Returns:
        List of ints (0 or 1) representing the bitstring, MSB first.
    """
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


def decode_most_likely_bitstring(
    final_distribution_int: dict[int, float],
    graph: rx.PyGraph,
    maxcut: Maxcut,
) -> tuple[list[int], list[list[int]]]:
    """Extract the most likely bitstring from the sampled distribution and decode the partition.

    The most likely bitstring is the one with the highest probability in the
    final distribution. It is then decoded into the two node sets of the
    Max-Cut partition using the Maxcut.interpret() method.

    Args:
        final_distribution_int: Probability distribution keyed by integer bitstrings.
        graph:                  rustworkx graph (used to determine number of nodes).
        maxcut:                 Maxcut instance (used to interpret the bitstring).

    Returns:
        most_likely_bitstring: Bitstring as a list of ints (one per node).
        partition:             Two node sets [[set_0], [set_1]] of the best cut found.
    """
    keys = list(final_distribution_int.keys())
    values = list(final_distribution_int.values())

    # Pick the bitstring with the highest probability
    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely, len(graph))
    most_likely_bitstring.reverse()

    print("Result bitstring:", most_likely_bitstring)

    # Decode into the two node sets of the partition
    partition = maxcut.interpret(np.array(most_likely_bitstring))
    print("Partition:", partition)

    return most_likely_bitstring, partition


def plot_result_distribution(final_distribution_bin: dict[str, float]) -> None:
    """Plot the final bitstring probability distribution, highlighting the top 4 most likely.

    The top 4 most probable bitstrings are highlighted in purple;
    the rest are shown in grey. The figure is saved to output/figures/.

    Args:
        final_distribution_bin: Probability distribution keyed by binary strings.
    """
    plt.rcParams.update({"font.size": 10})
    final_bits = final_distribution_bin
    values = np.abs(list(final_bits.values()))
    top_4_values = sorted(values, reverse=True)[:4]
    positions = []
    for value in top_4_values:
        positions.append(np.where(values == value)[0])

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(rotation=45)
    plt.title("Result Distribution")
    plt.xlabel("Bitstrings (reversed)")
    plt.ylabel("Probability")
    ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
    for p in positions:
        ax.get_children()[int(p[0])].set_color("tab:purple")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "result_distribution.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Figure saved to: {output_path}")

    plt.show()



def plot_best_cut(graph: rx.PyGraph, most_likely_bitstring: list[int]) -> None:
    """Plot the graph with nodes colored by their partition assignment.

    Nodes in set 0 are shown in grey, nodes in set 1 in purple.
    The figure is saved to output/figures/.

    Args:
        graph:                 rustworkx graph to visualize.
        most_likely_bitstring: Bitstring as a list of ints (one per node).
    """
    colors = ["tab:grey" if i == 0 else "tab:purple" for i in most_likely_bitstring]
    pos, _default_axes = rx.spring_layout(graph), plt.axes(frameon=True)
    rx.visualization.mpl_draw(
        graph, node_color=colors, node_size=100, alpha=0.8, pos=pos
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "best_cut.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"    Figure saved to: {output_path}")

    plt.show()