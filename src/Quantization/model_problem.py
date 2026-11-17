import rustworkx as rx
from qiskit_addon_opt_mapper.applications import Maxcut
from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem


def model_problem(graph: rx.PyGraph) -> tuple[Maxcut, OptimizationProblem]:
    """Model the Max-Cut problem as a Qiskit OptimizationProblem.

    The Maxcut class accepts a rustworkx graph directly and returns an
    OptimizationProblem whose objective is to maximize the cut value.

    Each edge (i, j) contributes the term:
        x_i + x_j - 2 * x_i * x_j
    to the objective, where x_i ∈ {0, 1} indicates which set node i belongs to.

    Args:
        graph: rustworkx graph with the nodes and edges of the problem.

    Returns:
        maxcut: Maxcut instance (used later to interpret results).
        maxcut_problem: OptimizationProblem ready to be converted to QUBO.
    """
    # Create the Maxcut instance from the graph.
    # Internally builds the objective: max sum_{(i,j)} x_i + x_j - 2*x_i*x_j
    maxcut = Maxcut(graph)

    # Convert to OptimizationProblem: defines binary variables x_0..x_n
    # and the maximization objective polynomial
    maxcut_problem = maxcut.to_optimization_problem()

    print(maxcut_problem.prettyprint())

    return maxcut, maxcut_problem