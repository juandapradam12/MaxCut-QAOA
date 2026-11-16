import rustworkx as rx
from qiskit_addon_opt_mapper.applications import Maxcut
from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem


def model_problem(graph: rx.PyGraph) -> tuple[Maxcut, OptimizationProblem]:
    """Modela el problema Max-Cut como un OptimizationProblem de Qiskit.

    La clase Maxcut acepta directamente el grafo de rustworkx y devuelve un
    OptimizationProblem cuyo objetivo es maximizar el valor del corte.

    Cada arista (i, j) contribuye el término:
        x_i + x_j - 2 * x_i * x_j
    al objetivo, donde x_i ∈ {0, 1} indica a qué conjunto pertenece el nodo i.

    Args:
        graph: Grafo de rustworkx con los nodos y aristas del problema.

    Returns:
        maxcut: Instancia de Maxcut (usada luego para interpretar resultados).
        maxcut_problem: OptimizationProblem listo para convertir a QUBO.
    """
    # Crear la instancia de Maxcut a partir del grafo.
    # Internamente construye el objetivo: max sum_{(i,j)} x_i + x_j - 2*x_i*x_j
    maxcut = Maxcut(graph)

    # Convertir a OptimizationProblem: define variables binarias x_0..x_n
    # y el polinomio objetivo de maximización
    maxcut_problem = maxcut.to_optimization_problem()

    print(maxcut_problem.prettyprint())

    return maxcut, maxcut_problem