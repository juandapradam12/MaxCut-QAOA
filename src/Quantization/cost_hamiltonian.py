from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import to_ising
from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem


def build_cost_hamiltonian(maxcut_problem: OptimizationProblem) -> tuple[SparsePauliOp, float]:
    """Construye el Hamiltoniano de costo H_C a partir del OptimizationProblem.

    Proceso en dos pasos:
    1. Convertir a QUBO con OptimizationProblemToQubo:
       - Transforma la maximización en minimización del valor negado del corte.
    2. Traducir el QUBO a Hamiltoniano de Ising con to_ising:
       - Aplica la sustitución x_i → (1 - Z_i) / 2
       - Devuelve H_C como SparsePauliOp y un escalar `offset` tal que:
           valor_QUBO = <H_C> + offset
         Por tanto, el valor del corte de cualquier bitstring es -(< H_C> + offset).

    Para max-cut sin pesos, los coeficientes lineales desaparecen (b_i = 0)
    y cada arista (i,j) contribuye el término (1/2) Z_i Z_j.

    Args:
        maxcut_problem: OptimizationProblem generado por Maxcut.to_optimization_problem().

    Returns:
        cost_hamiltonian: H_C como SparsePauliOp.
        offset: Escalar de desplazamiento para recuperar el valor del corte.
    """
    # Paso 1: convertir el problema de optimización a forma QUBO
    qubo = OptimizationProblemToQubo().convert(maxcut_problem)

    # Paso 2: traducir el QUBO al Hamiltoniano de Ising H_C
    cost_hamiltonian, offset = to_ising(qubo)

    print("Hamiltoniano de función de costo:", cost_hamiltonian)
    print("Offset:", offset)

    return cost_hamiltonian, offset