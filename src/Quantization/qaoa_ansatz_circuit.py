from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


def build_qaoa_ansatz(cost_hamiltonian: SparsePauliOp, reps: int = 2) -> QAOAAnsatz:
    """Construye el circuito ansatz QAOA parametrizado.

    QAOA prepara soluciones candidatas aplicando capas alternadas de:
      - Operador de costo:   e^{-i * gamma_k * H_C}
      - Operador mezclador:  e^{-i * beta_k * H_m}

    Con reps=2 se generan 4 parámetros: beta[0], beta[1], gamma[0], gamma[1].

    Args:
        cost_hamiltonian: H_C como SparsePauliOp obtenido del paso anterior.
        reps: Número de capas QAOA (p). Por defecto 2.

    Returns:
        circuit: Circuito QAOA parametrizado con medición en todos los qubits.
    """
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    circuit.measure_all()

    return circuit