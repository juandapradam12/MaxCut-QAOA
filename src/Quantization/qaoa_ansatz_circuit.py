from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


def build_qaoa_ansatz(cost_hamiltonian: SparsePauliOp, reps: int = 2) -> QAOAAnsatz:
    """Build the parametrized QAOA ansatz circuit.

    QAOA prepares candidate solutions by applying alternating layers of:
      - Cost operator:   e^{-i * gamma_k * H_C}
      - Mixer operator:  e^{-i * beta_k * H_m}

    With reps=2, four parameters are generated: beta[0], beta[1], gamma[0], gamma[1].

    Args:
        cost_hamiltonian: H_C as a SparsePauliOp obtained from the previous step.
        reps: Number of QAOA layers (p). Defaults to 2.

    Returns:
        circuit: Parametrized QAOA circuit with measurements on all qubits.
    """
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    circuit.measure_all()

    return circuit