import numpy as np
import rustworkx as rx

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit_addon_opt_mapper.applications import Maxcut
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import to_ising
from qiskit_addon_opt_mapper.problems.optimization_problem import OptimizationProblem


N_LARGE = 100


def build_large_graph(
    backend: IBMBackend,
) -> tuple[rx.PyGraph, Maxcut, OptimizationProblem, SparsePauliOp, float, QAOAAnsatz]:
    """Build the 100-node graph, cost Hamiltonian, and QAOA ansatz for the large-scale example.

    The graph is constructed from the backend's coupling map: each physical coupler
    between qubits i and j (both < N_LARGE) becomes an edge with weight 1.0.
    Since the coupling map lists both directions of each coupler, endpoints are
    sorted to avoid adding the same edge twice.

    The pipeline then follows the same three steps as the small-scale example:
      1. Model the problem with Maxcut → OptimizationProblem
      2. Convert to QUBO with OptimizationProblemToQubo
      3. Translate to Ising Hamiltonian with to_ising

    At utility scale, reps=1 (a single QAOA layer) is used to keep the circuit
    depth manageable on real hardware.

    Args:
        backend: IBM Quantum backend whose coupling map defines the graph topology.

    Returns:
        graph_100:           100-node rustworkx graph.
        maxcut_100:          Maxcut instance (used later to interpret results).
        maxcut_problem_100:  OptimizationProblem for the large graph.
        cost_hamiltonian_100: H_C as a SparsePauliOp.
        offset_100:          Scalar offset to recover the cut value.
        circuit_100:         QAOA ansatz circuit with reps=1, ready to transpile.
    """
    # Build the graph from the backend's coupling map
    graph_100 = rx.PyGraph()
    graph_100.add_nodes_from(np.arange(0, N_LARGE, 1))
    elist = set()
    for u, v in backend.coupling_map:
        # The coupling map lists both directions of each coupler, so sort
        # the endpoints to avoid adding the same edge twice.
        if u < N_LARGE and v < N_LARGE:
            elist.add((min(u, v), max(u, v), 1.0))
    graph_100.add_edges_from(sorted(elist))
    print(f"    Graph built: {len(graph_100)} nodes, {graph_100.num_edges()} edges")

    # Map to cost Hamiltonian
    maxcut_100 = Maxcut(graph_100)
    maxcut_problem_100 = maxcut_100.to_optimization_problem()
    qubo_100 = OptimizationProblemToQubo().convert(maxcut_problem_100)
    cost_hamiltonian_100, offset_100 = to_ising(qubo_100)
    print(f"    Cost Hamiltonian: {len(cost_hamiltonian_100)} Pauli terms | Offset: {offset_100}")

    # Build the QAOA ansatz with reps=1 for utility-scale hardware
    circuit_100 = QAOAAnsatz(cost_operator=cost_hamiltonian_100, reps=1)
    circuit_100.measure_all()
    print(f"    QAOA circuit parameters: {circuit_100.parameters}")

    return graph_100, maxcut_100, maxcut_problem_100, cost_hamiltonian_100, offset_100, circuit_100
