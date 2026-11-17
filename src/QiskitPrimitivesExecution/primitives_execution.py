import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.ibm_backend import IBMBackend

OUTPUT_DIR = Path("output/figures")


def run_qaoa_optimization(
    candidate_circuit: QuantumCircuit,
    cost_hamiltonian: SparsePauliOp,
    backend: IBMBackend,
) -> tuple[list[float], object]:
    """Run the QAOA optimization loop using the Estimator primitive inside a Session.

    The optimization loop runs inside an IBM Quantum Compute Session to keep
    the device reserved across iterations, reducing queue latency between calls.

    At each iteration, the Estimator evaluates <H_C> with the current parameters,
    and the classical COBYLA optimizer updates them until convergence.

    Initial angles:
      - beta  ∈ [0, π]   → midpoint: π/2
      - gamma ∈ [0, 2π]  → midpoint: π
    For harder problems, warm-starting from known good angles or transferring
    parameters from smaller instances is recommended.

    Error suppression options applied:
      - Dynamical decoupling (XY4 sequence)
      - Gate twirling with automatic number of randomizations

    Args:
        candidate_circuit: Transpiled QAOA circuit with assigned layout.
        cost_hamiltonian:  H_C as a SparsePauliOp.
        backend:           IBM Quantum backend returned by transpile_circuit().

    Returns:
        objective_func_vals: List of cost values recorded at each iteration.
        result:              Optimization result from scipy.optimize.minimize.
    """
    # QAOA doesn't prescribe principled default angles — any bounded choice
    # works as a warm start for problems this small. beta and gamma are
    # periodic (beta in [0, pi] and gamma in [0, 2*pi] modulo the underlying
    # Pauli-rotation periods), and pi/2 and pi are just midpoints of those
    # ranges. For harder problems you would typically warm start from known
    # good angles or transfer parameters from smaller instances.
    initial_gamma = np.pi
    initial_beta = np.pi / 2
    init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]

    objective_func_vals: list[float] = []

    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        # Transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost = results.data.evs

        objective_func_vals.append(cost)

        return cost

    # Note: Session requires a Premium Plan. With the Open Plan we use
    # job mode — each iteration is submitted as an independent job.
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 1000

    # Set simple error suppression/mitigation options
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"
    estimator.options.environment.job_tags = ["TUT_QAOA"]

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    print(result)

    plt.figure(figsize=(12, 6))
    plt.plot(objective_func_vals)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    return objective_func_vals, result


def sample_final_distribution(
    candidate_circuit: QuantumCircuit,
    result,
    backend: IBMBackend,
) -> tuple[QuantumCircuit, dict[int, float], dict[str, float]]:
    """Assign optimized parameters and sample the final distribution with the Sampler.

    After the COBYLA optimizer converges, this function:
      1. Assigns the optimal angles (result.x) to the transpiled circuit.
      2. Saves the optimized circuit diagram to output/figures/.
      3. Runs the Sampler with 10,000 shots to build the final bitstring distribution.

    Args:
        candidate_circuit: Transpiled QAOA circuit (parametrized).
        result:            Optimization result from run_qaoa_optimization().
        backend:           IBM Quantum backend.

    Returns:
        optimized_circuit:      Circuit with optimal parameters assigned.
        final_distribution_int: Probability distribution keyed by integer bitstrings.
        final_distribution_bin: Probability distribution keyed by binary strings.
    """
    # Assign the optimal angles found by COBYLA to the circuit
    optimized_circuit = candidate_circuit.assign_parameters(result.x)

    # Save the optimized circuit diagram
    fig = optimized_circuit.draw("mpl", fold=False, idle_wires=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "optimized_circuit.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure saved to: {output_path}")

    # Sample the final distribution using the Sampler primitive
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 10000

    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"
    sampler.options.environment.job_tags = ["TUT_QAOA"]

    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val / shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}

    print(f"    Sampled {len(counts_int)} unique bitstrings over {shots} shots.")

    return optimized_circuit, final_distribution_int, final_distribution_bin


def run_qaoa_optimization_large(
    candidate_circuit: QuantumCircuit,
    cost_hamiltonian: SparsePauliOp,
    backend: IBMBackend,
) -> tuple[list[float], object]:
    """Run the QAOA optimization loop for the large-scale (100-node) instance.

    Same structure as run_qaoa_optimization but with reps=1, so only 2
    parameters: [beta, gamma]. Uses job mode (Open Plan compatible).

    Args:
        candidate_circuit: Transpiled 100-node QAOA circuit.
        cost_hamiltonian:  H_C for the 100-node graph.
        backend:           IBM Quantum backend.

    Returns:
        objective_func_vals: Cost values recorded at each iteration.
        result:              Optimization result from scipy.optimize.minimize.
    """
    # reps=1 → only 2 parameters: [beta, gamma]
    initial_gamma = np.pi
    initial_beta = np.pi / 2
    init_params = [initial_beta, initial_gamma]

    objective_func_vals: list[float] = []

    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        results = job.result()[0]
        cost = results.data.evs
        objective_func_vals.append(cost)
        return cost

    # Job mode — compatible with Open Plan
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 1000
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"
    estimator.options.environment.job_tags = ["TUT_QAOA"]

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
    )
    print(result)

    return objective_func_vals, result


def sample_final_distribution_large(
    candidate_circuit: QuantumCircuit,
    result,
    backend: IBMBackend,
) -> tuple[QuantumCircuit, dict[int, float]]:
    """Assign optimized parameters and sample the final distribution for the large-scale instance.

    Same as sample_final_distribution but returns only the integer distribution,
    since the large-scale post-processing uses evaluate_sparse_pauli instead of
    binary string decoding.

    Args:
        candidate_circuit: Transpiled 100-node QAOA circuit.
        result:            Optimization result from run_qaoa_optimization_large().
        backend:           IBM Quantum backend.

    Returns:
        optimized_circuit_100:    Circuit with optimal parameters assigned.
        final_distribution_100_int: Probability distribution keyed by integer bitstrings.
    """
    # Assign the optimal angles to the circuit
    optimized_circuit_100 = candidate_circuit.assign_parameters(result.x)

    # Save the optimized circuit diagram
    fig = optimized_circuit_100.draw("mpl", fold=False, idle_wires=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "optimized_circuit_large.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure saved to: {output_path}")

    # Sample with the Sampler primitive
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 10000
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"
    sampler.options.environment.job_tags = ["TUT_QAOA"]

    pub = (optimized_circuit_100,)
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    shots = sum(counts_int.values())
    final_distribution_100_int = {key: val / shots for key, val in counts_int.items()}

    print(f"    Sampled {len(counts_int)} unique bitstrings over {shots} shots.")

    return optimized_circuit_100, final_distribution_100_int
