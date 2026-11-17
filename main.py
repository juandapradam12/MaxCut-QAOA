import argparse
import numpy as np

from src.GraphInitialization.small_scale import build_small_graph
from src.GraphInitialization.large_scale import build_large_graph
from src.Quantization.model_problem import model_problem
from src.Quantization.cost_hamiltonian import build_cost_hamiltonian
from src.Quantization.qaoa_ansatz_circuit import build_qaoa_ansatz
from src.QuantumHardwareOptimization.circuit_transpilation import transpile_circuit, transpile_with_backend
from src.QiskitPrimitivesExecution.primitives_execution import (
    run_qaoa_optimization,
    sample_final_distribution,
    run_qaoa_optimization_large,
    sample_final_distribution_large,
)
from src.PostProcess.bitstring_decoding import decode_most_likely_bitstring, plot_result_distribution, plot_best_cut, to_bitstring
from src.PostProcess.brute_force import brute_force_max_cut
from src.PostProcess.large_scale_analysis import best_solution, samples_to_objective_values
from src.PostProcess.cdf_visualization import save_cdf_figure, plot_convergence


def run_small_scale():
    """Run the full QAOA workflow on a small 5-node Max-Cut instance."""
    print("\n[1] Inicializando grafo pequeño (5 nodos)...")
    graph = build_small_graph()
    print(f"    Grafo creado: {len(graph)} nodos, {graph.num_edges()} aristas")

    print("\n[2] Modelando el problema Max-Cut como OptimizationProblem...")
    maxcut, maxcut_problem = model_problem(graph)

    print("\n[3] Construyendo el Hamiltoniano de costo H_C...")
    cost_hamiltonian, offset = build_cost_hamiltonian(maxcut_problem)

    print("\n[4] Construyendo el circuito ansatz QAOA...")
    circuit = build_qaoa_ansatz(cost_hamiltonian, reps=2)
    print(f"    Parámetros del circuito: {circuit.parameters}")

    print("\n[5] Transpilando el circuito al hardware de IBM Quantum...")
    candidate_circuit, backend = transpile_circuit(circuit)

    print("\n[6] Running the QAOA optimization loop...")
    objective_func_vals, result = run_qaoa_optimization(candidate_circuit, cost_hamiltonian, backend)
    print(f"    Iterations: {len(objective_func_vals)} | Final cost: {result.fun:.6f}")

    print("\n[7] Sampling the final distribution with optimized parameters...")
    optimized_circuit, final_distribution_int, final_distribution_bin = sample_final_distribution(
        candidate_circuit, result, backend
    )

    print("\n[8] Decoding the most likely bitstring...")
    most_likely_bitstring, partition = decode_most_likely_bitstring(
        final_distribution_int, graph, maxcut
    )

    print("\n[9] Plotting the result distribution...")
    plot_result_distribution(final_distribution_bin)

    print("\n[10] Plotting the best cut on the graph...")
    plot_best_cut(graph, most_likely_bitstring)

    print("\n[11] Evaluating cut value...")
    cut_value = maxcut_problem.objective.evaluate(np.array(most_likely_bitstring))
    print(f"    The value of the cut is: {cut_value}")
    print(f"    Cut value implied by the optimized cost: {-(result.fun + offset):.6f}")

    print("\n[12] Comparing with brute-force classical optimum...")
    classical_best, classical_x = brute_force_max_cut(maxcut_problem)
    print(f"    Classical optimum (brute force): {classical_best}")
    print(f"    QAOA cut value:                  {cut_value}")


def run_large_scale():
    """Run the full QAOA workflow on a large 100-node utility-scale instance."""
    print("\n[1] Transpiling a small circuit to get the backend...")
    # We need the backend first to build the graph from the coupling map,
    # so we transpile a placeholder circuit to get the backend object.
    from src.Quantization.qaoa_ansatz_circuit import build_qaoa_ansatz
    from qiskit.quantum_info import SparsePauliOp
    _placeholder_h = SparsePauliOp(["ZZ"], coeffs=[0.5])
    _placeholder_circuit = build_qaoa_ansatz(_placeholder_h, reps=1)
    _, backend = transpile_circuit(_placeholder_circuit)

    print("\n[2] Building the 100-node graph, cost Hamiltonian, and QAOA ansatz...")
    graph_100, maxcut_100, maxcut_problem_100, cost_hamiltonian_100, offset_100, circuit_100 = (
        build_large_graph(backend)
    )

    print("\n[3] Transpiling the 100-node circuit for hardware...")
    candidate_circuit_100 = transpile_with_backend(circuit_100, backend)

    print("\n[4] Running the QAOA optimization loop (large scale)...")
    objective_func_vals_100, result_100 = run_qaoa_optimization_large(
        candidate_circuit_100, cost_hamiltonian_100, backend
    )
    print(f"    Iterations: {len(objective_func_vals_100)} | Final cost: {result_100.fun:.6f}")

    print("\n[5] Sampling the final distribution (large scale)...")
    optimized_circuit_100, final_distribution_100_int = sample_final_distribution_large(
        candidate_circuit_100, result_100, backend
    )

    print("\n[6] Post-processing: extracting the best cut...")
    best_sol_100 = best_solution(final_distribution_100_int, cost_hamiltonian_100)
    best_sol_bitstring_100 = to_bitstring(int(best_sol_100), len(graph_100))
    best_sol_bitstring_100.reverse()
    print(f"    Result bitstring: {best_sol_bitstring_100}")

    cut_value_100 = maxcut_problem_100.objective.evaluate(np.array(best_sol_bitstring_100))
    print(f"    The value of the cut is: {cut_value_100}")

    print("\n[7] Plotting convergence, best cut, and CDF...")
    plot_convergence(objective_func_vals_100, title="QAOA Convergence — Large Scale (100 nodes)")

    plot_best_cut(graph_100, best_sol_bitstring_100)

    result_dist_100 = samples_to_objective_values(final_distribution_100_int, cost_hamiltonian_100)
    save_cdf_figure(result_dist_100, backend.name)


def main():
    parser = argparse.ArgumentParser(description="MaxCut - QAOA")
    parser.add_argument(
        "scale",
        choices=["small", "large"],
        help="Workflow scale: 'small' (5-node) or 'large' (100-node utility-scale)",
    )
    args = parser.parse_args()

    print("=== MaxCut - QAOA ===")
    print(f"    Scale: {args.scale}")

    if args.scale == "small":
        run_small_scale()
    elif args.scale == "large":
        run_large_scale()


if __name__ == "__main__":
    main()
