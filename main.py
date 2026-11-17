from src.GraphInitialization.small_scale import build_small_graph
from src.Quantization.model_problem import model_problem
from src.Quantization.cost_hamiltonian import build_cost_hamiltonian
from src.Quantization.qaoa_ansatz_circuit import build_qaoa_ansatz
from src.QuantumHardwareOptimization.circuit_transpilation import transpile_circuit
from src.QiskitPrimitivesExecution.primitives_execution import run_qaoa_optimization, sample_final_distribution


def main():
    print("=== MaxCut - QAOA ===")

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


if __name__ == "__main__":
    main()
