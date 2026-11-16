from src.GraphInitialization.small_scale import build_small_graph
from src.Quantization.model_problem import model_problem
from src.Quantization.cost_hamiltonian import build_cost_hamiltonian
from src.Quantization.qaoa_ansatz_circuit import build_qaoa_ansatz


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


if __name__ == "__main__":
    main()
