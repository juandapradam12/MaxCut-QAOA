import numpy as np
from collections import defaultdict

from qiskit.quantum_info import SparsePauliOp


# Precomputed parity lookup table: _PARITY[b] = +1 if popcount(b) is even, else -1.
# We use this to vectorize expectation-value evaluation across all Pauli terms.
_PARITY = np.array(
    [-1 if bin(i).count("1") % 2 else 1 for i in range(256)],
    dtype=np.complex128,
)


def evaluate_sparse_pauli(state: int, observable: SparsePauliOp) -> complex:
    """Expectation value of a SparsePauliOp on a single computational-basis state.

    For a Z-only observable (which QAOA cost Hamiltonians are, after the
    QUBO-to-Hamiltonian mapping), the eigenvalue of each Pauli term on a
    computational-basis state is simply (-1)**popcount(z_mask AND state),
    i.e., the parity of the bitwise-AND of the term's Z-support and the
    measured bitstring.

    This routine packs the Z-support of every Pauli term into bytes, ANDs
    them against the measured state in a single vectorized op, and looks up
    the parity in _PARITY. For a 100-qubit / ~hundreds-of-terms Hamiltonian
    over 10_000 samples, this is dramatically faster than calling
    SparsePauliOp.expectation_value per sample.

    Args:
        state:      Integer representation of the measured bitstring.
        observable: Cost Hamiltonian H_C as a SparsePauliOp.

    Returns:
        Expectation value of the observable on the given basis state.
    """
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(
        state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8
    )
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])


def best_solution(samples: dict[str, float], hamiltonian: SparsePauliOp) -> int:
    """Return the sampled bitstring (as int) with the lowest Hamiltonian cost.

    Args:
        samples:    Probability distribution keyed by integer bitstrings.
        hamiltonian: Cost Hamiltonian H_C as a SparsePauliOp.

    Returns:
        Integer bitstring with the minimum Hamiltonian expectation value.
    """
    min_cost = float("inf")
    min_sol = None
    for bit_str in samples.keys():
        candidate_sol = int(bit_str)
        fval = evaluate_sparse_pauli(candidate_sol, hamiltonian).real
        if fval <= min_cost:
            min_cost = fval
            min_sol = candidate_sol
    return min_sol


def samples_to_objective_values(
    samples: dict[str, float], hamiltonian: SparsePauliOp
) -> dict[float, float]:
    """Convert the sampled distribution to a distribution over objective function values.

    Groups bitstrings by their Hamiltonian cost and accumulates their probabilities,
    producing a probability distribution over cost values rather than bitstrings.

    Args:
        samples:     Probability distribution keyed by integer bitstrings.
        hamiltonian: Cost Hamiltonian H_C as a SparsePauliOp.

    Returns:
        Dictionary mapping objective function values to their total probability.
    """
    objective_values: dict[float, float] = defaultdict(float)
    for bit_str, prob in samples.items():
        candidate_sol = int(bit_str)
        fval = evaluate_sparse_pauli(candidate_sol, hamiltonian).real
        objective_values[fval] += prob
    return objective_values
