import argparse

DEFAULT_SHOTS = 1000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaxCut-QAOA runtime arguments")
    parser.add_argument("--n_qubits", type=int, default=5, help="Number of qubits")
    parser.add_argument("--n_shots", type=int, default=DEFAULT_SHOTS, help="Number of shots")
    parser.add_argument("--primitive", type=str, default="estimator", choices=["estimator", "sampler"])
    parser.add_argument("--mode", type=str, default="session", choices=["session", "job"])
    return parser


def _get_args() -> argparse.Namespace:
    parser = _build_parser()
    # parse_known_args avoids errors when called from Jupyter or pytest
    args, _ = parser.parse_known_args()
    return args


def get_n_qubits() -> int:
    return _get_args().n_qubits


def get_n_shots() -> int:
    return _get_args().n_shots


def get_primitive() -> str:
    return _get_args().primitive


def get_mode() -> str:
    return _get_args().mode
