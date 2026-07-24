"""Shared inference configuration types and execution helpers."""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Protocol, TypeVar


_Result = TypeVar("_Result")


class InferenceConfiguration(Protocol):
    horizon: int
    message_passing_iterations: int
    convergence_tolerance: float
    policy_workers: int


def map_policies(
    function: Callable[[int], _Result],
    num_policies: int,
    policy_workers: int,
) -> List[_Result]:
    """Evaluate independent policies in deterministic order."""
    if policy_workers == 1 or num_policies <= 1:
        return [function(policy_index) for policy_index in range(num_policies)]

    workers = min(policy_workers, num_policies)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, range(num_policies)))
