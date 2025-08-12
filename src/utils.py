from typing import Any, List

import numpy as np


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two dense vectors.
    Returns 0.0 if either vector has zero magnitude.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity as a float.
    """
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)

    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    """
    Split a list into consecutive sublists of given size.

    Args:
        lst: The input list to chunk.
        size: The desired chunk size.

    Returns:
        A list of sublists, each of length <= size.
    """
    return [lst[i: i + size] for i in range(0, len(lst), size)]
