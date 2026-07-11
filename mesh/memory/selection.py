"""
Facility location selection for memory management.

Pure math module — no IO, no async. Implements the submodular facility
location function and dynamic withholding-cost swapping algorithm.

The facility location function f(S) measures total "coverage" of an
embedding set. Each entry's value is its maximum similarity to any other
entry in the set. A diverse set has high f(S).

f(S) = Σ_{v ∈ S} max_{s ∈ S, s≠v} sim(v, s)

Self-similarity is excluded (diagonal zeroed) so that the function
measures cross-coverage between distinct entries, not just set size.

All selection decisions use this single function:
- Candidate marginal gain: w_m = f(S ∪ {m}) - f(S)
- Withholding cost per entry: w_a = f(S) - f(S \ {a})
- Swap if w_m > min(w_a)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def facility_location(embs: list[np.ndarray]) -> float:
    """
    Facility location objective: total coverage of the embedding set.

    f(S) = Σ_{v ∈ S} max_{s ∈ S} sim(v, s)

    With n embeddings, this is O(n²) similarity computations.
    For n=30, that's ~900 ops — microseconds.
    """
    if not embs:
        return 0.0
    n = len(embs)
    # Build similarity matrix once
    sim_matrix = _build_sim_matrix(embs)
    # f(S) = sum of max similarity for each element
    return float(np.sum(np.max(sim_matrix, axis=1)))


def compute_marginal_gain(
    embs: list[np.ndarray],
    candidate: np.ndarray,
    cached_f_s: float | None = None,
) -> float:
    """
    Compute the marginal gain of adding a candidate to the set.

    w_m = f(S ∪ {m}) - f(S)

    O(n) if cached_f_s is provided (only need to compute f(S ∪ {m})).
    O(n²) if cached_f_s is None (must compute both).
    """
    f_s = cached_f_s if cached_f_s is not None else facility_location(embs)
    f_s_with_m = facility_location(embs + [candidate])
    return f_s_with_m - f_s


def compute_withholding_costs(embs: list[np.ndarray]) -> tuple[list[float], float]:
    """
    Compute withholding costs for all entries in the set.

    w_a = f(S) - f(S \ {a}) for each a ∈ S

    Returns (costs, f_s) where f_s = f(S) (cached for reuse).

    O(n²) total — computes f(S) once, then f(S \ {a}) for each a.
    Each f(S \ {a}) is O((n-1)²), so total is O(n³) in theory,
    but with n=30 this is ~27,000 ops — still microseconds.
    """
    if not embs:
        return [], 0.0

    f_s = facility_location(embs)
    costs = []
    for i in range(len(embs)):
        without_i = embs[:i] + embs[i + 1:]
        f_without_i = facility_location(without_i)
        costs.append(f_s - f_without_i)
    return costs, f_s


def try_swap(
    embs: list[np.ndarray],
    candidate: np.ndarray,
    max_entries: int,
    cached_weights: list[float] | None = None,
    cached_f_s: float | None = None,
) -> tuple[bool, int | None, list[float], float]:
    """
    Full acceptance decision for a candidate memory entry.

    Returns:
        (accepted, evict_index, new_weights, new_f_s)

    - If store is not full: accept unconditionally, recompute weights.
    - If store is full: compare candidate gain against weakest entry.
      - Accept + swap if candidate gain > min withholding cost.
      - Reject otherwise (no recomputation needed).

    The returned weights and f_s should be cached by the caller
    for subsequent calls.
    """
    n = len(embs)

    # Cold start: accept unconditionally until full
    if n < max_entries:
        new_embs = embs + [candidate]
        new_weights, new_f_s = compute_withholding_costs(new_embs)
        return True, None, new_weights, new_f_s

    # Compute candidate's marginal gain
    f_s = cached_f_s if cached_f_s is not None else facility_location(embs)
    w_m = compute_marginal_gain(embs, candidate, cached_f_s=f_s)

    # Use cached weights or compute fresh
    if cached_weights is None:
        cached_weights, f_s = compute_withholding_costs(embs)

    # Compare against weakest entry
    min_idx = int(np.argmin(cached_weights))
    min_weight = cached_weights[min_idx]

    if w_m > min_weight:
        # Swap: evict weakest, insert candidate, recompute
        new_embs = embs[:min_idx] + embs[min_idx + 1:] + [candidate]
        new_weights, new_f_s = compute_withholding_costs(new_embs)
        logger.info(
            f"Memory swap: evicting index {min_idx} (w={min_weight:.4f}), "
            f"inserting candidate (w_m={w_m:.4f})"
        )
        return True, min_idx, new_weights, new_f_s

    # Reject — no changes
    logger.debug(
        f"Memory candidate rejected: w_m={w_m:.4f} <= min_w={min_weight:.4f}"
    )
    return False, None, cached_weights, f_s


def select_active_set(
    embs: list[np.ndarray],
    active_size: int,
) -> tuple[list[int], list[float], float]:
    """
    Greedy submodular batch selection of the best active_size entries.

    Given a pool of embeddings, selects the subset S of size min(active_size, n)
    that approximately maximizes f(S) via the standard greedy algorithm.

    Returns:
        (selected_indices, weights, f_s)
        - selected_indices: indices into the input list for the active set
        - weights: withholding costs for the selected set
        - f_s: f(S) for the selected set
    """
    n = len(embs)
    if n == 0:
        return [], [], 0.0

    k = min(active_size, n)

    if k == n:
        # Select all — just compute weights
        weights, f_s = compute_withholding_costs(embs)
        return list(range(n)), weights, f_s

    # Greedy: start with S=∅, add argmax marginal gain k times
    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_gain = -float("inf")

        # Current selected embeddings
        current_embs = [embs[i] for i in selected]
        current_f = facility_location(current_embs) if current_embs else 0.0

        for idx in remaining:
            candidate_embs = current_embs + [embs[idx]]
            f_with = facility_location(candidate_embs)
            gain = f_with - current_f
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        selected.append(best_idx)
        remaining.discard(best_idx)

    # Compute final weights for the selected set
    selected_embs = [embs[i] for i in selected]
    weights, f_s = compute_withholding_costs(selected_embs)
    return selected, weights, f_s


def lazy_greedy_fl(
    sim_matrix: np.ndarray,
    k: int,
) -> list[int]:
    """Lazy greedy maximization of monotone Facility Location.

    f(S) = Σ_{v ∈ V} max_{s ∈ S} sim(v, s)

    Uses the Minoux (1978) lazy evaluation: marginal gains only decrease
    under submodularity, so we maintain a max-heap of upper bounds and
    only recompute when an item reaches the top.

    Args:
        sim_matrix: (n × n) pairwise cosine similarity matrix (V × V).
        k: number of items to select.

    Returns:
        List of k selected indices into the ground set.
    """
    import heapq

    n = sim_matrix.shape[0]
    if n == 0:
        return []
    k = min(k, n)

    # best_sim[v] = max similarity from v to any selected item so far
    best_sim = np.zeros(n, dtype=np.float64)
    selected_mask = np.zeros(n, dtype=bool)
    selected: list[int] = []

    # Initial marginal gains: adding item s to empty S gives
    # Δ(s) = Σ_v max(sim(v,s), 0) = Σ_v sim(v,s) since best_sim=0
    # Use negative for max-heap via min-heap
    initial_gains = sim_matrix.sum(axis=0)
    heap: list[tuple[float, int, int]] = []  # (-gain, iteration_computed, index)
    for i in range(n):
        heapq.heappush(heap, (-initial_gains[i], 0, i))

    for step in range(1, k + 1):
        while heap:
            neg_gain, computed_at, idx = heapq.heappop(heap)
            if selected_mask[idx]:
                continue
            if computed_at == step:
                # Fresh computation at this step — accept
                selected.append(idx)
                selected_mask[idx] = True
                # Update best_sim for all ground set items
                col = sim_matrix[:, idx]
                best_sim = np.maximum(best_sim, col)
                break
            # Stale — recompute marginal gain
            # Δ(idx) = Σ_v max(sim(v, idx) - best_sim[v], 0)
            col = sim_matrix[:, idx]
            gain = float(np.sum(np.maximum(col - best_sim, 0.0)))
            heapq.heappush(heap, (-gain, step, idx))

    return selected


def _build_sim_matrix(embs: list[np.ndarray]) -> np.ndarray:
    """Build pairwise cosine similarity matrix."""
    n = len(embs)
    if n == 0:
        return np.array([])
    # Stack into matrix and normalize
    mat = np.stack(embs)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    normalized = mat / norms
    sim = normalized @ normalized.T
    # Zero self-similarity — without this, max sim for every entry
    # is 1.0 (itself), making f(S) degenerate to just counting entries.
    np.fill_diagonal(sim, 0.0)
    return sim
