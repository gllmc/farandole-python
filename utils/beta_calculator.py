import math
from collections import defaultdict


def get_centered_binomial_counts(eta):
    counts = {}
    for i in range(2 * eta + 1):
        val = i - eta
        counts[val] = math.comb(2 * eta, i)
    return counts


def get_squared_distribution(counts):
    sq_counts = defaultdict(int)
    for val, count in counts.items():
        sq_counts[val**2] += count
    return sq_counts


def convolve_distributions(dist1, dist2):
    res = defaultdict(int)
    for v1, c1 in dist1.items():
        for v2, c2 in dist2.items():
            res[v1 + v2] += c1 * c2
    return res


def power_distribution(dist, power):
    res = {0: 1}
    base = dist
    p = power
    while p > 0:
        if p % 2 == 1:
            res = convolve_distributions(res, base)
        base = convolve_distributions(base, base)
        p //= 2
    return res


def find_optimal_beta(n, k, eta, target_delta_log2):
    chi = get_centered_binomial_counts(eta)

    chi_sq = get_squared_distribution(chi)

    num_coeffs = k * n
    total_dist = power_distribution(chi_sq, num_coeffs)

    total_combinations = (2 ** (2 * eta)) ** num_coeffs

    sorted_vals = sorted(total_dist.keys())
    accumulated_count = 0

    for val in reversed(sorted_vals):
        accumulated_count += total_dist[val]
        prob_log2 = math.log2(accumulated_count) - math.log2(total_combinations)

        if prob_log2 > target_delta_log2:
            return val + 1


if __name__ == "__main__":
    n = 256
    k = 3
    eta = 1
    q = 7681
    target_delta_log2 = -256

    print(f"n={n}, k={k}, q={q}, eta={eta}, threshold=2^{target_delta_log2}")

    min_beta = find_optimal_beta(n, k, eta, target_delta_log2)

    max_p = (q - 1 - 2 * eta) / (4 * min_beta)

    print(f"We need β ⩾ {min_beta}, i.e. p ⩽ {max_p:.4f}.")
    if max_p >= 2 * eta + 1:
        print(f"OK: we can take p = {2 * eta + 1}.")
    else:
        print(f"NOK: p = {2 * eta + 1} is too large.")
