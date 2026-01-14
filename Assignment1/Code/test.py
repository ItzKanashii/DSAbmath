from pseudoRNG import XorShift64, LCG64, INV_2_64
from time import time_ns
import math
import numpy as np
from collections import Counter


def equidistribution_kd(samples: np.ndarray, k: int, m: int) -> float:
    """
    Compute a chi-square statistic to test k-dimensional equidistribution.

    The function forms overlapping k-tuples from the input sample sequence,
    partitions the unit interval into m equal bins per dimension, and compares
    observed frequencies of k-dimensional bins against the uniform expectation.

    Parameters
    ----------
    samples : numpy.ndarray
        One-dimensional array of samples assumed to lie in [0, 1).
    k : int
        Dimension of the equidistribution test (block length).
    m : int
        Number of bins per dimension.

    Returns
    -------
    float
        Chi-square statistic measuring deviation from k-dimensional uniformity.
    """
    # Number of k-dimensional blocks
    t = samples.size - k + 1

    # Form overlapping k-tuples (sliding window)
    blocks = np.lib.stride_tricks.sliding_window_view(samples, k)

    # Map samples to discrete bins {0, ..., m-1}
    bins = (blocks * m).astype(np.int64)
    bins[bins == m] = m - 1

    # Encode each k-tuple as a single integer index in base m
    idx = bins @ (m ** np.arange(k))

    # Count occurrences of each k-dimensional bin
    counts = np.bincount(idx, minlength=m ** k)

    # Expected count under uniform distribution
    expected = t / (m ** k)

    # Pearson chi-square statistic
    return float(np.sum((counts - expected) ** 2 / expected))


def testXorShift64():
    """
    Empirically test the k-dimensional equidistribution of XorShift64 generators.

    This function performs a parameter sweep over triples (a, b, c) of distinct
    small prime shift values and evaluates the quality of each resulting
    XorShift64 generator using a k-dimensional chi-square equidistribution test.

    For each parameter triple:
    - A XorShift64 generator is initialized with a time-based odd seed
    - N samples are generated and scaled to [0, 1)
    - The k-dimensional equidistribution chi-square statistic is computed
    - Parameter sets passing the chi-square threshold are written to a file
    - Failing parameter sets are reported to stdout

    Notes
    -----
    - The chi-square critical value corresponds to significance level p = 0.01
    - Only parameter triples with distinct shifts are tested
    - Running this function typically takes about 8–10 minutes due to the
      exhaustive parameter sweep and large sample size

    Output
    ------
    Writes passing (a, b, c) parameter triples and their chi-square values
    to the file "../data/xorShift64.txt".
    """
    seed = time_ns() | 1          # Odd seed to avoid trivial zero state
    k = 5                         # Dimension of equidistribution test
    m = 6                         # Number of bins per dimension
    N = 200_000                   # Number of samples
    CHI2_CRITICAL = 8_070         # Critical chi-square value (p = 0.01)

    PRIME = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    with open("../data/xorShift64.txt", "w") as f:
        for a in PRIME:
            for b in PRIME:
                if b == a:
                    continue
                for c in PRIME:
                    if c == a or c == b:
                        continue

                    rng = XorShift64(seed, a, b, c)
                    data = rng.produce(N).astype(np.float64) * INV_2_64
                    chi2 = equidistribution_kd(data, k, m)

                    if chi2 <= CHI2_CRITICAL:
                        f.write(f"{a = }, {b = }, {c = }, {chi2:.2f}\n")
                    else:
                        print(f"Failed! {a=} {b=} {c=}")


def testLCG64():
    """
    Empirically test the k-dimensional equidistribution of 64-bit LCG parameters.

    This function evaluates multiple (a, c) parameter pairs for a 64-bit linear
    congruential generator using a k-dimensional chi-square equidistribution test.

    Parameter constraints:
    - a ≡ 1 (mod 4), required for maximal period when modulus is 2^64
    - c is odd, required for maximal period
    - Only a small parameter range is tested due to the high computational cost

    For each valid (a, c) pair:
    - An LCG64 generator is initialized with a time-based odd seed
    - N samples are generated and scaled to [0, 1)
    - The k-dimensional equidistribution chi-square statistic is computed
    - Passing parameter pairs are written to a file
    - Failing parameter pairs are reported to stdout

    Notes
    -----
    - The chi-square critical value corresponds to significance level p = 0.01
    - Running this function typically takes about 3–5 minutes

    Output
    ------
    Writes passing (a, c) parameter pairs and their chi-square values
    to the file "../data/lcg64.txt".
    """
    seed = time_ns() | 1          # Odd seed to avoid degenerate states
    k = 5                         # Dimension of equidistribution test
    m = 6                         # Number of bins per dimension
    N = 200_000                   # Number of samples
    CHI2_CRITICAL = 8_070         # Critical chi-square value (p = 0.01)

    # Parameter ranges (kept small due to runtime cost)
    A_START, A_END = 1, 1 << 7    # a ≡ 1 (mod 4)
    C_START, C_END = 1, 1 << 7    # c odd

    with open("../data/lcg64.txt", "w") as f:
        for a in range(A_START, A_END, 4):
            for c in range(C_START, C_END, 2):
                if a == c:
                    continue

                rng = LCG64(seed, a, c)
                data = rng.produce(N).astype(np.float64) * INV_2_64
                chi2 = equidistribution_kd(data, k, m)

                if chi2 <= CHI2_CRITICAL:
                    f.write(f"{a = }, {c = }, {chi2:.2f}\n")
                else:
                    print(f"Failed {a=}, {c=}")


def rngToBytes(rng: XorShift64 | LCG64, n_samples: int) -> bytes:
    """
    Convert RNG output into a byte stream.

    This function draws successive 64-bit outputs from a pseudo-random number
    generator and serializes each value into 8 bytes using big-endian order.
    The resulting byte stream can be used for entropy estimation or other
    byte-level statistical tests.

    Parameters
    ----------
    rng : XorShift64 | LCG64
        Pseudo-random number generator instance.
    n_samples : int
        Number of 64-bit samples to extract.

    Returns
    -------
    bytes
        Byte sequence of length 8 * n_samples.
    """
    data = bytearray()
    for _ in range(n_samples):
        val = rng.next()
        data.extend(val.to_bytes(8, 'big'))
    return bytes(data)


def entropyCalc(data: bytes) -> float:
    """
    Compute the empirical Shannon entropy of a byte sequence.

    The entropy is computed over the 256 possible byte values. For a perfectly
    uniform byte distribution, the entropy is 8 bits per byte. Values closer
    to 8 indicate higher-quality randomness, while lower values indicate
    detectable bias or structure.

    Parameters
    ----------
    data : bytes
        Input byte sequence.

    Returns
    -------
    float
        Shannon entropy (in bits per byte).
    """
    n = len(data)
    freq = Counter(data)

    H = 0.0
    for count in freq.values():
        p = count / n
        H -= p * math.log2(p)

    return H


if __name__ == "__main__":
    """
    Entry point for running RNG quality tests.

    Uncomment the desired test function to evaluate the corresponding
    pseudo-random number generator. Execution time is measured and
    reported in minutes.
    """
    start = time_ns()

    # --- RNG parameter sweep tests ---
    # testXorShift64()   # Takes ~8–10 minutes
    # testLCG64()        # Takes ~3–5 minutes

    # --- Example entropy test ---
    # rng = XorShift64(time_ns() | 1, a=41, b=17, c=47)   # or fill with any value of a, b and c from the file
    # data = rngToBytes(rng, 10_000_000)                    # Takes ~40-60 seconds
    # print(f"Entropy: {entropyCalc(data):.6f} bits/byte")

    # rng = LCG64(time_ns() | 1, a=101, c=31)             # or fill with any value of a and c from the file
    # data = rngToBytes(rng, 10_000_000)
    # print(f"Entropy: {entropyCalc(data):.6f} bits/byte")

    end = time_ns()
    print(f"Took {((end - start) / 1e9) / 60:.2f} min")
