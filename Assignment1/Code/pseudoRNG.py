import numpy as np

# Mask to enforce 64-bit unsigned integer arithmetic (mod 2^64)
MASK64 = (1 << 64) - 1

# Reciprocal of 2^64 (useful for mapping integers to floats in [0, 1))
INV_2_64 = 1.0 / (1 << 64)


class XorShift64:
    """
    64-bit xorshift pseudo-random number generator.

    The internal state is updated using a sequence of XOR and bit-shift
    operations. All arithmetic is performed modulo 2^64.
    """

    def __init__(self, seed: int, a: int, b: int, c: int):
        """
        Initialize the generator.

        Parameters
        ----------
        seed : int
            Initial 64-bit seed.
        a, b, c : int
            Shift parameters defining the xorshift recurrence.
        """
        self.x = seed & MASK64
        self.a = a
        self.b = b
        self.c = c

    def next(self) -> int:
        """
        Generate the next pseudo-random 64-bit integer.

        Returns
        -------
        int
            The next value in the xorshift sequence.
        """
        x = self.x
        x ^= (x << self.a) & MASK64
        x ^= (x >> self.b)
        x ^= (x << self.c) & MASK64
        self.x = x
        return x

    def produce(self, n: int) -> np.ndarray:
        """
        Generate a sequence of pseudo-random numbers.

        Parameters
        ----------
        n : int
            Number of values to generate.

        Returns
        -------
        numpy.ndarray
            Array of n unsigned 64-bit pseudo-random integers.
        """
        out = np.empty(n, dtype=np.uint64)
        for i in range(n):
            out[i] = self.next()
        return out


class LCG64:
    """
    64-bit linear congruential generator (LCG) with output mixing.

    State update: x_{n+1} = (a * x_n + c) mod 2^64
    Output: (x >> 32) XOR x
    """

    def __init__(self, seed: int, a: int, c: int):
        """
        Initialize the generator.

        Parameters
        ----------
        seed : int
            Initial 64-bit seed.
        a : int
            Multiplier.
        c : int
            Increment.
        """
        self.x = seed & MASK64
        self.a = a & MASK64
        self.c = c & MASK64

    def next(self) -> int:
        """
        Generate the next pseudo-random 64-bit integer.

        Returns
        -------
        int
            The next value produced by the LCG.
        """
        self.x = (self.a * self.x + self.c) & MASK64
        return (self.x >> 32) ^ self.x

    def produce(self, n: int) -> np.ndarray:
        """
        Generate a sequence of pseudo-random numbers.

        Parameters
        ----------
        n : int
            Number of values to generate.

        Returns
        -------
        numpy.ndarray
            Array of n unsigned 64-bit pseudo-random integers.
        """
        out = np.empty(n, dtype=np.uint64)
        for i in range(n):
            out[i] = self.next()
        return out
