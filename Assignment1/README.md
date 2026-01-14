# 64-bit Pseudo-Random Number Generators: Implementation, Testing, and Visualization

This assignment studies the design and empirical evaluation of **64-bit pseudo-random number generators (PRNGs)**.  
Two generators are implemented and analyzed:

- **XorShift64**
- **Linear Congruential Generator (LCG64)**

The project consists of:
1. Implementing the generators
2. Statistically validating their parameters
3. Visualizing generator output
4. Applying the generators to a sorting task

---

## Directory Structure

<pre>
Assignment-1/
 ├── code/
 │ ├── pseudoRNG.py
 │ ├── test.py
 │ ├── generateIMG.py
 │ └── main.py
 ├── data/
 │ ├── xorShift64.txt
 │ └── lcg64.txt
 ├── Images/
 │ ├── xorShift64/
 │ └── lcg64/
 ├── RandomNoises/
 │ ├── xorShift64/
 │ └── lcg64/
 └── README.md
  
</pre>


---

## Code Overview

### `pseudoRNG.py`
Implements the core pseudo-random number generators.

#### XorShift64
- State update via XOR and bit-shift operations
- Arithmetic performed modulo \(2^{64}\)
- Parameterized by shifts `(a, b, c)`

#### LCG64
- State update:
  \[
  x_{n+1} = (a x_n + c) \bmod 2^{64}
  \]
- Output mixing via `(x >> 32) XOR x`
- Parameterized by `(a, c)`

Utility constants:
- `MASK64`: enforces 64-bit arithmetic
- `INV_2_64`: maps integers to `[0, 1)`

---

### `test.py`
Performs **statistical validation of RNG parameters**.

#### Tests implemented
- **k-dimensional equidistribution chi-square test**
- **Shannon entropy estimation** (byte-level)

#### Parameter search
- **XorShift64**:  
  \( 0 < a, b, c < 64 \), with distinct shifts
- **LCG64**:
  - \( a \equiv 1 \pmod{4} \)
  - \( c \) odd

Only parameter sets passing the chi-square threshold (p = 0.01) are accepted.

#### Output
- Valid XorShift64 parameters → `data/xorShift64/xorshift64.txt`
- Valid LCG64 parameters → `data/lcg64/lcg64.txt`

These files contain **only statistically acceptable parameters**.

---

### `generateIMG.py` (optional visualization)
Generates **grayscale noise images** from RNG output for visual inspection.

- Each image represents successive RNG outputs
- Pixel intensities are derived from the most significant 8 bits
- One image per parameter set

Output directories: This was .gitignore'd as containing 3400+ non relevant images
- `RandomNoises/xorShift64/`
- `RandomNoises/lcg64/`

This step is optional and used only to visually inspect structural artifacts.

---

### `main.py`
Demonstrates a **practical application** of the validated RNGs.

#### Pipeline
1. Selects a statistically valid RNG parameter set
2. Generates **1000 random 64-bit integers**
3. Sorts the values using **Quick Sort**
4. Visualizes the data:
   - **Barcode-style grayscale image**
   - **Bar-graph image**

Both **unsorted** and **sorted** outputs are visualized.

#### Sorting
- Functional Quick Sort implementation
- Average-case complexity: \( O(n \log n) \)

#### Output
Images are saved in:
- `Images/xorShift64/`
- `Images/lcg64/`

---

## Data Directory

The `data/` directory stores only **validated RNG parameters** obtained from statistical testing.

- `xorshift64.txt`: valid `(a, b, c)` triples
- `lcg64.txt`: valid `(a, c)` pairs

These values are reused across visualization and application code.

---

## Images and Visual Output

### `Images/`
Contains visualizations of sorted and unsorted random sequences:
- Barcode images
- Bar-graph images

### `RandomNoises/`
Contains raw noise images generated directly from RNG output to detect:
- Correlations
- Lattice structures
- Non-uniformity artifacts

---

## Summary

This assignment integrates:
- Low-level PRNG implementation
- Statistical testing of randomness
- Visual diagnostics
- Algorithmic application (sorting)

The codebase is modular, reproducible, and separates **generation**, **testing**, and **analysis** clearly.

