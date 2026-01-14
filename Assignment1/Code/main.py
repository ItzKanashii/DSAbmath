from pseudoRNG import XorShift64, LCG64, MASK64, INV_2_64
from time import time_ns
from PIL import Image
import numpy as np


def quick_sort(arr: list[int]) -> list[int]:
    """
    Sort a list of integers using the Quick Sort algorithm.

    This implementation uses a recursive, functional-style approach:
    - The last element is chosen as the pivot
    - Elements less than or equal to the pivot are placed in the left sublist
    - Elements greater than the pivot are placed in the right sublist
    - The sublists are recursively sorted and concatenated

    Parameters
    ----------
    arr : list[int]
        List of integers to be sorted.

    Returns
    -------
    list[int]
        New list containing the elements of `arr` in non-decreasing order.

    Notes
    -----
    - This implementation is not in-place and uses additional memory
    - Worst-case time complexity is O(n²) when the pivot choice is poor
    - Average-case time complexity is O(n log n)
    """
    n = len(arr)
    if n <= 1:
        return arr

    pivot = arr[-1]      # Choose last element as pivot
    L = []               # Elements <= pivot
    R = []               # Elements > pivot

    for i in range(n - 1):
        if arr[i] <= pivot:
            L.append(arr[i])
        else:
            R.append(arr[i])

    # Recursively sort partitions
    L = quick_sort(L)
    R = quick_sort(R)

    return L + [pivot] + R


def quick_sort_inplace(arr: list[int], low: int = 0, high: int | None = None) -> None:
    """
    Sort a list of integers in place using the Quick Sort algorithm.

    This implementation uses the Lomuto partition scheme:
    - The last element of the current subarray is chosen as the pivot
    - Elements less than or equal to the pivot are moved to the left
    - Elements greater than the pivot are moved to the right
    - The algorithm recursively sorts the subarrays in place

    Parameters
    ----------
    arr : list[int]
        List of integers to be sorted in place.
    low : int, optional
        Starting index of the subarray to sort.
    high : int or None, optional
        Ending index of the subarray to sort. If None, the full array is used.

    Returns
    -------
    None
        The input list is modified in place.

    Notes
    -----
    - Average-case time complexity: O(n log n)
    - Worst-case time complexity: O(n²)
    - Space complexity: O(log n) due to recursion stack
    """
    if high is None:
        high = len(arr) - 1

    if low >= high:
        return

    # Partition using Lomuto scheme
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # Place pivot in its correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    p = i + 1

    # Recursively sort subarrays
    quick_sort_inplace(arr, low, p - 1)
    quick_sort_inplace(arr, p + 1, high)


def generate_img(arr: list[int], output: str) -> None:
    """
    Generate a grayscale image from a sequence of integers.

    Each value in the input array is mapped to a grayscale intensity using
    the most significant 8 bits of the 64-bit integer. The resulting image
    visualizes the structure of the sequence as vertical grayscale stripes,
    which is useful for detecting patterns or non-uniformity.

    Parameters
    ----------
    arr : list[int]
        Sequence of integers to be visualized (interpreted as 64-bit values).
    output : str
        Path where the generated image will be saved.

    Notes
    -----
    - The image height is fixed at 200 pixels
    - The image width equals the length of the input array
    - Each column corresponds to one array element
    """
    arr = np.asarray(arr, dtype=np.uint64)

    h = 200
    w = arr.size

    # Image array (uint8 for grayscale values)
    img = np.empty((h, w), dtype=np.uint8)
    x: int
    for j, x in enumerate(arr):
        # Use the most significant 8 bits of the value
        img[:, j] = (int(x) * 256) >> 64

    # Save grayscale image
    Image.fromarray(img, mode="L").save(output)


def generateBarGraph(arr: list[int], output: str) -> None:
    """
    Generate a bar-graph image from a sequence of 64-bit integers.

    Each value in the array is mapped linearly to a bar height, where:
    - 0            maps to height 0
    - 2^64         maps to height 2^12 (= 4096 pixels)

    Bars are rendered as solid vertical rectangles with:
    - Fixed bar width of 16 pixels
    - Fixed gap of 2 pixels between bars
    - Constant image height of 4096 pixels

    No plotting or graphing libraries are used; the image is constructed
    directly at the pixel level.

    Parameters
    ----------
    arr : list[int]
        Sequence of integers (interpreted as 64-bit values).
    output : str
        Path where the generated bar-graph image will be saved.

    Notes
    -----
    - The image background is black
    - Bars are drawn in white
    - Taller bars correspond to larger values
    """
    # Image and bar layout parameters
    BAR_WIDTH = 16
    GAP = 4
    IMG_HEIGHT = 1 << 12          # 4096 pixels
    MAX_VAL = 1 << 64

    n = len(arr)

    # Compute image width with padding gap on both ends
    img_width = n * (BAR_WIDTH + GAP) + GAP

    # Initialize black image
    img = np.zeros((IMG_HEIGHT, img_width), dtype=np.uint8)

    for i, x in enumerate(arr):
        # Linearly scale value to bar height
        bar_height = int((x / MAX_VAL) * IMG_HEIGHT)

        # Horizontal span of the bar
        x_start = GAP + i * (BAR_WIDTH + GAP)
        x_end = x_start + BAR_WIDTH

        # Vertical span (bars grow upward from bottom)
        y_start = IMG_HEIGHT - bar_height
        y_end = IMG_HEIGHT

        # Draw bar
        img[y_start:y_end, x_start:x_end] = 255

    # Save bar graph image
    Image.fromarray(img, mode="L").save(output)


def mainXorShift64() -> None:
    """
    Demonstration pipeline for XorShift64 output, sorting, and visualization.

    This function performs the following steps:
    1. Initializes a XorShift64 generator with fixed (a, b, c) parameters
       and a time-based odd seed.
    2. Generates 1000 pseudo-random 64-bit integers.
    3. Prints the unsorted sequence and visualizes it as a grayscale image.
    4. Generates a bar-graph image of the unsorted sequence.
    5. Sorts the sequence using Quick Sort (functional version).
    6. Prints the sorted sequence and visualizes it as a grayscale image.
    7. Generates a bar-graph image of the sorted sequence.

    The generated images allow visual comparison of structure and distribution
    before and after sorting.
    """
    a, b, c = 41, 17, 47            # Chosen XorShift64 parameters
    seed = time_ns() | 1            # Odd seed to avoid degenerate state
    rng = XorShift64(seed, a, b, c)

    num = 1000                      # Number of random values to generate
    randomNum = list(rng.produce(num))

    # Output and visualize unsorted random numbers
    randomPath = "../Images/xorShift64/ranImgXorShift64.png"
    barRandomPath = "../Images/xorShift64/barRanImgXorShift64.png"
    for x in randomNum:
        print(f"{x}", end=" ")      # Optional processing before printing
    print()
    generate_img(randomNum, randomPath)
    generateBarGraph(randomNum, barRandomPath)

    # Sort the numbers using Quick Sort
    sortedNum = quick_sort(randomNum)

    # Output and visualize sorted numbers
    sortedPath = "../Images/xorShift64/sortedImg1.png"
    barSortedPath = "../Images/xorShift64/barSortedImg1.png"
    for x in sortedNum:
        print(f"{x}", end=" ")      # Optional processing before printing
    print()
    generate_img(sortedNum, sortedPath)
    generateBarGraph(sortedNum, barSortedPath)


def mainLCG64() -> None:
    """
    Demonstration pipeline for LCG64 output, sorting, and visualization.

    This function performs the following steps:
    1. Initializes an LCG64 generator with fixed (a, c) parameters and a
       time-based odd seed.
    2. Generates 1000 pseudo-random 64-bit integers.
    3. Prints the unsorted sequence and visualizes it as a grayscale image.
    4. Generates a bar-graph image of the unsorted sequence.
    5. Sorts the sequence using Quick Sort (functional version).
    6. Prints the sorted sequence and visualizes it as a grayscale image.
    7. Generates a bar-graph image of the sorted sequence.

    The before-and-after images provide a visual comparison of the structure
    and distribution of the LCG output.
    """
    a, c = 101, 31                  # Chosen LCG64 parameters
    seed = time_ns() | 1            # Odd seed to avoid degenerate state
    num = 1000                      # Number of random values to generate

    rng = LCG64(seed, a, c)
    randomNum = list(rng.produce(num))

    # Output and visualize unsorted random numbers
    randomPath = "../Images/lcg64/ranImgLCG64.png"
    barRandomPath = "../Images/lcg64/barRanImgLCG64.png"
    for x in randomNum:
        print(f"{x}", end=" ")      # Optional processing before printing
    print()
    generate_img(randomNum, randomPath)
    generateBarGraph(randomNum, barRandomPath)

    # Sort the numbers using Quick Sort
    sortedNum = quick_sort(randomNum)

    # Output and visualize sorted numbers
    sortedPath = "../Images/lcg64/sortedImg2.png"
    barSortedPath = "../Images/lcg64/barSortedImg2.png"
    for x in sortedNum:
        print(f"{x}", end=" ")      # Optional processing before printing
    print()
    generate_img(sortedNum, sortedPath)
    generateBarGraph(sortedNum, barSortedPath)


if __name__ == "__main__":
    """
    This block executes demonstration pipelines for both RNGs:
    - LCG64
    - XorShift64

    Each pipeline performs the following steps:
    1. Generate 1000 pseudo-random 64-bit integers
    2. Print the unsorted sequence to stdout
    3. Generate a grayscale image of the unsorted sequence
    4. Sort the sequence using Quick Sort
    5. Print the sorted sequence to stdout
    6. Generate a grayscale image of the sorted sequence

    Output Files
    ------------
    - ranImgLCG64.png      : Unsorted LCG64 output
    - sortedImg2.png       : Sorted LCG64 output
    - ranImgXorShift64.png : Unsorted XorShift64 output
    - sortedImg1.png       : Sorted XorShift64 output

    Notes
    -----
    - These are dry runs intended for demonstration and visualization,
      not exhaustive statistical testing.
    - Execution time is short (a few seconds), unlike the full parameter
      sweep tests which take several minutes.
    - RandomNoises are saved in the current working directory.
    """
    mainLCG64()
    mainXorShift64()
