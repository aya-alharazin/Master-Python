# 🎓 Complete NumPy Slicing & Broadcasting Masterclass
## From Zero to Expert - Deep Understanding with Comprehensive Examples

---

## 📚 Table of Contents

1. [Foundation: Understanding Memory & Arrays](#foundation)
2. [1D Array Slicing - Deep Dive](#1d-slicing)
3. [2D Array Slicing - Complete Guide](#2d-slicing)
4. [3D Arrays & Beyond](#3d-slicing)
5. [Views vs Copies - The Complete Truth](#views-copies)
6. [Slice Assignment - All Rules Explained](#slice-assignment)
7. [Broadcasting - From Basic to Advanced](#broadcasting)
8. [Advanced Slicing Techniques](#advanced-slicing)
9. [Common Patterns & Use Cases](#patterns)
10. [Pitfalls & Debugging](#pitfalls)
11. [Performance Considerations](#performance)
12. [Practice Problems - Progressive Difficulty](#practice)

---

<a name="foundation"></a>
## 🧱 CHAPTER 1: Foundation - Understanding Memory & Arrays

### 1.1 How Python Lists Work

```python
# Python list
py_list = [10, 20, 30, 40, 50]
```

**Memory structure:**
```
py_list → [ptr1, ptr2, ptr3, ptr4, ptr5]
           ↓     ↓     ↓     ↓     ↓
          [10]  [20]  [30]  [40]  [50]
```

- Each element is a separate Python object
- List stores pointers to objects
- Objects can be different types
- Inefficient for numerical computation

### 1.2 How NumPy Arrays Work

```python
import numpy as np
np_array = np.array([10, 20, 30, 40, 50])
```

**Memory structure:**
```
np_array → [10][20][30][40][50]  (contiguous memory block)
```

**Key properties:**
- All elements same type (homogeneous)
- Stored in contiguous memory
- Much faster for numerical operations
- Fixed size after creation

### 1.3 The Array Object Anatomy

```python
a = np.array([1, 2, 3, 4, 5])
print(a.shape)    # (5,)        - dimensions
print(a.dtype)    # int64       - data type
print(a.strides)  # (8,)        - bytes to next element
print(a.nbytes)   # 40          - total bytes
print(a.itemsize) # 8           - bytes per element
```

**What is stride?**
Stride tells you how many bytes to skip to get to the next element along each dimension.

```python
a = np.array([10, 20, 30, 40, 50], dtype=np.int64)
# Memory: [10][20][30][40][50]
#         0  8  16  24  32  (byte offsets)
# stride = 8 (each int64 is 8 bytes)
```

### 1.4 Why Understanding Memory Matters

This foundation is crucial because:
1. **Views share memory** - changing one changes the other
2. **Strides determine how slices work** - slicing just changes strides
3. **Broadcasting uses stride tricks** - no actual data copying

---

<a name="1d-slicing"></a>
## 🔍 CHAPTER 2: 1D Array Slicing - Deep Dive

### 2.1 The Complete Slicing Syntax

```python
array[start:stop:step]
```

**Rules:**
- `start`: inclusive (default: 0)
- `stop`: exclusive (default: len(array))
- `step`: increment (default: 1)

### 2.2 Basic Slicing - All Variations

```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

#### Omitting start
```python
a[:5]      # [0, 1, 2, 3, 4]
# Same as: a[0:5]
```

#### Omitting stop
```python
a[5:]      # [5, 6, 7, 8, 9]
# Same as: a[5:10]
```

#### Omitting both
```python
a[:]       # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Entire array - this is important for views!
```

#### Using step
```python
a[::2]     # [0, 2, 4, 6, 8]          - every 2nd element
a[1::2]    # [1, 3, 5, 7, 9]          - every 2nd, starting from 1
a[::3]     # [0, 3, 6, 9]             - every 3rd element
a[2:8:2]   # [2, 4, 6]                - from 2 to 8, step 2
```

### 2.3 Negative Indices - Complete Understanding

```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#  Indices:   0  1  2  3  4  5  6  7  8  9
# Negative:  -10 -9 -8 -7 -6 -5 -4 -3 -2 -1
```

#### Basic negative indexing
```python
a[-1]      # 9    - last element
a[-2]      # 8    - second-to-last
a[-3]      # 7    - third-to-last
```

#### Negative in slicing
```python
a[-3:]     # [7, 8, 9]                - last 3 elements
a[:-3]     # [0, 1, 2, 3, 4, 5, 6]    - all except last 3
a[-5:-2]   # [5, 6, 7]                - from -5 to -2 (exclusive)
a[-8:-3:2] # [2, 4, 6]                - with step
```

### 2.4 Negative Step - Reversing

```python
a[::-1]    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  - complete reverse
a[::-2]    # [9, 7, 5, 3, 1]                  - reverse, every 2nd
a[8:2:-1]  # [8, 7, 6, 5, 4, 3]               - backward from 8 to 3
a[8:2:-2]  # [8, 6, 4]                        - backward with step 2
```

**Important:** With negative step:
- Start should be greater than stop
- Iteration goes backward

### 2.5 Edge Cases & Tricky Examples

#### Example 1: Empty slices
```python
a[5:2]     # []  - start > stop with positive step
a[2:5:-1]  # []  - start < stop with negative step
```

#### Example 2: Out of bounds indices
```python
a[5:100]   # [5, 6, 7, 8, 9]  - NumPy doesn't error!
a[-100:3]  # [0, 1, 2]         - clips to valid range
```

#### Example 3: Step larger than array
```python
a[::100]   # [0]  - just first element
```

### 2.6 Mental Model for Slicing

Think of slicing as creating a "window" with these parameters:
1. **Where to start looking** (start)
2. **Where to stop looking** (stop)
3. **How big are your steps** (step)

```python
# Visual example
a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#    ^     ^     ^     ^     ^
#    |--2--|--2--|--2--|--2--|
# a[::2] creates a window that jumps by 2
```

### 2.7 What Actually Happens in Memory

```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
b = a[2:7:2]  # [2, 4, 6]
```

**Before slicing:**
```
a: memory: [0][1][2][3][4][5][6][7][8][9]
   offset:  0  8  16 24 32 40 48 56 64 72
   stride: 8
```

**After slicing:**
```
b: points to same memory, but:
   - starts at offset 16 (element 2)
   - stride becomes 16 (skip every other element)
   - length becomes 3
```

**No data is copied!** Just metadata changes.

---

<a name="2d-slicing"></a>
## 🎯 CHAPTER 3: 2D Array Slicing - Complete Guide

### 3.1 Understanding 2D Array Structure

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
```

**Mental model:**
```
       Column 0  Column 1  Column 2  Column 3
Row 0  [   1        2         3         4    ]
Row 1  [   5        6         7         8    ]
Row 2  [   9       10        11        12    ]
```

**Shape:** `(3, 4)` means 3 rows, 4 columns

**Memory layout (row-major, C-order):**
```
[1][2][3][4][5][6][7][8][9][10][11][12]
```
Stored row by row in a flat contiguous block.

### 3.2 Basic 2D Indexing

#### Single element access
```python
A[0, 0]     # 1   - row 0, col 0
A[1, 2]     # 7   - row 1, col 2
A[-1, -1]   # 12  - last row, last col
```

#### Row access
```python
A[0]        # [1, 2, 3, 4]       - first row
A[1]        # [5, 6, 7, 8]       - second row
A[-1]       # [9, 10, 11, 12]    - last row
```

**Note:** `A[0]` is shorthand for `A[0, :]`

#### Column access
```python
A[:, 0]     # [1, 5, 9]          - first column
A[:, 2]     # [3, 7, 11]         - third column
A[:, -1]    # [4, 8, 12]         - last column
```

### 3.3 2D Slicing Syntax

```python
A[row_slice, column_slice]
```

Both `row_slice` and `column_slice` follow 1D slicing rules: `[start:stop:step]`

### 3.4 Row Slicing

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
```

#### Select multiple rows
```python
A[0:2]      # [[1, 2, 3, 4],
            #  [5, 6, 7, 8]]
# Same as: A[0:2, :]
```

#### With step
```python
A[::2]      # [[1, 2, 3, 4],
            #  [9, 10, 11, 12]]
# Rows 0 and 2
```

#### Reverse rows
```python
A[::-1]     # [[9, 10, 11, 12],
            #  [5, 6, 7, 8],
            #  [1, 2, 3, 4]]
```

### 3.5 Column Slicing

```python
A[:, 0:2]   # [[1, 2],
            #  [5, 6],
            #  [9, 10]]
# All rows, columns 0-1
```

```python
A[:, 1:4:2] # [[2, 4],
            #  [6, 8],
            #  [10, 12]]
# All rows, columns 1 and 3
```

```python
A[:, ::-1]  # [[4, 3, 2, 1],
            #  [8, 7, 6, 5],
            #  [12, 11, 10, 9]]
# All rows, reverse columns
```

### 3.6 Submatrix Extraction

#### Example 1: Corner extraction
```python
# Top-left 2x2
A[0:2, 0:2]
# [[1, 2],
#  [5, 6]]

# Top-right 2x2
A[0:2, 2:4]
# [[3, 4],
#  [7, 8]]

# Bottom-left 2x2
A[1:3, 0:2]
# [[5, 6],
#  [9, 10]]

# Bottom-right 2x2
A[1:3, 2:4]
# [[7, 8],
#  [11, 12]]
```

#### Example 2: Center extraction
```python
# Middle 2x2
A[0:2, 1:3]
# [[2, 3],
#  [6, 7]]
```

#### Example 3: With steps
```python
# Every other row and column
A[::2, ::2]
# [[1, 3],
#  [9, 11]]
```

### 3.7 Combining Indexing and Slicing

```python
# Single row, slice of columns
A[1, 0:3]       # [5, 6, 7]

# Slice of rows, single column
A[0:2, 2]       # [3, 7]

# This is IMPORTANT for understanding shapes!
```

### 3.8 Shape Changes with Slicing

```python
A.shape         # (3, 4)

A[1].shape      # (4,)        - single row → 1D
A[1, :].shape   # (4,)        - explicit row slice → 1D
A[1:2].shape    # (1, 4)      - row slice → 2D!
A[1:2, :].shape # (1, 4)      - explicit → 2D

A[:, 1].shape   # (3,)        - single column → 1D
A[:, 1:2].shape # (3, 1)      - column slice → 2D!
```

**Critical insight:**
- Single index → reduces dimension
- Slice (even of length 1) → keeps dimension

### 3.9 Visual Understanding Exercise

```python
A = np.array([[10, 11, 12, 13],
              [14, 15, 16, 17],
              [18, 19, 20, 21],
              [22, 23, 24, 25]])
```

**Exercise:** Before running, predict the output:

```python
# 1.
A[1:3, 1:3]

# 2.
A[::2, 1::2]

# 3.
A[:2, -2:]

# 4.
A[-1::-1, ::-1]

# 5.
A[1::2, ::2]
```

**Answers:**
```python
# 1. [[15, 16], [19, 20]]
# 2. [[11, 13], [19, 21]]
# 3. [[12, 13], [16, 17]]
# 4. [[25, 24, 23, 22], [21, 20, 19, 18], [17, 16, 15, 14], [13, 12, 11, 10]]
# 5. [[14, 16], [22, 24]]
```

---

<a name="3d-slicing"></a>
## 📦 CHAPTER 4: 3D Arrays & Beyond

### 4.1 Understanding 3D Arrays

Think of 3D arrays as **stacks of 2D matrices**.

```python
# Create a 3D array: 2 matrices, each 3x4
B = np.array([[[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]],
              
              [[13, 14, 15, 16],
               [17, 18, 19, 20],
               [21, 22, 23, 24]]])

print(B.shape)  # (2, 3, 4)
```

**Interpretation:**
- Dimension 0: 2 "layers" or "depth"
- Dimension 1: 3 rows in each layer
- Dimension 2: 4 columns in each row

**Visual model:**
```
Layer 0:              Layer 1:
[1,  2,  3,  4]      [13, 14, 15, 16]
[5,  6,  7,  8]      [17, 18, 19, 20]
[9, 10, 11, 12]      [21, 22, 23, 24]
```

### 4.2 3D Slicing Syntax

```python
B[depth_slice, row_slice, col_slice]
```

### 4.3 3D Slicing Examples

#### Accessing entire layers
```python
B[0]           # First layer (2D array)
# [[1, 2, 3, 4],
#  [5, 6, 7, 8],
#  [9, 10, 11, 12]]

B[1]           # Second layer
# [[13, 14, 15, 16],
#  [17, 18, 19, 20],
#  [21, 22, 23, 24]]
```

#### Accessing rows across layers
```python
B[:, 0, :]     # First row from all layers
# [[1, 2, 3, 4],
#  [13, 14, 15, 16]]

B[:, 1, :]     # Second row from all layers
# [[5, 6, 7, 8],
#  [17, 18, 19, 20]]
```

#### Accessing columns across layers
```python
B[:, :, 0]     # First column from all layers
# [[1, 5, 9],
#  [13, 17, 21]]

B[:, :, 2]     # Third column from all layers
# [[3, 7, 11],
#  [15, 19, 23]]
```

#### Single element
```python
B[0, 1, 2]     # Layer 0, Row 1, Col 2 = 7
B[1, 2, 3]     # Layer 1, Row 2, Col 3 = 24
```

#### Complex slicing
```python
B[0:1, 1:3, 2:4]
# From layer 0, rows 1-2, cols 2-3
# [[[7, 8],
#   [11, 12]]]
# Shape: (1, 2, 2)
```

### 4.4 Real-World 3D Array Example: RGB Images

```python
# Image: 256 pixels high, 256 pixels wide, 3 color channels (RGB)
image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

print(image.shape)  # (256, 256, 3)
```

**Useful operations:**

```python
# Get red channel
red_channel = image[:, :, 0]

# Get green channel
green_channel = image[:, :, 1]

# Get blue channel
blue_channel = image[:, :, 2]

# Get top-left 50x50 pixels, all channels
top_left = image[0:50, 0:50, :]

# Flip image horizontally
flipped = image[:, ::-1, :]

# Flip image vertically
flipped_v = image[::-1, :, :]

# Convert to grayscale (average of RGB)
gray = image.mean(axis=2)
```

### 4.5 Higher Dimensions

NumPy supports arrays of any dimension!

```python
# 4D array: (batches, height, width, channels)
batch_images = np.random.rand(32, 64, 64, 3)
# 32 images, each 64x64 pixels with 3 channels

# 5D array: (videos, frames, height, width, channels)
video_batch = np.random.rand(10, 100, 256, 256, 3)
# 10 videos, each 100 frames, 256x256 resolution, RGB
```

**General principle:**
```python
array[dim0_slice, dim1_slice, dim2_slice, ..., dimN_slice]
```

---

<a name="views-copies"></a>
## 🔬 CHAPTER 5: Views vs Copies - The Complete Truth

### 5.1 What is a View?

A **view** is a different way of looking at the same data in memory.

```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]  # b is a view of a

print(b)           # [2, 3, 4]
print(b.base is a) # True - b is based on a
```

**Memory diagram:**
```
a: [1][2][3][4][5]
    └─b points here: [2][3][4]
```

### 5.2 Why Views Matter

#### Modification through view affects original
```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]
b[0] = 999

print(a)  # [1, 999, 3, 4, 5]  ← a changed!
print(b)  # [999, 3, 4]
```

#### Multiple views of same data
```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]
c = a[2:]
b[1] = 100  # Changes index 2 of a

print(a)  # [1, 2, 100, 4, 5]
print(b)  # [2, 100, 4]
print(c)  # [100, 4, 5]  ← c affected too!
```

### 5.3 What is a Copy?

A **copy** creates new, independent data in memory.

```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4].copy()

b[0] = 999
print(a)  # [1, 2, 3, 4, 5]  ← a unchanged
print(b)  # [999, 3, 4]
```

**Memory diagram:**
```
a: [1][2][3][4][5]
b: [2][3][4]  ← separate memory
```

### 5.4 Operations That Create Views

1. **Basic slicing**
```python
b = a[1:3]
b = a[::2]
b = a[:]
```

2. **Reshaping (usually)**
```python
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(2, 3)  # View if possible
```

3. **Transpose**
```python
A = np.array([[1, 2], [3, 4]])
B = A.T  # View
```

4. **Diagonal**
```python
d = np.diag(A)  # View
```

### 5.5 Operations That Create Copies

1. **Fancy indexing**
```python
indices = [0, 2, 4]
b = a[indices]  # Copy
```

2. **Boolean indexing**
```python
b = a[a > 3]  # Copy
```

3. **Explicit copy**
```python
b = a.copy()
```

4. **Some array operations**
```python
b = a + 1     # Copy
b = np.sin(a) # Copy
```

### 5.6 How to Check: View or Copy?

#### Method 1: Check `.base`
```python
a = np.array([1, 2, 3, 4, 5])

b = a[1:3]
print(b.base is a)  # True → view

c = a[[1, 2]]
print(c.base is None)  # True → copy
```

#### Method 2: Check flags
```python
print(a.flags['OWNDATA'])  # True → owns its data (not a view)
print(b.flags['OWNDATA'])  # False → doesn't own data (is a view)
```

#### Method 3: Modify and observe
```python
b[0] = 999
if a[1] == 999:
    print("b is a view of a")
else:
    print("b is a copy of a")
```

### 5.7 Advanced Example: 2D View Behavior

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Views
B = A[1:, :]       # View
C = A[:, 1]        # View
D = A[0:2, 1:3]    # View

# Modify through view
B[0, 0] = 999

print(A)
# [[1,   2,   3],
#  [999, 5,   6],
#  [7,   8,   9]]

C[0] = 111
print(A)
# [[1,  111, 3],
#  [999, 5,   6],
#  [7,   8,   9]]
```

### 5.8 The Subtle Case: Reshape

```python
a = np.array([1, 2, 3, 4, 5, 6])

# This creates a view (possible with strides)
b = a.reshape(2, 3)
b[0, 0] = 999
print(a)  # [999, 2, 3, 4, 5, 6]

# But this might create a copy (if data is not contiguous)
c = a[::2]           # [1, 3, 5]
d = c.reshape(3, 1)  # Might need to copy
```

### 5.9 Best Practices

1. **When you need independence, always copy explicitly**
```python
safe_array = original_array.copy()
```

2. **When memory is a concern, use views**
```python
# Process large array in chunks without copying
for chunk in [big_array[i:i+1000] for i in range(0, len(big_array), 1000)]:
    process(chunk)
```

3. **Document your intent**
```python
# Good: explicit
def modify_in_place(arr):
    """Modifies arr in place (expects a view)"""
    arr[:] = arr * 2

# Good: explicit
def safe_operation(arr):
    """Returns modified copy, original unchanged"""
    result = arr.copy()
    result *= 2
    return result
```

---

<a name="slice-assignment"></a>
## ✏️ CHAPTER 6: Slice Assignment - All Rules Explained

### 6.1 Fundamental Constraint: Fixed Size

NumPy arrays have **fixed size**. You cannot:
- Insert elements
- Delete elements
- Change total number of elements

```python
a = np.array([1, 2, 3, 4, 5])

# ❌ THESE DON'T WORK:
# a[2:2] = [10]      # Can't insert
# a[1:3] = []        # Can't delete
# a[1:4] = [7, 8]    # Can't change size
```

### 6.2 Rule 1: Scalar Assignment (Broadcasting)

When you assign a scalar to a slice, it broadcasts to all positions.

```python
a = np.array([1, 2, 3, 4, 5])
a[1:4] = 0

print(a)  # [1, 0, 0, 0, 5]
```

**What happens:**
1. Slice `a[1:4]` selects 3 positions
2. Scalar `0` is broadcast to all 3
3. All 3 positions get value `0`

#### More examples:
```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

a[::2] = -1
print(a)  # [-1, 1, -1, 3, -1, 5, -1, 7, -1, 9]

a[2:8] = 100
print(a)  # [-1, 1, 100, 100, 100, 100, 100, 100, -1, 9]
```

### 6.3 Rule 2: Array Assignment (Must Match)

When assigning an array to a slice, **shapes must match exactly**.

```python
a = np.array([1, 2, 3, 4, 5])

# ✅ Correct: 3 positions, 3 values
a[1:4] = [10, 20, 30]
print(a)  # [1, 10, 20, 30, 5]

# ❌ Wrong: 3 positions, 2 values
a[1:4] = [10, 20]  # ValueError!

# ❌ Wrong: 3 positions, 4 values
a[1:4] = [10, 20, 30, 40]  # ValueError!
```

#### Step slicing assignment:
```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Selects indices: 0, 2, 4, 6, 8 (5 positions)
a[::2] = [10, 20, 30, 40, 50]
print(a)  # [10, 1, 20, 3, 30, 5, 40, 7, 50, 9]

# ❌ Wrong: 5 positions, 3 values
a[::2] = [10, 20, 30]  # ValueError!
```

### 6.4 Rule 3: 2D Scalar Assignment

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
```

#### Assign to entire row
```python
A[1, :] = 0
print(A)
# [[1, 2, 3],
#  [0, 0, 0],
#  [7, 8, 9]]
```

#### Assign to entire column
```python
A[:, 1] = 0
print(A)
# [[1, 0, 3],
#  [4, 0, 6],
#  [7, 0, 9]]
```

#### Assign to submatrix
```python
A[0:2, 1:3] = 0
print(A)
# [[1, 0, 0],
#  [4, 0, 0],
#  [7, 8, 9]]
```

### 6.5 Rule 4: 2D Array Assignment

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
```

#### Assign row with matching array
```python
A[1, :] = [10, 20, 30]
print(A)
# [[1,  2,  3],
#  [10, 20, 30],
#  [7,  8,  9]]
```

#### Assign column with matching array
```python
A[:, 1] = [100, 200, 300]
print(A)
# [[1, 100, 3],
#  [4, 200, 6],
#  [7, 300, 9]]
```

#### Assign to submatrix
```python
A[0:2, 1:3] = [[10, 20],
               [30, 40]]
print(A)
# [[1, 10, 20],
#  [4, 30, 40],
#  [7,  8,  9]]
```

### 6.6 Complex Assignment Examples

#### Example 1: Diagonal assignment
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Get diagonal as a view
d = np.diag(A)
d[:] = 0

print(A)
# [[0, 2, 3],
#  [4, 0, 6],
#  [7, 8, 0]]
```

#### Example 2: Conditional assignment (creates copy)
```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# This creates a copy, not a view!
a[a > 5] = 0
print(a)  # [1, 2, 3, 4, 5, 0, 0, 0, 0]
```

#### Example 3: Multiple dimension assignment
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Every other row, every other column
A[::2, ::2] = 0
print(A)
# [[0, 2, 0],
#  [4, 5, 6],
#  [0, 8, 0]]
```

### 6.7 Special Case: Using Views

```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:4]  # b is a view

# Assignment to b affects a
b[:] = 0
print(a)  # [1, 0, 0, 0, 5]

# Single element assignment
b[1] = 100
print(a)  # [1, 0, 100, 0, 5]
```

### 6.8 Common Mistakes

#### Mistake 1: Trying to resize
```python
a = np.array([1, 2, 3, 4, 5])

# ❌ Won't work
a[1:3] = [10]  # Slice has 2 positions, RHS has 1
```

#### Mistake 2: Shape mismatch
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# ❌ Wrong: column has 2 elements, giving 3
A[:, 1] = [10, 20, 30]  # ValueError
```

#### Mistake 3: Forgetting about broadcasting
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A[0] = 10  # This works! 10 broadcasts to [10, 10, 10]
print(A)
# [[10, 10, 10],
#  [ 4,  5,  6]]
```

### 6.9 Assignment Through Boolean Masks

```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Boolean mask
mask = a > 5
print(mask)  # [False, False, False, False, False, True, True, True, True]

# Assignment
a[mask] = 0
print(a)  # [1, 2, 3, 4, 5, 0, 0, 0, 0]

# Shorthand
a[a % 2 == 0] = -1  # Replace all even numbers with -1
```

---

<a name="broadcasting"></a>
## 📡 CHAPTER 7: Broadcasting - From Basic to Advanced

### 7.1 What is Broadcasting?

Broadcasting is NumPy's way of performing operations on arrays of different shapes.

**Simple example:**
```python
a = np.array([1, 2, 3])
b = 10

result = a + b
print(result)  # [11, 12, 13]
```

What happened? `b` was "broadcast" to match shape of `a`:
```
a:      [1, 2, 3]
b:      10 → [10, 10, 10]  (broadcasted)
result: [11, 12, 13]
```

### 7.2 The Broadcasting Rules

NumPy compares array shapes **element-wise from right to left**.

Two dimensions are compatible when:
1. They are **equal**, OR
2. One of them is **1**

If these conditions aren't met, NumPy raises a `ValueError`.

### 7.3 Broadcasting Rule Examples

#### Example 1: Scalar + Array
```python
Shape: (3,)
       ()      → treated as (1,) → broadcasts to (3,)
```

```python
a = np.array([1, 2, 3])
b = 5
result = a + b  # [6, 7, 8]
```

#### Example 2: 1D Array + 2D Array
```python
A.shape: (3, 4)
b.shape: (4,)    → treated as (1, 4)

Comparison right-to-left:
4 == 4 ✓
3 vs 1 → broadcast 1 to 3 ✓
```

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

b = np.array([10, 20, 30, 40])

result = A + b
# b broadcasts to:
# [[10, 20, 30, 40],
#  [10, 20, 30, 40],
#  [10, 20, 30, 40]]
#
# Result:
# [[11, 22, 33, 44],
#  [15, 26, 37, 48],
#  [19, 30, 41, 52]]
```

#### Example 3: Column vector + 2D Array
```python
A.shape: (3, 4)
c.shape: (3, 1)

Comparison:
4 vs 1 → broadcast 1 to 4 ✓
3 == 3 ✓
```

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

c = np.array([[10],
              [20],
              [30]])

result = A + c
# c broadcasts to:
# [[10, 10, 10, 10],
#  [20, 20, 20, 20],
#  [30, 30, 30, 30]]
#
# Result:
# [[11, 12, 13, 14],
#  [25, 26, 27, 28],
#  [39, 40, 41, 42]]
```

#### Example 4: Two 1D arrays
```python
a.shape: (4,)    → (1, 4)
b.shape: (3,)    → (3, 1) with reshape

Result.shape: (3, 4)
```

```python
a = np.array([1, 2, 3, 4])        # shape (4,)
b = np.array([10, 20, 30])[:, np.newaxis]  # shape (3, 1)

result = a + b
# [[11, 12, 13, 14],
#  [21, 22, 23, 24],
#  [31, 32, 33, 34]]
```

### 7.4 Visual Broadcasting Examples

#### Visualizing row broadcast
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

row = np.array([10, 20, 30])

# Visual: row gets "copied" down
[[10, 20, 30],      [[1, 2, 3],
 [10, 20, 30],   +   [4, 5, 6],   = result
 [10, 20, 30]]       [7, 8, 9]]
```

#### Visualizing column broadcast
```python
col = np.array([[10],
                [20],
                [30]])

# Visual: col gets "copied" across
[[10, 10, 10],      [[1, 2, 3],
 [20, 20, 20],   +   [4, 5, 6],   = result
 [30, 30, 30]]       [7, 8, 9]]
```

### 7.5 Broadcasting with Operations

Broadcasting works with all element-wise operations:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

row = np.array([10, 20, 30])

# Addition
A + row
# [[11, 22, 33],
#  [14, 25, 36]]

# Multiplication
A * row
# [[10, 40, 90],
#  [40, 100, 180]]

# Power
A ** row
# [[1, 1048576, 205891132094649],
#  [4, 9765625, 729000000000000]]

# Comparison
A > row
# [[False, False, False],
#  [False, False, False]]

A < row
# [[True, True, True],
#  [True, True, True]]
```

### 7.6 Common Broadcasting Patterns

#### Pattern 1: Normalize each row
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Row means
row_means = A.mean(axis=1, keepdims=True)
print(row_means.shape)  # (3, 1)

# Normalize
A_normalized = A - row_means
```

#### Pattern 2: Normalize each column
```python
# Column means
col_means = A.mean(axis=0, keepdims=True)
print(col_means.shape)  # (1, 3)

# Normalize
A_normalized = A - col_means
```

#### Pattern 3: Outer product
```python
a = np.array([1, 2, 3])      # (3,)
b = np.array([10, 20, 30])    # (3,)

# Reshape for broadcasting
a_col = a[:, np.newaxis]      # (3, 1)
b_row = b[np.newaxis, :]      # (1, 3)

outer = a_col * b_row
# [[10, 20, 30],
#  [20, 40, 60],
#  [30, 60, 90]]
```

#### Pattern 4: Distance matrix
```python
points = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])  # 3 points in 2D

# Reshape for broadcasting
p1 = points[:, np.newaxis, :]  # (3, 1, 2)
p2 = points[np.newaxis, :, :]  # (1, 3, 2)

# Compute all pairwise distances
distances = np.sqrt(((p1 - p2) ** 2).sum(axis=2))
```

### 7.7 When Broadcasting Fails

```python
A.shape: (3, 4)
B.shape: (3, 3)

Comparison:
4 != 3 and neither is 1 ✗
ValueError: operands could not be broadcast together
```

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

A + B  # ❌ ValueError
```

### 7.8 Advanced: Broadcasting in Assignment

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Broadcast scalar
A[:, 1] = 0
# [[1, 0, 3],
#  [4, 0, 6],
#  [7, 0, 9]]

# Broadcast row
A[:] = [10, 20, 30]
# [[10, 20, 30],
#  [10, 20, 30],
#  [10, 20, 30]]

# Broadcast column
A[:] = [[1], [2], [3]]
# [[1, 1, 1],
#  [2, 2, 2],
#  [3, 3, 3]]
```

### 7.9 keepdims Parameter

Many NumPy functions have a `keepdims` parameter to facilitate broadcasting:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Without keepdims
row_sum = A.sum(axis=1)
print(row_sum.shape)  # (3,)

# With keepdims
row_sum = A.sum(axis=1, keepdims=True)
print(row_sum.shape)  # (3, 1) - ready for broadcasting!

# Now you can do:
A_normalized = A / row_sum
```

---

<a name="advanced-slicing"></a>
## 🚀 CHAPTER 8: Advanced Slicing Techniques

### 8.1 Ellipsis (...)

The ellipsis `...` represents "all remaining dimensions".

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# These are equivalent:
A[1, :]
A[1, ...]

# For 3D arrays:
B = np.random.rand(2, 3, 4, 5)

# These are equivalent:
B[0, :, :, :]
B[0, ...]

# Get all last elements across all dimensions
B[..., -1]  # Same as B[:, :, :, -1]
```

### 8.2 np.newaxis

Add a new axis of length 1 to change array shape:

```python
a = np.array([1, 2, 3])
print(a.shape)  # (3,)

# Add axis at beginning
b = a[np.newaxis, :]
print(b.shape)  # (1, 3)
print(b)  # [[1, 2, 3]]

# Add axis at end
c = a[:, np.newaxis]
print(c.shape)  # (3, 1)
print(c)
# [[1],
#  [2],
#  [3]]

# Multiple new axes
d = a[np.newaxis, :, np.newaxis]
print(d.shape)  # (1, 3, 1)
```

**Use case: broadcasting**
```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# Create outer product using broadcasting
result = a[:, np.newaxis] * b[np.newaxis, :]
# [[10, 20, 30],
#  [20, 40, 60],
#  [30, 60, 90]]
```

### 8.3 Integer Array Indexing (Fancy Indexing)

Select arbitrary elements:

```python
a = np.array([10, 20, 30, 40, 50])

# Select elements at indices 0, 2, 4
indices = [0, 2, 4]
result = a[indices]
print(result)  # [10, 30, 50]

# Can repeat indices
indices = [0, 0, 2, 2, 4, 4]
result = a[indices]
print(result)  # [10, 10, 30, 30, 50, 50]
```

**Important:** This creates a **copy**, not a view!

#### 2D Fancy Indexing
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Select specific elements
rows = [0, 1, 2]
cols = [0, 1, 2]
result = A[rows, cols]
print(result)  # [1, 5, 9] - diagonal!

# Select corners
rows = [0, 0, 2, 2]
cols = [0, 2, 0, 2]
result = A[rows, cols]
print(result)  # [1, 3, 7, 9]
```

### 8.4 Boolean Indexing

Select elements based on condition:

```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create boolean mask
mask = a > 5
print(mask)  # [False, False, False, False, False, True, True, True, True]

# Apply mask
result = a[mask]
print(result)  # [6, 7, 8, 9]

# Shorthand
result = a[a > 5]
print(result)  # [6, 7, 8, 9]

# Multiple conditions
result = a[(a > 3) & (a < 7)]
print(result)  # [4, 5, 6]

# Important: & (and), | (or), ~ (not)
result = a[(a < 3) | (a > 7)]
print(result)  # [1, 2, 8, 9]
```

**Important:** Boolean indexing creates a **copy**!

#### 2D Boolean Indexing
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Select elements > 5
result = A[A > 5]
print(result)  # [6, 7, 8, 9] - flattened!

# Modify with boolean indexing
A[A > 5] = 0
print(A)
# [[1, 2, 3],
#  [4, 5, 0],
#  [0, 0, 0]]
```

### 8.5 Combining Slicing and Fancy Indexing

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

# Regular slice for rows, fancy index for columns
rows = slice(0, 3)  # Rows 0-2
cols = [0, 2]       # Columns 0 and 2
result = A[rows, cols]  # ❌ This doesn't work as expected!

# Correct way:
result = A[0:3, :][:, [0, 2]]
# Or:
result = A[np.ix_([0, 1, 2], [0, 2])]
```

### 8.6 np.ix_ for Cartesian Product

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

# Select rows 0, 2 and columns 1, 3
rows = [0, 2]
cols = [1, 3]

result = A[np.ix_(rows, cols)]
print(result)
# [[2, 4],
#  [10, 12]]
```

### 8.7 Advanced Boolean Operations

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Conditions along axis
# Where any element in row > 5
row_mask = (A > 5).any(axis=1)
print(row_mask)  # [False, True, True]

# Select entire rows
result = A[row_mask]
# [[4, 5, 6],
#  [7, 8, 9]]

# Where all elements in row > 5
row_mask = (A > 5).all(axis=1)
print(row_mask)  # [False, False, True]

result = A[row_mask]
# [[7, 8, 9]]
```

### 8.8 np.where

Find indices where condition is true:

```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Get indices where a > 5
indices = np.where(a > 5)
print(indices)  # (array([5, 6, 7, 8]),)

# Use these indices
result = a[indices]
print(result)  # [6, 7, 8, 9]

# Conditional replacement
result = np.where(a > 5, 100, a)
print(result)  # [1, 2, 3, 4, 5, 100, 100, 100, 100]
```

#### 2D np.where
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Find indices where A > 5
rows, cols = np.where(A > 5)
print(rows)  # [1, 2, 2, 2]
print(cols)  # [2, 0, 1, 2]

# Access these elements
print(A[rows, cols])  # [6, 7, 8, 9]
```

---

<a name="patterns"></a>
## 🎨 CHAPTER 9: Common Patterns & Use Cases

### 9.1 Data Normalization

#### Min-Max Normalization
```python
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Normalize to [0, 1]
min_val = data.min()
max_val = data.max()
normalized = (data - min_val) / (max_val - min_val)
```

#### Z-Score Normalization
```python
mean = data.mean()
std = data.std()
normalized = (data - mean) / std
```

#### Per-column normalization
```python
# Each column independently
col_min = data.min(axis=0, keepdims=True)
col_max = data.max(axis=0, keepdims=True)
normalized = (data - col_min) / (col_max - col_min)
```

### 9.2 Image Processing

#### Grayscale Conversion
```python
# RGB image: shape (height, width, 3)
rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Convert to grayscale: average across channels
gray_image = rgb_image.mean(axis=2)

# Or use standard weights
weights = np.array([0.299, 0.114, 0.587])
gray_image = (rgb_image * weights).sum(axis=2)
```

#### Cropping
```python
# Crop center 50x50 region
height, width = 100, 100
crop_size = 50
start_h = (height - crop_size) // 2
start_w = (width - crop_size) // 2

cropped = rgb_image[start_h:start_h+crop_size, 
                    start_w:start_w+crop_size, :]
```

#### Flipping
```python
# Horizontal flip
flipped_h = rgb_image[:, ::-1, :]

# Vertical flip
flipped_v = rgb_image[::-1, :, :]

# Both
flipped_both = rgb_image[::-1, ::-1, :]
```

#### Rotation 90°
```python
# Rotate 90° clockwise
rotated = np.rot90(rgb_image, k=-1)

# Rotate 90° counter-clockwise
rotated = np.rot90(rgb_image, k=1)
```

### 9.3 Sliding Window Operations

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window_size = 3

# Extract all windows
windows = []
for i in range(len(data) - window_size + 1):
    window = data[i:i+window_size]
    windows.append(window)

windows = np.array(windows)
print(windows)
# [[1, 2, 3],
#  [2, 3, 4],
#  [3, 4, 5],
#  [4, 5, 6],
#  [5, 6, 7],
#  [6, 7, 8],
#  [7, 8, 9],
#  [8, 9, 10]]

# Compute statistics on windows
window_means = windows.mean(axis=1)
window_max = windows.max(axis=1)
```

### 9.4 Matrix Operations

#### Extract diagonal
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

diag = np.diag(A)
print(diag)  # [1, 5, 9]

# Set diagonal
np.fill_diagonal(A, 0)
# [[0, 2, 3],
#  [4, 0, 6],
#  [7, 8, 0]]
```

#### Extract upper/lower triangle
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Upper triangle
upper = np.triu(A)
# [[1, 2, 3],
#  [0, 5, 6],
#  [0, 0, 9]]

# Lower triangle
lower = np.tril(A)
# [[1, 0, 0],
#  [4, 5, 0],
#  [7, 8, 9]]
```

### 9.5 Batch Processing

```python
# Process images in batches
batch_size = 32
num_images = 1000
image_shape = (64, 64, 3)

images = np.random.rand(num_images, *image_shape)

# Process batch by batch
for i in range(0, num_images, batch_size):
    batch = images[i:i+batch_size]
    # Process batch...
    processed = batch * 2  # Example operation
```

### 9.6 Masking and Filtering

```python
data = np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10])

# Replace negative values with 0
data[data < 0] = 0
print(data)  # [1, 0, 3, 0, 5, 0, 7, 0, 9, 0]

# Clip values to range
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
clipped = np.clip(data, 3, 7)
print(clipped)  # [3, 3, 3, 4, 5, 6, 7, 7, 7, 7]
```

### 9.7 Creating Patterns

```python
# Checkerboard pattern
size = 8
checkerboard = np.zeros((size, size))
checkerboard[::2, ::2] = 1
checkerboard[1::2, 1::2] = 1
```

---

<a name="pitfalls"></a>
## ⚠️ CHAPTER 10: Pitfalls & Debugging

### 10.1 View Modification Pitfall

```python
# Pitfall: Unintentional modification
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# This creates a view!
B = A[0:2, :]

# Modifying B affects A
B[0, 0] = 999
print(A[0, 0])  # 999 - A changed!

# Solution: Explicit copy
B = A[0:2, :].copy()
B[0, 0] = 999
print(A[0, 0])  # 1 - A unchanged
```

### 10.2 Broadcasting Shape Mismatch

```python
# Pitfall: Unexpected shapes
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)

b = np.array([10, 20])     # (2,)

# This fails!
try:
    result = A + b
except ValueError as e:
    print(f"Error: {e}")
    # operands could not be broadcast together

# Solution: Reshape
b = b[:, np.newaxis]       # (2, 1)
result = A + b             # Works!
```

### 10.3 Copy vs View Confusion

```python
# Confusing case
a = np.array([1, 2, 3, 4, 5])

# Basic slicing: view
b = a[1:4]
print(b.base is a)  # True

# Fancy indexing: copy
c = a[[1, 2, 3]]
print(c.base is a)  # False

# Boolean indexing: copy
d = a[a > 2]
print(d.base is a)  # False
```

### 10.4 Slice Assignment Size Mismatch

```python
# Pitfall: Wrong size
a = np.array([1, 2, 3, 4, 5])

try:
    a[1:4] = [10, 20]  # 3 positions, 2 values
except ValueError as e:
    print(f"Error: {e}")
    # could not broadcast input array

# Solution: Match sizes
a[1:4] = [10, 20, 30]  # Works!
```

### 10.5 Negative Step Confusion

```python
# Pitfall: Wrong direction
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# This gives empty array!
result = a[2:8:-1]
print(result)  # []

# Solution: Start should be > stop for negative step
result = a[8:2:-1]
print(result)  # [8, 7, 6, 5, 4, 3]
```

### 10.6 Dimension Reduction Surprise

```python
# Pitfall: Lost dimension
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Single index reduces dimension
row = A[1]
print(row.shape)  # (3,) - 1D!

# Slice preserves dimension
row = A[1:2]
print(row.shape)  # (1, 3) - 2D!

# Which one to use?
# Depends on what you need:
# - 1D for simple operations
# - 2D for matrix operations and broadcasting
```

### 10.7 Boolean Indexing Returns Flattened Array

```python
# Pitfall: Shape change
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

result = A[A > 5]
print(result)        # [6, 7, 8, 9]
print(result.shape)  # (4,) - flattened!

# If you need original positions, use np.where
rows, cols = np.where(A > 5)
```

### 10.8 Memory Order (C vs Fortran)

```python
# Usually not a problem, but can affect performance
A = np.array([[1, 2, 3],
              [4, 5, 6]], order='C')  # Row-major (default)

B = np.array([[1, 2, 3],
              [4, 5, 6]], order='F')  # Column-major

# They look the same
print(A)
print(B)

# But memory layout differs
print(A.flags['C_CONTIGUOUS'])  # True
print(A.flags['F_CONTIGUOUS'])  # False

print(B.flags['C_CONTIGUOUS'])  # False
print(B.flags['F_CONTIGUOUS'])  # True
```

---

<a name="performance"></a>
## ⚡ CHAPTER 11: Performance Considerations

### 11.1 Views are Fast (No Copy)

```python
import time

# Large array
large_array = np.random.rand(10000, 10000)

# View (fast)
start = time.time()
view = large_array[1000:2000, 1000:2000]
view_time = time.time() - start
print(f"View: {view_time:.6f} seconds")  # Very fast

# Copy (slower)
start = time.time()
copy = large_array[1000:2000, 1000:2000].copy()
copy_time = time.time() - start
print(f"Copy: {copy_time:.6f} seconds")  # Much slower
```

### 11.2 Broadcasting vs Loops

```python
# Slow: Python loop
A = np.random.rand(1000, 1000)
b = np.random.rand(1000)

start = time.time()
result = np.zeros_like(A)
for i in range(A.shape[0]):
    result[i] = A[i] + b
loop_time = time.time() - start

# Fast: Broadcasting
start = time.time()
result = A + b
broadcast_time = time.time() - start

print(f"Loop: {loop_time:.4f}s")
print(f"Broadcasting: {broadcast_time:.4f}s")
print(f"Speedup: {loop_time / broadcast_time:.1f}x")
```

### 11.3 Vectorization

```python
# Slow: Element-by-element
n = 1000000
a = np.random.rand(n)
b = np.zeros(n)

start = time.time()
for i in range(n):
    b[i] = a[i] ** 2
loop_time = time.time() - start

# Fast: Vectorized
start = time.time()
b = a ** 2
vec_time = time.time() - start

print(f"Loop: {loop_time:.4f}s")
print(f"Vectorized: {vec_time:.4f}s")
print(f"Speedup: {loop_time / vec_time:.1f}x")
```

### 11.4 Memory Contiguity

```python
# Contiguous array (fast)
A = np.random.rand(1000, 1000)
print(A.flags['C_CONTIGUOUS'])  # True

start = time.time()
result = A.sum()
contig_time = time.time() - start

# Non-contiguous view (slower)
B = A[::2, ::2]
print(B.flags['C_CONTIGUOUS'])  # False

start = time.time()
result = B.sum()
non_contig_time = time.time() - start

print(f"Contiguous: {contig_time:.6f}s")
print(f"Non-contiguous: {non_contig_time:.6f}s")
```

### 11.5 In-place Operations

```python
# Creates new array
a = np.random.rand(1000000)

start = time.time()
b = a * 2
new_array_time = time.time() - start

# In-place (faster, less memory)
start = time.time()
a *= 2
inplace_time = time.time() - start

print(f"New array: {new_array_time:.6f}s")
print(f"In-place: {inplace_time:.6f}s")
```

---

<a name="practice"></a>
## 💪 CHAPTER 12: Practice Problems - Progressive Difficulty

### Level 1: Basics

#### Problem 1.1
```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# Extract elements at indices 2, 5, 8
```

<details>
<summary>Solution</summary>

```python
result = a[[2, 5, 8]]
# or
result = a[2::3]
print(result)  # [2, 5, 8]
```
</details>

#### Problem 1.2
```python
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Replace all even numbers with -1
```

<details>
<summary>Solution</summary>

```python
a[a % 2 == 0] = -1
print(a)  # [1, -1, 3, -1, 5, -1, 7, -1, 9, -1]
```
</details>

#### Problem 1.3
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# Extract the 2x2 bottom-right corner
```

<details>
<summary>Solution</summary>

```python
result = A[1:, 1:]
print(result)
# [[5, 6],
#  [8, 9]]
```
</details>

### Level 2: Intermediate

#### Problem 2.1
```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
# Extract elements at positions (0,0), (1,1), (2,2), (3,3)
# Should get [1, 6, 11, 16]
```

<details>
<summary>Solution</summary>

```python
# Method 1: Fancy indexing
result = A[[0, 1, 2, 3], [0, 1, 2, 3]]

# Method 2: Use np.diag
result = np.diag(A)

print(result)  # [1, 6, 11, 16]
```
</details>

#### Problem 2.2
```python
a = np.array([1, 2, 3, 4, 5])
# Without using loops, create:
# [[1, 2, 3, 4, 5],
#  [2, 4, 6, 8, 10],
#  [3, 6, 9, 12, 15]]
```

<details>
<summary>Solution</summary>

```python
multipliers = np.array([1, 2, 3])[:, np.newaxis]
result = multipliers * a
print(result)
```
</details>

#### Problem 2.3
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# Set all elements in the second row to the sum of the first and third rows
```

<details>
<summary>Solution</summary>

```python
A[1, :] = A[0, :] + A[2, :]
print(A)
# [[1, 2, 3],
#  [8, 10, 12],
#  [7, 8, 9]]
```
</details>

### Level 3: Advanced

#### Problem 3.1
```python
# Create a 5x5 checkerboard pattern
# [[1, 0, 1, 0, 1],
#  [0, 1, 0, 1, 0],
#  [1, 0, 1, 0, 1],
#  [0, 1, 0, 1, 0],
#  [1, 0, 1, 0, 1]]
```

<details>
<summary>Solution</summary>

```python
checkerboard = np.zeros((5, 5), dtype=int)
checkerboard[::2, ::2] = 1
checkerboard[1::2, 1::2] = 1
print(checkerboard)
```
</details>

#### Problem 3.2
```python
A = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25]])
# Extract elements forming an X pattern:
# [1, 7, 13, 19, 25, 5, 9, 17, 21]
```

<details>
<summary>Solution</summary>

```python
# Main diagonal
main_diag = np.diag(A)

# Anti-diagonal
anti_diag = np.diag(np.fliplr(A))

# Combine (remove center if odd size)
result = np.concatenate([main_diag, anti_diag])
# Remove duplicate center
result = np.unique(result)
print(result)
```
</details>

#### Problem 3.3
```python
# Create a function that extracts all possible 3x3 windows from a larger array
def extract_windows(arr, window_size=3):
    # Your code here
    pass

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

windows = extract_windows(A, 3)
# Should return 4 windows: top-left, top-right, bottom-left, bottom-right
```

<details>
<summary>Solution</summary>

```python
def extract_windows(arr, window_size=3):
    h, w = arr.shape
    windows = []
    for i in range(h - window_size + 1):
        for j in range(w - window_size + 1):
            window = arr[i:i+window_size, j:j+window_size]
            windows.append(window)
    return windows

windows = extract_windows(A, 3)
for idx, window in enumerate(windows):
    print(f"Window {idx+1}:")
    print(window)
    print()
```
</details>

### Level 4: Expert

#### Problem 4.1
```python
# Implement a function that normalizes each row to have mean=0 and std=1
def normalize_rows(arr):
    # Your code here
    pass

A = np.random.rand(5, 10)
normalized = normalize_rows(A)
# Check: normalized.mean(axis=1) should be close to 0
# Check: normalized.std(axis=1) should be close to 1
```

<details>
<summary>Solution</summary>

```python
def normalize_rows(arr):
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    return (arr - mean) / std

A = np.random.rand(5, 10)
normalized = normalize_rows(A)
print("Row means:", normalized.mean(axis=1))
print("Row stds:", normalized.std(axis=1))
```
</details>

#### Problem 4.2
```python
# Create a function that computes pairwise distances between all points
# Input: points array of shape (N, D) where N is number of points, D is dimensions
# Output: distance matrix of shape (N, N)
def pairwise_distances(points):
    # Your code here (use broadcasting, no loops!)
    pass

points = np.array([[0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]])
dist_matrix = pairwise_distances(points)
```

<details>
<summary>Solution</summary>

```python
def pairwise_distances(points):
    # Expand dimensions for broadcasting
    p1 = points[:, np.newaxis, :]  # (N, 1, D)
    p2 = points[np.newaxis, :, :]  # (1, N, D)
    
    # Compute squared differences
    diff = p1 - p2  # (N, N, D)
    
    # Sum over dimensions and take square root
    distances = np.sqrt((diff ** 2).sum(axis=2))
    
    return distances

points = np.array([[0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]])
dist_matrix = pairwise_distances(points)
print(dist_matrix)
# [[0.         1.         1.         1.41421356]
#  [1.         0.         1.41421356 1.        ]
#  [1.         1.41421356 0.         1.        ]
#  [1.41421356 1.         1.         0.        ]]
```
</details>

#### Problem 4.3
```python
# Implement Conway's Game of Life step using only NumPy operations
def game_of_life_step(grid):
    # Rules:
    # 1. Any live cell with 2-3 neighbors survives
    # 2. Any dead cell with exactly 3 neighbors becomes alive
    # 3. All other cells die or stay dead
    # Your code here (use slicing and broadcasting!)
    pass

# Test with a glider pattern
grid = np.array([[0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])
next_grid = game_of_life_step(grid)
```

<details>
<summary>Solution</summary>

```python
def game_of_life_step(grid):
    h, w = grid.shape
    
    # Count neighbors using slicing
    neighbors = np.zeros_like(grid)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            
            # Handle boundaries with padding
            padded = np.pad(grid, 1, mode='constant')
            shifted = padded[1+i:1+i+h, 1+j:1+j+w]
            neighbors += shifted
    
    # Apply rules
    # Cell survives if it has 2-3 neighbors and is alive
    survives = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    
    # Cell is born if it has exactly 3 neighbors
    born = (grid == 0) & (neighbors == 3)
    
    return (survives | born).astype(int)

# Test
grid = np.array([[0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])

print("Initial:")
print(grid)
print("\nAfter 1 step:")
print(game_of_life_step(grid))
```
</details>

---

## 🎯 Final Summary & Key Takeaways

### Critical Concepts to Master

1. **Memory Model**
   - Arrays are contiguous blocks of memory
   - Slicing creates views (shares memory)
   - Fancy indexing creates copies

2. **Slicing Syntax**
   - `[start:stop:step]` for each dimension
   - Negative indices count from end
   - Negative step reverses direction

3. **Broadcasting Rules**
   - Compare shapes right-to-left
   - Dimensions compatible if equal or one is 1
   - Scalar broadcasts to any shape

4. **Assignment Rules**
   - Fixed size - no insertion/deletion
   - Scalar broadcasts automatically
   - Array must match or be broadcastable

5. **Views vs Copies**
   - Basic slicing → view
   - Fancy/boolean indexing → copy
   - Modifying view affects original

### Study Strategy

1. **Master the basics first** - Understand 1D slicing thoroughly
2. **Visualize** - Draw arrays and slice boundaries
3. **Practice broadcasting** - It's everywhere in NumPy
4. **Experiment** - Try examples in Python
5. **Check understanding** - Use `.base`, `.flags`, shape checking

### Common Exam Patterns

- Predict output of slice operations
- Determine if operation creates view or copy
- Identify broadcasting compatibility
- Debug shape mismatch errors
- Optimize code using vectorization

---

## 📚 Additional Resources

- NumPy documentation: https://numpy.org/doc/
- Practice: Project Euler problems with NumPy
- Visualize: Use `print(arr)` liberally
- Debug: Check shapes with `.shape` constantly

---

**🎓 You now have a complete guide to NumPy slicing and broadcasting!**

**Practice the problems, experiment with the examples, and you'll master these concepts.**

Good luck! 🚀