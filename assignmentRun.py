### Q1: Machine Precision ###

from assignmentFunctions import *
# Calculate epsilon for single precision
epsilon('single')
# Calculate epsilon for double precision
epsilon('double')
# Calculate epsilon for long double precision
epsilon('extended')

### Q2: LU Decomposition ###
from assignmentFunctions import *
A = [[3,1,0,0,0],
    [3,9,4,0,0],
    [0,9,20,10,0],
    [0,0,-22,31,-25],
    [0,0,0,-55,61]]

# 2a: 
Crout(A, separate = False)

# 2b:
from assignmentFunctions import *
# Return result of LU decomposition as separate matrices
L, U = Crout(A)
print("L:\n", L)
print("U:\n", U)
print("LU:\n", np.matmul(L, U))
det1 = determinant(A)
det2 = np.linalg.det(A)
print("Calculated determinant: ", det1)
print("np.linalg determinant: ", det2)

# 2c: solveMatrices() function
# 2d: Solve matrix equation Ax = b
from assignmentFunctions import *
b = [2,5,-4,8,9]
q_2d(A, b) # Contains solveMatrices and checkSolution functions.

# 2e: 
from assignmentFunctions import *
matrixInverse(A)
# Check that the inverse is correct
inverseCheck(A)
# Returned matrix should be the identity matrix, with an error on each point due
# to machine accuracy, i.e - zeroes may not be exactly zero, but of the same order 
# as epsilon for double precision floats.

### Q3: Interpolation ###

# 3a: interpolate() function
# 3b: cubicSpline() function
# 3c: 
from assignmentFunctions import *
x_data = [-2.1, -1.45, -1.3, -0.2, 0.1, 0.15, 0.9, 1.1, 1.5, 2.8, 3.8]
f_data = [0.012155, 0.122151, 0.184520, 0.960789, 0.990050, 0.977751, 
        0.422383, 0.298197, 0.105399, 3.936690e-4, 5.355348e-7]

plotBothInterpolants(x_data, f_data)

### Q4: FFTs and Convolution ###

# 4a: covolution() function, used with h() and g().
# 4b:
from assignmentFunctions import *
q4_plot() # takes arguments start, stop, samples. This plots using their default values.

### Q5: Random Numbers ###

# 5a:
from assignmentFunctions import *
seed = 0
samples = 1e5
bins = 100
uniform = uniformRandom(seed, samples) # Generates numbers
plotRandom(uniform, bins)

### 5b: Random numbers following a PDF using the transformation method                  ###
###     Desired PDF: P(y) = 0.5*cos(y/2).                                               ###
###     Cumulative distribution: F(y) = integral of P(_y) d_y between 0 and y           ###
###                              (0 not -inf since interval is between 0 and pi).       ###
###                              = sin(y/2)                                             ###
###     F(y) = x  =>  y = F_inverse(x) where x is uniform deviate.                      ###
###                 ~ y = 2arcisn(x) ~                                                  ###

from assignmentFunctions import *
x = uniformRandom(seed, samples)
transform = transformDeviate(x)
# Plot the numbers in a binned histogram against the expected PDF.
plotRandom(transform, bins, pdf_5b)

# 5c: 
rejection = rejection(seed, samples)
plotRandom(rejection, bins, pdf_5c)