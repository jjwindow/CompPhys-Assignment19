"""
Complete library of functions for Computational Physics Assignment, Autumn 2019.
Made by J. J. Window

These are the main functions written for each assignment question, organised by question.
These functions make use of the library assignmentTools, which contains small functions used
throughout the code but not specific to certain questions, such as type checking procedures.

The functions written here are utilised in the execution file.

"""

import numpy as np
import random
import copy
from functools import reduce
import matplotlib.pyplot as plt
from assignmentTools import *

"""
Q1a - short program to determine machine epsilon.
"""

def epsilon(precision = 'double'):
    """ Binary search approach used. Machine roundoff is greatest for values which round to 1, so 1 is used as
    a starting value. We add an initial small difference d which is continually decreased until it exceeds the machine
    precision and is rounded to 1.
    """
    n = 1   # Machine epsilon is the greatest round off error, which happens 
            # around 1.
    d = 0.1 
    t = n + d # Add a small amount to 1.

    def epsilon_single(t, d):
        """
        numpy.float32 used to calculate epsilon for single precision floats.
        Returns a value of 1.1920928955078125e-07.
        Mantissa size in bits of a 32-bit double is p=22. 
        Expected machine epsilon is then 2^(22-1) = 1.19209289 x 10^-16.
        """
        while t > n:
            d = 0.5 * d
            t_old = t
            t = np.float32(n + d)  # t rounded down to single precision.
        epsilon = t_old - 1
        epsilon_expected = 2**(-23)
        print("float32 epsilon: ", epsilon, '\n'"Expected value: ", epsilon_expected)
        return epsilon

    def epsilon_double(t, d):
        """
        Calculates epsilon for double-precision floats.
        Returns a value of 2.220446049250313e-16.
        Mantissa size in bits of a 64-bit double is p=53. 
        Expected machine epsilon is then 2^(54-1) = 2.22044605 x 10^-16.
        """
        while t > n:
            d = 0.5 * d
            t_old = t
            t = np.float64(n + d) 
        epsilon = t_old - 1
        epsilon_expected = 2**(-52)
        print("float64 epsilon: ", epsilon, '\n'"Expected value: ", epsilon_expected)
        return epsilon
    
    def epsilon_extended(t, d):
        """
        Calculates epsilon for long double precision, which is different depending on the machine
        used. In many cases, numpy uses double precision for longdouble.
        """

        while t > n:
            d = 0.5 * d
            t_old = t
            t = np.longdouble(n+d)
        epsilon = t_old - 1
        epsilon_expected = 2**(-1 * np.finfo(np.longdouble).precision)
        print("Long double precision is platform defined.")
        print("epsilon: ", epsilon)
        return epsilon

    if precision == 'single':
        return epsilon_single(t, d)
    elif precision == 'double':
        return epsilon_double(t, d)
    elif precision == 'extended':
        return epsilon_extended(t, d)
    else:
        raise Exception('Precision must be "single", "double" or "extended".')

### Q2: Crout LU decomposition ###

def Crout(A, separate = True):
    """
    Performs Crout's method for LU decomposition on an NxN matrix.
    Does not use pivoting.
    """
    isSquare = matrixCheck(A)
    if isSquare is False:
        raise TypeError('Matrix must be square.')


    N = len(A)
    # Initialise Lower and Upper matrices.
    L = np.zeros(shape=(N, N))
    U = np.zeros(shape=(N, N))

    for i in range(N):
        L[i][i] = 1 # Set diagonal elements of L to be 1

    for j in range(N):
        # Find elements of U from backward substitution
        for i in range(j+1):
            tailSum = sum(np.fromiter((L[i][k]*U[k][j] 
                          for k in range(0,i)), float)) # Sums the tail of dot product
            U[i][j] = A[i][j] - tailSum

        for i in range(j+1, N):
            # Find elements of L from forward substitution
            tailSum = sum(np.fromiter((L[i][k]*U[k][j] for k in range(0, i)), float))
            L[i][j] = (A[i][j]-tailSum)/U[j][j]

    if separate == False: # For question 2a.
        for i in range(N):
            L[i][i] = 0 # Strip diagonal elements of L 
        return np.add(L, U) # Returns combined matrix of L and U, without diagonal L elements.
    return L, U
    
### 2b: Decomposing the given matrix. ###
def determinant(A):
    """
    Uses Crout() function to decompose the matrix then calculates the determinant from the product of the diagonal
    elements of U.
    """
    L, U = Crout(A)
    # Determinant is product of diagonals of U
    det = reduce(lambda x, y : x*y, [U[i,i] for i in range(len(U))])
    return det

### 2c: Function to solve a matrix equation of form A.x = b where A can be decomposed into LU. ###

def solveMatrices(L, U, b):
    """
    Inputs of a lower diagonal (L) and upper diagonal (U) matrix as well as a resultant vector b to form equation LU.x=L.y=b. 
    Returns the vector x.
    """
    diagonalPairCheck(L, U) # Checks the matrices are in correct form.
    N = len(L) # Number of rows and columns in L and U.
        
    # Carry out forward substitution to find elements of y.
    y = [] # Initialise y=U.x.
    y.append(b[0]/L[0][0]) # Calculate first element of y.

    for i in range(1,N): # Calculates elements forwards from 1st element of y.
        _sum = 0

        for j in range(i):
            _sum += L[i][j]*y[j]

        y_new = (b[i]-_sum)/L[i][i]
        y.append(y_new)
    
    # Carry out backward substitution for x.
    x = np.zeros(N) # Initialise vector x.
    _sum = 0
    x[N-1] = (y[N-1]/U[N-1][N-1]) # Calculates Nth element of x.

    for i in range(N-2, -1, -1): # Calculates elements backwards from N-1th element.
        _sum = 0
        for j in range(i,N):
            _sum += U[i][j]*x[j]

        x[i] = (y[i]-_sum)/U[i][i] # Append cannot be used since substitution works backwards

    return x

def checkSolution(L, U, x, b):
    """
    Checks that the calculated value of x from solveMatrices() produces the correct result.
    performs L.U.x to get a test result for b, then compares to the b used in the equation.
    """
    y = np.matmul(U, x)
    b_test = np.matmul(L, y)
    _epsilon = epsilon() # Machine epsilon

    for i in range(len(b)):
        # Maximum and minimum bounds for any difference between b[i] and b_test[i] 
        # to be due to machine precision.
        b_test_min = b_test[i] - _epsilon
        b_test_max = b_test[i] + _epsilon

        # Checks if uncertainty due to machine precision includes b[i]
        if not ((b[i] >= b_test_min) and (b[i] <= b_test_max)) :
            raise Exception('Matrix solution is not correct.')

    return True

def q_2d(A, b):
    """
    Performs the calculation required for question 2d. Takes any A and b.
    Checks if the result it correct, and if it is then returns x.
    """
    L, U = Crout(A)
    x = solveMatrices(L, U, b)
    # Raise an exception if the result is wrong.
    checkSolution(L, U, x, b) 
    print("x: ", x)
    return x

def matrixInverse(A):
    """
    Finds the inverse of a matrix A by solving A.A_inv = I where I is 
    the identity matrix.
    """
    # Decompose the matrix A so it can be solved using matrixSolve().
    L, U = Crout(A)
    N = len(L)
    # Initialise I
    I = np.zeros((N, N))
    for i in range(N):
        I[i][i] = 1 
    
    # Initialise inverse matrix.
    A_inv = np.zeros((N, N))

    # Loop through columns on A_inv
    for i in range(N):
        b = []
        for j in range(N):
            # Selects column from I to use as b in matrix equation
            b.append(I[j][i])
        # Solves for one column of A_inv
        A_invColumn = solveMatrices(L, U, b)
        for j in range(N):
            A_inv[j][i] = A_invColumn[j]
    print("Inverse of A:\n", A_inv)
    return A_inv

def inverseCheck(A):
    """
    Returns the product of A and its calculated inverse from matrixInverse().
    Result should give the identity matrix, with the exception of floating
    point rounding errors of order 10^-16 or less.
    """
    return np.matmul(A, matrixInverse(A))

### 3a: Linear interpolation ###

def interpolate(x_data, f_data):
    """
    Function which generates a cubic spline interpolation for a tabulated data set, composed of
    x_data, f_data
    """
    # Check that the arguments passed are a valid data set for interpolation.
    interpolantCheck(x_data,f_data)

    # Generate a set of x values in the same range as the data points to plot the interpolated function with.
    N = len(f_data)
    x_num = 1000*N
    x = np.linspace(min(x_data), max(x_data), x_num) # Number of points must be large to make sure points close to the data values are plotted.
    f = [] # Initialise interpolated function values

    def lineFunc(i):
        """
        Function to generate the linear function that interpolates between the ith and (i+1)th 
        data points in x_data.
        """
        # Assigns x_i, x_(i+1), f_i, f_(i+1)
        x_left = x_data[i]
        x_right = x_data[i+1]
        f_left = f_data[i]
        f_right = f_data[i+1]

        def func(_x, i):
            """
            Returns the line that joins the two adjacent points.
            """
            A = (x_right-_x)/(x_right - x_left)
            B = 1-A
            return A*f_left + B*f_right
        return func

    # A list of interpolant functions for every data interval in the set.
    intervalList = []
    for i in range(N-1):
        intervalList.append(lineFunc(i))
    def findInterval(_x):
        """
        Given a data point _x, this function identifies which interval (i) the point lies in.
        The function returns the corresponding interpolant function for that interval.
        """
        for i in range(N-1):
            if _x >= x_data[i] and _x < x_data[i+1]:
                localInterval = intervalList[i]
                return localInterval(_x, i)

    # Returns the evenly spaced x values and the interpolating line.
    return x, findInterval

### 3b: Cubic splines ###

def cubicSpline(x_data, f_data):
    """
    Function which generates a cubic spline interpolation for a tabulated data set, composed of
    data = [x_data, f_data]
    """
    # Check that the arguments passed are a valid data set for interpolation.
    interpolantCheck(x_data,f_data)

    # Precalculations for spline calculation
    def h(i):
        """
        Returns the difference between adjacent x values for a given index i,
        i.e - x[i+1] - x[i]. Forms the adjacent diagonals of the matrix.
        Index i should range from 0 to n-1.
        """
        if type(i) is not int:
            raise TypeError('index must be an integer.')
        return x_data[i+1] - x_data[i]

    def b(i):
        """
        Finds the gradient of the function at the interval between point i and i+1.
        Index i should range between 0 and n-1.
        """
        return (f_data[i+1]-f_data[i])/h(i)

    def v(i):
        """
        Diagonal elements of the matrix. Index should range between 1 and n-1.
        """
        return 2*(h(i-1) + h(i))

    def u(i):
        """
        The 'b' vector in the matrix eqn. Index should range betwen 1 and n-1.
        """
        return 6*(b(i)-b(i-1))

    N = len(x_data)
    M = np.zeros((N-2,N-2))
    _u = []

    # Populate tridiagonal matrix M
    for i in range(1,N-1):
        _u.append(u(i))
        if i < N-2:
            M[i-1][i-1] = v(i)
            M[i][i-1] = h(i)
            M[i-1][i] = h(i)
        else:
            M[i-1][i-1] = v(i)

    # Uses Crout() from Q2 to decompose matrix into L and U for matrix equation solving.
    L, U = Crout(M)

    # Solves matrix equation L.U._z = _u
    _z = solveMatrices(L, U, _u)
    # Add 0 to start and end of list (natural boundary conditions, since z is list of 2nd derivatives)
    z = [0]
    z.extend(_z)
    z.append(0)

    # Generate a set of x values in the same range as the data points to plot the interpolated function with.
    x_num = 100*N # Number of fit line points must be much larger than N to show a continuouis line.
    x = np.linspace(min(x_data), max(x_data), x_num)

    def splineFunc(i):
        """
        Function to generate a function that interpolates between the ith and (i+1)th 
        data points in x_data. For a given i, it calculates all the parameters in the functions
        defined below and then returns the function that can be used to calculate the value of the 
        function at that point.
        """
        def A(_x, i):
            return (x_data[i+1] - _x)/h(i)
        def B(_x, i):
            return (_x - x_data[i])/h(i)
        def C(_x, i):
            return ((A(_x, i)**3 - A(_x, i))*h(i)**2)/6
        def D(_x, i):
            return ((B(_x, i)**3 - B(_x, i))*h(i)**2)/6
        def func(_x):
            """
            Returns the cubic spline that meets the conditions on the derivatives of f.
            """
            return A(_x, i)*f_data[i] + B(_x, i)*f_data[i+1] + C(_x, i)*z[i] + D(_x, i)*z[i+1]
        return func

    # A list of interpolant functions for every data interval in the set.
    splineList = []
    for i in range(N-1):
        splineList.append(splineFunc(i))

    def selectSpline(_x):
        """
        Given a data point _x, this function identifies which interval (i) the point lies in.
        The function returns the corresponding interpolant function for that interval.
        """
        for i in range(N-1):
            if _x >= x_data[i] and _x < x_data[i+1]:
                localSpline = splineList[i]
                return localSpline(_x)

    # Returns the evenly spaced x values and the cubic spline function.
    return x, selectSpline

### 3c: Uses both interpolant functions and plot them on the same graph. ###

def plotBothInterpolants(x_data, f_data):
    """
    The plotting function which calls both the linear interpolator and the cubic spline algorithm.
    """
    x_c, func_cubic = cubicSpline(x_data, f_data)
    x_l, func_lin = interpolate(x_data, f_data)
    print(f_data)

    plt.rcParams['axes.facecolor'] = 'black'
    plt.plot(x_c, [func_cubic(i) for i in x_c], color = 'blue', label = 'Cubic')
    plt.plot(x_l, [func_lin(i) for i in x_l], color = 'red', label = 'Linear')
    plt.plot(x_data, f_data, '+', color = 'white', label = 'Data')
    plt.rcParams["legend.facecolor"] = 'white'
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    return None

### Q4a: Convolve a square pulse with a Gaussian ###

def make_t(start, end, number):
    """
    Generates an array to use as the time domain values.
    """
    return np.linspace(start, end, number)

def h(_t):
    """
    The square pulse function h as described in the problem.
    """
    if _t>=3 and _t<=5:
        return 4
    else:
        return 0

def g(_t):
    """
    The Gaussian function described in the problem.
    """
    return  np.exp((-_t**2)/2)/np.sqrt(2*np.pi)

def convolution(h, g, t):
    """
    Function to perform the convolution using numpy.fft module. Uses the
    convolution theorem, f conv g = inverse FT(FT(f)*FT(g)).
    """
    dt = abs(t[0]-t[1]) # Normalisation factor

    # Fourier transforms input functions
    F_h = np.fft.fft([h(_t) for _t in t])
    F_g = np.fft.fft([g(_t) for _t in t])
    # Unnormalised convolution. numpy.fft.ifft returns a complex array
    # so real values must be taken.
    h_conv_g_unnorm = [np.real(val) for val in np.fft.ifft(F_h*F_g)]
    # Normalises the convolution.
    h_conv_g = [i * dt for i in h_conv_g_unnorm]
    # Shift the time domain.
    t_shift = np.fft.ifftshift(t)
    return t_shift, h_conv_g

### Q4b: plot all three functions against t ###

def q4_plot(start=-10, end=10, samples=1024):
    """
    Plots all three functions on the same axes - h, g and h convoluted with g.
    Start and end default to -10 and 10 to include all relevant features of the functions.
    Samples defaults to 1024 to avoid undersampling the behaviour of the square pulse at the value change.
    """
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams["legend.facecolor"] = 'white'
    # Make t values
    t = make_t(start, end, samples) 
    # Plot h and g functions
    plt.plot(t, [h(i) for i in t ], label = 'h(t)')
    plt.plot(t, [g(i) for i in t ], label = 'g(t)')
    t_shift, conv = convolution(h, g, t)
    # Plot convolution
    plt.plot(t_shift, conv, 'white', label = 'Convolution')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('f')
    plt.show()
    return None

### 5a: Histogram from uniform deviate generator. ###

def uniformRandom(seed, samples):
    """
    Generates an array of size 'samples' of pseudo-random numbers given an input seed.
    Numbers are all between 0 and 1 and are generated using the Mersenne Twister algorithm
    in random.random().
    """
    samples = int(samples)
    i = 0
    r = np.zeros(samples) # Initialise output array
    while i < samples: # Iterate for number of samples
        random.seed(seed) # Set seed for the random algorithm
        r_new = random.random()
        r[i] = r_new
        seed = r_new # Set the new seed to continue the sequence
        i += 1
    return r

# def uniformFunc(_x, samples, bins):
#     return samples/bins

def plotRandom(r, num_bins, pdf = None):
    """
    Plots an array of randomly distributed numbers against an input
    function, which should be the PDF.
    """
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['legend.facecolor'] = 'white'
    if pdf is not None:
        # Plot the PDF
        x = np.linspace(min(r), max(r), 1000)
        plt.plot(x, [pdf(i) for i in x], 'b', label = "PDF")
    # Plot the random numbers
    plt.hist(r, bins = num_bins, density = True, label = "Data")
    plt.xlabel('Random Number')
    plt.ylabel('Scaled Frequency')
    plt.legend()
    plt.show()
    return None

def pdf_5b(_x):
    # _x is used here to distinguish between a dummy variable representing a float and
    # the uniform deviate array x used throughout the code.
    return 0.5*np.cos(_x/2)

def inv_func_5b(_x):
    return 2*np.arcsin(_x)

def transformDeviate(x):
    """
    Performs the transformation method on a uniform deviate x to return a non-uniform 
    deviate y with PDF 0.5*cos(x/2).
    """
    return [inv_func_5b(i) for i in x]

### 5c: Rejection method for P(x) = (2/pi) * cos^2(x/2). ###

def pdf_5c(_x):
    return (2/np.pi)*(np.cos(_x/2)**2)

def comparison(_x):
    """
    Use arcsin(x), as it will produce a distribution twice the size of the PDF.
    """
    return 1.5 * inv_func_5b(_x)

def rejection(seed, samples):

    randomOut = []
    samples = int(samples)
    random.seed(seed)

    for j in range(samples):
        y_i = random.random() * np.pi
        p_i = comparison(random.random())
        if pdf_5c(y_i) >= p_i:
            randomOut.append(y_i)
    return randomOut
