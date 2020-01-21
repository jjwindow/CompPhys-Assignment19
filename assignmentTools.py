"""
assignmentTools library for Computational Physics Assignment 1, Autumn 2019.
Made by J. J. Window

Collection of functions developed during the computational physics course, moved 
into one library to keep code as succinct and clear as possible.

"""

import numpy as np
import random
import copy
import functools

def matrixCheck(A):
    """
    Checks if a function input is a valid matrix. Verifies the data types of both
    the matrix and all rows, as well as verifying that row length is consistent.
    If any conditions are not me tthen a TypeError is raised
    Returns a boolean value isSquare, which is True for a square matrix and False otherwise.
    """
    if (type(A) is not list) and (type(A) is not np.ndarray):
        raise TypeError('Input matrix is not a list or numpy array.')
    for row in A:
        if (type(row) is not list) and (type(row) is not np.ndarray):
            raise TypeError('All rows in input matrix must be of type list or numpy array.')
    lastRow = A[0]
    for i in range(1, len(A)):
        if len(A[i]) != len(lastRow):
            raise TypeError('All rows in input matrix must have same length.')
    isSquare = False
    if len(A) == len(A[0]):
        isSquare = True
    return isSquare

def matrixSize(A):
    matrixCheck(A) # Checks the input is a matrix.
    # Returns the dimensions of the matrix as [rows, columns]
    return [len(A), len(A[0])]

def diagonalPairCheck(L, U):
    """
    Checks if a pair of lower and upper diagonal matrices are in the correct
    form. Does not check that they meet the Crout decomposition criteria,
    L[i][i] = 1, because solving a matrix equation does not depend on this
    condition, so it is kept as general as possible.
    """
    # Check the matrices are square and of the same size first
    if matrixCheck(L) is False:
        raise TypeError('L is not a square matrix.')
    elif matrixCheck(U) is False:
        raise TypeError('U is not a square matrix.')
    elif matrixSize(L) != matrixSize(U):
        raise TypeError('L and U must have the same dimensions')
    
    N = len(L) # Number of rows and columns in L and U.

    # ### UNCOMMENT TO INCLUDE CROUT DECOMPOSITION CRITERIA ###
    # for i in range(N):
    #     if L[i,i] != 1:
    #         raise TypeError('All diagonal elements of L must equal 1.')
    
    for j in range(N):
        for i in range(j+1,N):
            # Checks upper diagonal elements of L are 0.
            if L[j][i] != 0:
                raise TypeError('Matrix L is not in lower diagonal form.')
        for i in range(1, j-1):
            # Checks lower diagonal elements of U are 0.
            if U[j][i] != 0:
                raise TypeError('Matrix U is not in upper diagonal form.')
    # If no exceptions raised then matrices are in correct form, return True.
    return True

def interpolantCheck(x_data, f_data):
    """
    Checks that a given data set contains two lists or numpy arrays, and that
    they are of the same length.
    """
    if (type(x_data) is not list) and (type(x_data) is not np.ndarray):
        raise TypeError('x_data must be a list or numpy array.')
    if (type(f_data) is not list) and (type(f_data) is not np.ndarray):
        raise TypeError('f_data must be a list or numpy array.')
    if len(x_data) != len(f_data):
        raise TypeError('x_data and f_data must have same size.')
    return None


