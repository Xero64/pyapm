from numpy.matlib import matrix, divide

def diff(mat: matrix, axis: int = 0):
    if axis == 0:
        return mat[1:, :]-mat[:-1, :]
    if axis == 1:
        return mat[:, 1:]-mat[:, :-1]

def mean(mat: matrix, axis: int = 0):
    if axis == 0:
        return (mat[1:, :]+mat[:-1, :])/2
    if axis == 1:
        return (mat[:, 1:]+mat[:, :-1])/2

def derivative(mat: matrix, var: matrix, axis: int = 0):
    return divide(diff(mat, axis), diff(var, axis))
