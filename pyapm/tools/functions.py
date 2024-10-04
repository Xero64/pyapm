from numpy import divide, ndarray


def diff(mat: ndarray, axis: int = 0):
    if axis == 0:
        return mat[1:, :]-mat[:-1, :]
    if axis == 1:
        return mat[:, 1:]-mat[:, :-1]

def mean(mat: ndarray, axis: int = 0):
    if axis == 0:
        return (mat[1:, :]+mat[:-1, :])/2
    if axis == 1:
        return (mat[:, 1:]+mat[:, :-1])/2

def derivative(mat: ndarray, var: ndarray, axis: int = 0):
    return divide(diff(mat, axis), diff(var, axis))
