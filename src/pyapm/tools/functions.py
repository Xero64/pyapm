from typing import TYPE_CHECKING

from numpy import divide

if TYPE_CHECKING:
    from numpy.typing import NDArray


def diff(mat: 'NDArray', axis: int = 0):
    if axis == 0:
        return mat[1:, :]-mat[:-1, :]
    if axis == 1:
        return mat[:, 1:]-mat[:, :-1]

def mean(mat: 'NDArray', axis: int = 0):
    if axis == 0:
        return (mat[1:, :]+mat[:-1, :])/2
    if axis == 1:
        return (mat[:, 1:]+mat[:, :-1])/2

def derivative(mat: 'NDArray', var: 'NDArray', axis: int = 0):
    return divide(diff(mat, axis), diff(var, axis))
