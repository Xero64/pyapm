# from math import pi, cos, asin
from numpy.matlib import matrix, arange, cos, pi

def normalise_spacing(spacing: matrix) -> matrix:
    smin = spacing.min()
    smax = spacing.max()
    return (spacing-smin)/(smax-smin)

def semi_cosine_spacing(num: int) -> matrix:
    th = matrix(arange(num, -1, -1), dtype=float)*pi/2/num
    spc = cos(th)
    spc[0, 0] = 0.0
    return spc

def full_cosine_spacing(num: int) -> list:
    th = matrix(arange(num, -1, -1), dtype=float)*pi/num
    spc = (cos(th)+1.0)/2
    return spc

def equal_spacing(num: int) -> list:
    return matrix(arange(0, num+1, 1), dtype=float)/num
