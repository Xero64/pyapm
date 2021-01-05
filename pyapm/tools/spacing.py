from numpy import array, arange, cos, pi
from typing import List

def normalise_spacing(spacing: array):
    smin = spacing.min()
    smax = spacing.max()
    return (spacing-smin)/(smax-smin)

def semi_cosine_spacing(num: int):
    th = arange(num, -1, -1, dtype=float)*pi/2/num
    spc = cos(th)
    spc[0] = 0.0
    return spc.tolist()

def full_cosine_spacing(num: int) -> list:
    th = arange(num, -1, -1, dtype=float)*pi/num
    spc = (cos(th)+1.0)/2
    return spc.tolist()

def equal_spacing(num: int) -> list:
    spc = arange(0, num+1, 1, dtype=float)/num
    return spc.tolist()

def linear_bias_left(spc: List[float], ratio: float):
    ratio = abs(ratio)
    if ratio > 1.0:
        ratio = 1.0/ratio
    m = 1.0 - ratio
    return [s*(ratio + m*s) for s in spc]

def linear_bias_right(spc: List[float], ratio: float):
    ratio = abs(ratio)
    if ratio > 1.0:
        ratio = 1.0/ratio
    m = 1.0 - ratio
    return [1.0 - (1.0 - s)*(ratio + m*(1.0 - s)) for s in spc]
