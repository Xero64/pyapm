from math import sqrt
from .spacing import equal_spacing, semi_cosine_spacing, full_cosine_spacing
from .spacing import linear_bias_left, linear_bias_right

def betm_from_mach(mach: float=0.0):
    return sqrt(1-mach**2)
