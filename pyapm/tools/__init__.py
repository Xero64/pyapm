from math import sqrt
from .spacing import equal_spacing, semi_cosine_spacing, full_cosine_spacing
from .spacing import linear_bias_left, linear_bias_right

def betm_from_mach(mach: float=0.0):
    return sqrt(1-mach**2)

def read_dat(datfilepath: str):
    name = ''
    x = []
    y = []
    with open(datfilepath, 'rt') as file:
        for i, line in enumerate(file):
            line = line.rstrip('\n')
            if i == 0:
                name = line.strip()
            else:
                split = line.split()
                if len(split) == 2:
                    x.append(float(split[0]))
                    y.append(float(split[1]))
    return name, x, y
