from math import sqrt


def betm_from_mach(mach: float = 0.0) -> float:
    return sqrt(1.0 - mach**2)

def read_dat(datfilepath: str) -> tuple[str, list[float], list[float]]:
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
