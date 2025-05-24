from numpy import asarray, linspace
from pygeom.geom1d.cubicspline1d import CubicSpline1D
from pygeom.geom1d.linearspline1d import LinearSpline1D


class PanelFunction():
    var = None
    interp = None
    values = None
    spline = None

    def __init__(self, var: str, spacing: str, interp: str, values: list):
        self.var = var
        self.spacing = spacing
        self.interp = interp
        self.values = asarray(values)

    def set_spline(self, bmax: float):
        if self.spacing == 'equal':
            num = self.values.size
            nspc = linspace(0.0, 1.0, num)
            spc = bmax*nspc
        if self.interp == 'linear':
            self.spline = LinearSpline1D(spc, self.values)
        elif self.interp == 'cubic':
            self.spline = CubicSpline1D(spc, self.values)

    def interpolate(self, b: float):
        return self.spline.evaluate_points_at_t(b)


def panelfunction_from_json(funcdata: dict):
    var = funcdata["variable"]
    if "spacing" in funcdata:
        spacing = funcdata["spacing"]
    else:
        spacing = "equal"
    if "interp" in funcdata:
        interp = funcdata["interp"]
    else:
        interp = "linear"
    values = funcdata["values"]
    return PanelFunction(var, spacing, interp, values)
