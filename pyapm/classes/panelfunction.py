from pygeom.geom1d import LinearSpline, CubicSpline
from ..tools import equal_spacing

class PanelFunction(object):
    var = None
    interp = None
    values = None
    spline = None
    def __init__(self, var: str, spacing: str, interp: str, values: list):
        self.var = var
        self.spacing = spacing
        self.interp = interp
        self.values = values
    def set_spline(self, bmax: float):
        if self.spacing == 'equal':
            num = len(self.values)
            nspc = equal_spacing(num-1)
            spc = [bmax*nspci for nspci in nspc]
        if self.interp == 'linear':
            self.spline = LinearSpline(spc, self.values)
        elif self.interp == 'cubic':
            self.spline = CubicSpline(spc, self.values)
    def interpolate(self, b: float):
        return self.spline.single_interpolate_spline(b)

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
