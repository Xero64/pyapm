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
            from pyvlm.tools import equal_spacing
            nspc = equal_spacing(num-1)
            spc = [bmax*nspci for nspci in nspc]
        if self.interp == 'linear':
            from pygeom.geom1d import LinearSpline
            self.spline = LinearSpline(spc, self.values)
        elif self.interp == 'cubic':
            from pygeom.geom1d import CubicSpline
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
