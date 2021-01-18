from math import sqrt
from typing import List, Dict
from pygeom.geom3d import Vector
from .grid import Grid
from .panel import Panel
from .panelsection import PanelSection, panelsection_from_json
from .panelfunction import PanelFunction, panelfunction_from_json
from .panelsheet import PanelSheet
from .panelstrip import PanelStrip
from .panelprofile import PanelProfile

class PanelSurface(object):
    name: str = None
    point: Vector = None
    twist: float = None
    mirror: bool = None
    scts: List[PanelSection] = None
    fncs: Dict[str, PanelFunction] = None
    close: bool = None
    cnum: int = None
    cspc: str = None
    twist: float = None
    _shts: List[PanelSheet] = None
    _strps: List[PanelStrip] = None
    _prfs: List[PanelProfile] = None
    _area: float = None
    grds: List[Grid] = None
    pnls: List[Panel] = None
    def __init__(self, name: str, point: Vector, twist: float, mirror: bool,
                 scts: List[PanelSection], fncs: List[PanelFunction], close: bool):
        self.name = name
        self.point = point
        self.twist = twist
        self.scts = scts
        self.mirror = mirror
        self.fncs = fncs
        self.close = close
        self.update()
    def update(self):
        bval = 0.0
        for i in range(len(self.scts)):
            self.scts[i].bval = bval
            self.scts[i].bpos = bval
            if i < len(self.scts)-1:
                delx = self.scts[i+1].point.x - self.scts[i].point.x
                dely = self.scts[i+1].point.y - self.scts[i].point.y
                delz = self.scts[i+1].point.z - self.scts[i].point.z
                bval += sqrt(delx**2 + dely**2 + delz**2)
        for fnc in self.fncs.values():
            fnc.set_spline(bval)
            if fnc.var == 'twist':
                for sct in self.scts:
                    sct.twist = fnc.interpolate(sct.bval)
            if fnc.var == 'chord':
                for sct in self.scts:
                    sct.chord = fnc.interpolate(sct.bval)
            if fnc.var == 'tilt':
                for sct in self.scts:
                    sct.set_tilt(fnc.interpolate(sct.bval))
        if self.mirror:
            ymir = self.scts[0].point.y
            scts = [sct.mirror_section_in_y(ymir=ymir) for sct in self.scts]
            scts.reverse()
            self.scts = scts[:-1] + self.scts
    def set_chord_spacing(self, cnum: int):
        self.cnum = cnum
        for sct in self.scts:
            sct.set_cnum(self.cnum)
    @property
    def shts(self):
        if self._shts is None:
            self._shts = []
            for i in range(len(self.scts)-1):
                self._shts.append(PanelSheet(self.scts[i], self.scts[i+1]))
                self._shts[i].fncs = self.fncs
        return self._shts
    @property
    def strps(self):
        if self._strps is None:
            self._strps = []
            for sht in self.shts:
                self._strps += sht.strps
        return self._strps
    @property
    def prfs(self):
        if self._prfs is None:
            self._prfs = []
            for sht in self.shts:
                scta = sht.scta
                sctb = sht.sctb
                self._prfs.append(scta)
                self._prfs += sht.prfs
            self._prfs.append(sctb)
        return self._prfs
    @property
    def strpb(self):
        return [strp.bpos for strp in self.strps]
    @property
    def strpy(self):
        return [strp.ypos for strp in self.strps]
    @property
    def strpz(self):
        return [strp.zpos for strp in self.strps]
    @property
    def prfb(self):
        return [prf.bpos for prf in self.prfs]
    @property
    def prfy(self):
        return [prf.point.y for prf in self.prfs]
    @property
    def prfz(self):
        return [prf.point.z for prf in self.prfs]
    @property
    def area(self):
        if self._area is None:
            self._area = 0.0
            for sht in self.shts:
                self._area += sht.area
        return self._area
    def mesh_grids(self, gid: int):
        self.grds = []
        for sht in self.shts:
            scta = sht.scta
            sctb = sht.sctb
            gid = scta.mesh_grids(gid)
            self.grds += scta.grds
            gid = sht.mesh_grids(gid)
            self.grds += sht.grds
        gid = sctb.mesh_grids(gid)
        self.grds += sctb.grds
        return gid
    def mesh_panels(self, pid: int):
        self.pnls = []
        for sht in self.shts:
            pid = sht.mesh_panels(pid)
            for pnl in sht.pnls:
                pnl.srfc = self
                self.pnls.append(pnl)
        if self.close:
            for sct in self.scts:
                pid = sct.mesh_panels(pid)
                for pnl in sct.pnls:
                    pnl.srfc = self
                    self.pnls.append(pnl)
        return pid
    # def mesh_tip_panels(self, pid: int):
    #     scta = self.scts[0]
    #     pid = scta.mesh_panels(pid, reverse=True)
    #     sctb = self.scts[-1]
    #     pid = sctb.mesh_panels(pid, reverse=False)
    #     return pid
    @property
    def pinds(self):
        pinds = []
        for pnl in self.pnls:
            pinds.append(pnl.ind)
        return pinds
    def __repr__(self):
        return f'<PanelSurface: {self.name:s}>'

def linear_interpolate_none(x: list, y: list):
    for i, yi in enumerate(y):
        if yi is None:
            for j in range(i, -1, -1):
                if y[j] is not None:
                    a = j
                    break
            for j in range(i, len(y)):
                if y[j] is not None:
                    b = j
                    break
            xa, xb = x[a], x[b]
            ya, yb = y[a], y[b]
            y[i] = (yb-ya)/(xb-xa)*(x[i]-xa)+ya
    return y

def panelsurface_from_json(surfdata: dict, display: bool=False):
    name = surfdata['name']
    if 'mirror' in surfdata:
        mirror = surfdata['mirror']
    else:
        mirror = False
    if display: print(f'Loading Surface: {name:s}')
    # Read Section Variables
    sects = []
    for sectdata in surfdata['sections']:
        sect = panelsection_from_json(sectdata)
        sects.append(sect)
    # Linear Interpolate Missing Variables
    x, y, z, c, a = [], [], [], [], []
    for sect in sects:
        x.append(sect.point.x)
        y.append(sect.point.y)
        z.append(sect.point.z)
        c.append(sect.chord)
        a.append(sect.twist)
    if None in y:
        if None is z:
            return ValueError
        else:
            y = linear_interpolate_none(z, y)
    else:
        z = linear_interpolate_none(y, z)
    lensects = len(sects)
    b = [0.0]
    for i in range(lensects-1):
        bi = b[i]+sqrt((y[i+1]-y[i])**2+(z[i+1]-z[i])**2)
        b.append(bi)
    x = linear_interpolate_none(b, x)
    c = linear_interpolate_none(b, c)
    a = linear_interpolate_none(b, a)
    for i, sect in enumerate(sects):
        sect.point.x = x[i]
        sect.point.y = y[i]
        sect.point.z = z[i]
        sect.chord = c[i]
        sect.twist = a[i]
    # Read in Function Data
    funcs = {}
    if 'functions' in surfdata:
        for funcdata in surfdata['functions']:
            func = panelfunction_from_json(funcdata)
            funcs[func.var] = func
    # Entire Surface Position
    xpos, ypos, zpos = 0.0, 0.0, 0.0
    if 'xpos' in surfdata:
        xpos = surfdata['xpos']
    if 'ypos' in surfdata:
        ypos = surfdata['ypos']
    if 'zpos' in surfdata:
        zpos = surfdata['zpos']
    point = Vector(xpos, ypos, zpos)
    twist = 0.0
    if 'twist' in surfdata:
        twist = surfdata['twist']
    for sect in sects:
        sect.offset_position(xpos, ypos, zpos)
        sect.offset_twist(twist)
    close = True
    if 'close' in surfdata:
        close = surfdata['close']
    surf = PanelSurface(name, point, twist, mirror, sects, funcs, close)
    if 'cnum' in surfdata:
        cnum = surfdata['cnum']
        surf.set_chord_spacing(cnum)
    return surf
