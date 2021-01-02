from math import sqrt, cos, sin, radians
from json import load
from typing import List, Dict
from pygeom.geom3d import Vector, Coordinate
from pygeom.matrix3d import MatrixVector, zero_matrix_vector
from .grid import Grid
from .panel import Panel
from numpy.matlib import matrix, absolute, zeros
from ..tools import equal_spacing, semi_cosine_spacing, full_cosine_spacing
from ..tools.airfoil import airfoil_from_dat, Airfoil
from ..tools.naca4 import NACA4

tol = 1e-12

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

def surffunc_from_json(funcdata: dict):
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

class PanelSection(object):
    point: Vector = None
    chord: float = None
    twist: float = None
    airfoil: Airfoil = None
    bnum: int = None
    bspc: str = None
    mirror: bool = None
    cnum: int = None
    xoc: float = None
    zoc: float = None
    tilt: float = None
    bval: float = None
    _profile: MatrixVector = None
    def __init__(self, point: Vector, chord: float, twist: float, airfoil):
        self.point = point
        self.chord = chord
        self.twist = twist
        self.airfoil = airfoil
        self.mirror = False
        self.bnum = 1
        self.bspc = 'equal'
        self.xoc = 0.0
        self.zoc = 0.0
        self.tilt = 0.0
    def mirror_section_in_y(self, ymir: float=0.0):
        point = Vector(self.point.x, ymir-self.point.y, self.point.z)
        chord = self.chord
        twist = self.twist
        airfoil = self.airfoil
        sect = PanelSection(point, chord, twist, airfoil)
        sect.mirror = True
        sect.bnum = self.bnum
        sect.bspc = self.bspc
        sect.xoc = self.xoc
        sect.zoc = self.zoc
        sect.tilt = -self.tilt
        sect.bval = self.bval
        return sect
    def set_cnum(self, cnum: int):
        self.cnum = cnum
        self.airfoil.update(self.cnum)
    @property
    def profile(self):
        if self._profile is None:
            num = self.cnum*2+1
            self._profile = zero_matrix_vector((1, num), dtype=float)
            for i in range(self.cnum+1):
                n = num-i-1
                self._profile[0, i] = Vector(self.airfoil.xl[-i-1], 0.0, self.airfoil.yl[-i-1])
                self._profile[0, n] = Vector(self.airfoil.xu[-i-1], 0.0, self.airfoil.yu[-i-1])
            self._profile.z[absolute(self._profile.z) < tol] = 0.0
            offset = Vector(self.xoc, 0.0, self.zoc)
            self._profile = self._profile-offset
        return self._profile
    def __repr__(self):
        return f'<PanelSection at {self.point:}>'

def panelsection_from_json(sectdata: dict) -> PanelSection:
    xpos = sectdata['xpos']
    ypos = sectdata['ypos']
    zpos = sectdata['zpos']
    point = Vector(xpos, ypos, zpos)
    chord = sectdata['chord']
    airfoilstr = sectdata['airfoil']
    if airfoilstr[-4:] == '.dat':
        airfoil = airfoil_from_dat(airfoilstr)
    elif airfoilstr[0:4].upper() == 'NACA':
        code = airfoilstr[4:].strip()
        if len(code) == 4:
            airfoil = NACA4(code)
    else:
        return ValueError(f'Airfoil identified by {airfoilstr:s} does not exist.')
    if 'twist' not in sectdata:
        twist = 0.0
    else:
        twist = sectdata['twist']
    sect = PanelSection(point, chord, twist, airfoil)
    if 'bnum' in sectdata:
        sect.bnum = sectdata['bnum']
    if 'bspc' in sectdata:
        sect.bspc = sectdata['bspc']
    if 'xoc' in sectdata:
        sect.xoc = sectdata['xoc']
    if 'zoc' in sectdata:
        sect.zoc = sectdata['zoc']
    return sect

class PanelProfile(object):
    point: Vector = None
    chord: float = None
    twist: float = None
    tilt: float = None
    profile: MatrixVector = None
    _crdsys: Coordinate = None
    _shape: MatrixVector = None
    def __init__(self, point: Vector, chord: float, twist: float, tilt: float,
                 profile: MatrixVector):
        self.point = point
        self.chord = chord
        self.twist = twist
        self.tilt = tilt
        self.profile = profile
    @property
    def crdsys(self):
        if self._crdsys is None:
            tilt = radians(self.tilt)
            sintilt = sin(tilt)
            costilt = cos(tilt)
            diry = Vector(0.0, costilt, sintilt)
            twist = radians(self.twist)
            sintwist = sin(twist)
            costwist = cos(twist)
            dirx = Vector(costwist, sintwist*sintilt, -sintwist*costilt)
            dirz = dirx**diry
            self._crdsys = Coordinate(self.point, dirx, diry, dirz)
        return self._crdsys
    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.crdsys.vector_to_global(self.profile*self.chord)+self.point
        return self._shape
    def __repr__(self):
        return f'<PanelProfile at {self.point:}>'

class PanelSheet(object):
    scta: PanelSection = None
    sctb: PanelSection = None
    fncs: Dict[str, PanelFunction] = None
    _bnum: int = None
    _bspc: str = None
    _bdst: List[float] = None
    _prfs: List[PanelProfile] = None
    _shps: MatrixVector = None
    _mirror: bool = None
    def __init__(self, scta: PanelSection, sctb: PanelSection):
        self.scta = scta
        self.sctb = sctb
        self.fncs = {}
    @property
    def mirror(self) -> bool:
        if self._mirror is None:
            self._mirror = self.scta.mirror
        return self._mirror
    @property
    def bnum(self) -> int:
        if self._bnum is None:
            if self.mirror:
                self._bnum = self.sctb.bnum
            else:
                self._bnum = self.scta.bnum
        return self._bnum
    @property
    def bspc(self) -> int:
        if self._bspc is None:
            if self.mirror:
                self._bspc = self.sctb.bspc
            else:
                self._bspc = self.scta.bspc
        return self._bspc
    @property
    def bdst(self) -> matrix:
        if self._bdst is None:
            if self.bspc == 'equal':
                bdst = equal_spacing(self.bnum)
            elif self.bspc == 'semi-cosine':
                bdst = semi_cosine_spacing(self.bnum)
            elif self.bspc == 'full-cosine':
                bdst = full_cosine_spacing(self.bnum)
            if self.mirror:
                bdst = 1.0-bdst
                self._bdst = bdst.tolist()[0]
                self._bdst.reverse()
            else:
                self._bdst = bdst.tolist()[0]
        return self._bdst
    @property
    def prfs(self):
        if self._prfs is None:
            self._prfs = []
            bmin = min(self.scta.bval, self.sctb.bval)
            bmax = max(self.scta.bval, self.sctb.bval)
            brng = bmax-bmin
            pointdir = self.sctb.point-self.scta.point
            profiledir = self.sctb.profile-self.scta.profile
            for bval in self.bdst:
                if self.mirror:
                    bint = bmax-bval*brng
                else:
                    bint = bmin+bval*brng
                point = self.scta.point + bval*pointdir
                if 'chord' in self.fncs:
                    chord = self.fncs['chord'].interpolate(bint)
                else:
                    chord = self.scta.chord*(1-bval)+self.sctb.chord*bval
                if 'twist' in self.fncs:
                    twist = self.fncs['twist'].interpolate(bint)
                else:
                    twist = self.scta.twist*(1-bval)+self.sctb.twist*bval
                if 'tilt' in self.fncs:
                    tilt = self.fncs['tilt'].interpolate(bint)
                else:
                    tilt = self.scta.tilt*(1-bval)+self.sctb.tilt*bval
                profile = self.scta.profile+bval*profiledir
                prof = PanelProfile(point, chord, twist, tilt, profile)
                self._prfs.append(prof)
        return self._prfs
    @property
    def shps(self):
        if self._shps is None:
            numprf = len(self.prfs)
            numvec = self.prfs[0].profile.shape[1]
            self._shps = zero_matrix_vector((numprf, numvec), dtype=float)
            for i, prf in enumerate(self.prfs):
                self._shps[i, :] = prf.shape
        return self._shps
    def __repr__(self):
        return '<PanelSheet>'

class PanelSurface(object):
    name: str = None
    point: Vector = None
    twist: float = None
    mirror: bool = None
    scts: List[PanelSection] = None
    fncs: Dict[str, PanelFunction] = None
    cnum: int = None
    cspc: str = None
    twist: float = None
    _shts: List[PanelSheet] = None
    _shps: MatrixVector = None
    gidmat: matrix = None
    grds: Dict[int, Grid] = None
    pnls: Dict[int, Panel] = None
    def __init__(self, name: str, point: Vector, twist: float, mirror: bool,
                 scts: List[PanelSection], fncs: List[PanelFunction]):
        self.name = name
        self.point = point
        self.twist = twist
        self.scts = scts
        self.mirror = mirror
        self.fncs = fncs
        self.update()
    def update(self):
        bval = 0.0
        for i in range(len(self.scts)):
            self.scts[i].bval = bval
            if i < len(self.scts)-1:
                delx = self.scts[i+1].point.x - self.scts[i].point.x
                dely = self.scts[i+1].point.y - self.scts[i].point.y
                delz = self.scts[i+1].point.z - self.scts[i].point.z
                bval += sqrt(delx**2 + dely**2 + delz**2)
        for fnc in self.fncs.values():
            fnc.set_spline(bval)
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
    def shps(self):
        if self._shps is None:
            numprf = sum([sht.shps.shape[0] for sht in self.shts])-len(self.shts)+1
            numvec = self.cnum*2+1
            self._shps = zero_matrix_vector((numprf, numvec), dtype=float)
            i = 0
            for sht in self.shts:
                n = i + sht.shps.shape[0]
                self._shps[i:n, :] = sht.shps
                i = n-1
        return self._shps
    def mesh(self, gid: int, pid: int):
        # Generate Surface Grid ID Matrix
        self.gidmat = gidmat = zeros(self.shps.shape, dtype=int)
        for j in range(self.shps.shape[1]):
            for i in range(self.shps.shape[0]):
                gid += 1
                self.gidmat[i, j] = gid
        self.grds = {}
        self.pnls = {}
        for i in range(self.gidmat.shape[0]):
            for j in range(self.gidmat.shape[1]):
                te = False
                if j == 0 or j == self.gidmat.shape[1]-1:
                    te = True
                gidij = self.gidmat[i, j]
                grdvc = self.shps[i, j]
                self.grds[gidij] = Grid(gidij, grdvc.x, grdvc.y, grdvc.z, te)
        for j in range(gidmat.shape[0]-1):
            for i in range(gidmat.shape[1]-1):
                pid += 1
                gids = [gidmat[j+1, i], gidmat[j, i], gidmat[j, i+1], gidmat[j+1, i+1]]
                pnl = Panel(pid, gids)
                self.pnls[pid] = pnl
                # grds = [self.grds[gid] for gid in pnl.gids]
                # pnl.set_grids(grds)
        # # Close Trailing Edge
        # for j in range(ynum-1):
        #     pid += 1
        #     gids = [gidmat[j+1, -1], gidmat[j, -1], gidmat[j, 0], gidmat[j+1, 0]]
        #     panels[pid] = Panel(pid, gids)

        # n = 2*xznum
        # # Close Ends
        # for i in range(xznum):
        #     pid += 1
        #     if i == xznum-1:
        #         gids = [gidmat[0, i+1], gidmat[0, i], gidmat[0, n-i]]
        #     else:
        #         gids = [gidmat[0, i+1], gidmat[0, i], gidmat[0, n-i], gidmat[0, n-i-1]]
        #     panels[pid] = Panel(pid, gids)

        # for i in range(xznum):
        #     pid += 1
        #     if i == xznum-1:
        #         gids = [gidmat[-1, i+1], gidmat[-1, i], gidmat[-1, n-i]]
        #     else:
        #         gids = [gidmat[-1, i+1], gidmat[-1, i], gidmat[-1, n-i], gidmat[-1, n-i-1]]
        #     gids.reverse()
        #     panels[pid] = Panel(pid, gids)
        return gid, pid
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
            func = surffunc_from_json(funcdata)
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
    surf = PanelSurface(name, point, twist, mirror, sects, funcs)
    if 'cnum' in surfdata:
        cnum = surfdata['cnum']
        surf.set_chord_spacing(cnum)
    return surf
