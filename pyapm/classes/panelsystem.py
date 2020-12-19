from json import load
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector, MatrixVector
from pygeom.matrix3d import solve_matrix_vector, elementwise_dot_product
from typing import Dict, List
from .panel import Panel
from .grid import Grid
from .horseshoe import HorseShoe
from numpy.matlib import zeros, matrix
from time import perf_counter

class PanelSystem(object):
    name: str = None
    grds: Dict[int, Grid] = None
    pnls: Dict[int, Panel] = None
    bref: float = None
    cref: float = None
    sref: float = None
    rref: float = None
    results = None
    _hsvs: List[HorseShoe] = None
    _numgrd: int = None
    _numpnl: int = None
    _pnts: MatrixVector = None
    _nrms: MatrixVector = None
    _pnla: matrix = None
    _pntr: MatrixVector = None
    _apd: matrix = None
    _aps: matrix = None
    _aph: matrix = None
    _apm: matrix = None
    _avd: MatrixVector = None
    _avs: MatrixVector = None
    _avh: MatrixVector = None
    _avm: MatrixVector = None
    _sig: MatrixVector = None
    _bps: MatrixVector = None
    _ans: matrix = None
    _mu: MatrixVector = None
    def __init__(self, name: str, grds: Dict[int, Grid], pnls: Dict[int, Panel],
                 bref: float, cref: float, sref: float, rref: Vector):
        self.name = name
        self.grds = grds
        self.pnls = pnls
        self.bref = bref
        self.cref = cref
        self.sref = sref
        self.rref = rref
        self.results = {}
        self.update()
    def update(self):
        for ind, pnl in enumerate(self.pnls.values()):
            grds = [self.grds[gid] for gid in pnl.gids]
            pnl.set_grids(grds)
            pnl.set_index(ind)
    def reset(self):
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None
    @property
    def numgrd(self):
        if self._numgrd is None:
            self._numgrd = self.grds.shape[0]
        return self._numgrd
    @property
    def numpnl(self):
        if self._numpnl is None:
            self._numpnl = len(self.pnls)
        return self._numpnl
    @property
    def pnts(self):
        if self._pnts is None:
            self._pnts = zero_matrix_vector((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._pnts[pnl.ind, 0] = pnl.pnto
        return self._pnts
    @property
    def pntr(self):
        if self._pntr is None:
            self._pntr = self.pnts-self.rref
        return self._pntr
    @property
    def nrms(self):
        if self._nrms is None:
            self._nrms = zero_matrix_vector((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._nrms[pnl.ind, 0] = pnl.nrm
        return self._nrms
    @property
    def pnla(self):
        if self._pnla is None:
            self._pnla = zeros((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._pnla[pnl.ind, 0] = pnl.area
        return self._pnla
    @property
    def hsvs(self):
        if self._hsvs is None:
            self._hsvs = []
            for pnl in self.pnls.values():
                self._hsvs = self._hsvs + pnl.hsvs
        return self._hsvs
    @property
    def bps(self):
        if self._bps is None:
            self._bps = -1.0*self.aps*self.sig
        return self._bps
    @property
    def apd(self):
        if self._apd is None:
            self.assemble_panels(False)
        return self._apd
    @property
    def avd(self):
        if self._avd is None:
            self.assemble_panels(False)
        return self._avd
    @property
    def aps(self):
        if self._aps is None:
            self.assemble_panels(False)
        return self._aps
    @property
    def avs(self):
        if self._avs is None:
            self.assemble_panels(False)
        return self._avs
    @property
    def aph(self):
        if self._aph is None:
            self.assemble_horseshoes(False)
        return self._aph
    @property
    def avh(self):
        if self._avh is None:
            self.assemble_horseshoes(False)
        return self._avh
    @property
    def apm(self):
        if self._apm is None:
            self._apm = self.apd + self.aph
        return self._apm
    @property
    def avm(self):
        if self._avm is None:
            self._avm = self.avd + self.avh
        return self._avm
    @property
    def ans(self):
        if self._ans is None:
            self._ans = elementwise_dot_product(self.nrms.repeat(self.numpnl, axis=1), self.avs)
        return self._ans
    @property
    def sig(self):
        if self._sig is None:
            self._sig = -self.nrms
            # self._sig = solve_matrix_vector(self.ans, -self.nrms)
        return self._sig
    @property
    def mu(self):
        if self._mu is None:
            self._mu = solve_matrix_vector(self.apm, self.bps)
        return self._mu
    def assemble_panels(self, time: bool = True):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        self._apd = zeros(shp, dtype=float)
        self._aps = zeros(shp, dtype=float)
        self._avd = zero_matrix_vector(shp, dtype=float)
        self._avs = zero_matrix_vector(shp, dtype=float)
        for pnl in self.pnls.values():
            ind = pnl.ind
            self._apd[:, ind], self._aps[:, ind], self._avd[:, ind], self._avs[:, ind] = pnl.influence_coefficients(self.pnts)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Panel assembly time is {elapsed:.3f} seconds.')
    def assemble_horseshoes(self, time: bool = True):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        self._aph = zeros(shp, dtype=float)
        self._avh = zero_matrix_vector(shp, dtype=float)
        for hsv in self.hsvs:
            ind = hsv.ind
            self._aph[:, ind], self._avh[:, ind] = hsv.doublet_influence_coefficients(self.pnts)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Horse shoe assembly time is {elapsed:.3f} seconds.')
    def solve_system(self, time: bool = True):
        if time:
            start = perf_counter()
        self.mu
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    
def panel_system_json(jsonfilepath):
    with open(jsonfilepath, 'rt') as jsonfile:
        data = load(jsonfile)
    name = data['name']
    bref = data['bref']
    cref = data['cref']
    sref = data['sref']
    xref = data['xref']
    yref = data['yref']
    zref = data['zref']
    rref = Vector(xref, yref, zref)
    grids = {}
    griddata = data['grids']
    for gidstr, gd in griddata.items():
        gid = int(gidstr)
        if 'te' not in gd:
            gd['te'] = False
        grids[gid] = Grid(gid, gd['x'], gd['y'], gd['z'], gd['te'])
    panels = {}
    paneldata = data['panels']
    for pidstr, pd in paneldata.items():
        pid = int(pidstr)
        panels[pid] = Panel(pid, pd['gids'])
    return PanelSystem(name, grids, panels, bref, cref, sref, rref)
