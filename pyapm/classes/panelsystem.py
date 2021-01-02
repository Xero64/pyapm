from json import load
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector, MatrixVector
from pygeom.matrix3d import solve_matrix_vector, elementwise_dot_product
from typing import Dict, List
from .panel import Panel
from .grid import Grid
from .horseshoe import HorseShoe
from numpy.matlib import zeros, matrix, divide, repeat
from time import perf_counter

class PanelSystem(object):
    name: str = None
    grds: Dict[int, Grid] = None
    pnls: Dict[int, Panel] = None
    bref: float = None
    cref: float = None
    sref: float = None
    rref: float = None
    srfcs = None
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
    _anm: matrix = None
    _bnm: matrix = None
    _mu: MatrixVector = None
    _ar: float = None
    _area: float = None
    _grdavg: matrix = None
    _grdvec: MatrixVector = None
    _gpd: matrix = None
    _gps: matrix = None
    _gph: matrix = None
    _gpm: matrix = None
    _gvd: MatrixVector = None
    _gvs: MatrixVector = None
    _gvh: MatrixVector = None
    _gvm: MatrixVector = None
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
        for ind, grd in enumerate(self.grds.values()):
            grd.set_index(ind)
        for ind, pnl in enumerate(self.pnls.values()):
            grds = [self.grds[gid] for gid in pnl.gids]
            pnl.set_grids(grds)
            pnl.set_index(ind)
    def reset(self):
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None
    def set_horseshoes(self, diro: Vector):
        self._hsvs = []
        for pnl in self.pnls.values():
            pnl.set_horseshoes(diro)
            self._hsvs = self._hsvs + pnl.hsvs
        self._aph = None
        self._avh = None
        self._apm = None
        self._avm = None
        self._anm = None
        self._gph = None
        self._gvh = None
        self._gpm = None
        self._gvm = None
    @property
    def ar(self):
        if self._ar is None:
            self._ar = self.bref**2/self.sref
        return self._ar
    @property
    def area(self):
        if self._area is None:
            self._area = 0.0
            for pnl in self.pnls.values():
                self._area += pnl.area
        return self._area
    @property
    def grdavg(self):
        if self._grdavg is None:
            grdavg = zeros((self.numgrd, self.numpnl), dtype=float)
            for pnl in self.pnls.values():
                pind = pnl.ind
                for i, grd in enumerate(pnl.grds):
                    gind = grd.ind
                    grdavg[gind, pind] = pnl.grdinva[i]
            sumgrdavg = repeat(grdavg.sum(axis=1), self.numpnl, axis=1)
            self._grdavg = divide(grdavg, sumgrdavg)
        return self._grdavg
    @property
    def numgrd(self):
        if self._numgrd is None:
            self._numgrd = len(self.grds)
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
    def bnm(self):
        if self._bnm is None:
            self._bnm = -self.nrms-self.ans*self.sig
        return self._bnm
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
    def anm(self):
        if self._anm is None:
            self._anm = elementwise_dot_product(self.nrms.repeat(self.numpnl, axis=1), self.avm)
        return self._anm
    @property
    def grdvec(self):
        if self._grdvec is None:
            self._grdvec = zero_matrix_vector((self.numgrd, 1), dtype=float)
            for grd in self.grds.values():
                self._grdvec[grd.ind, 0] = grd
        return self._grdvec
    def assemble_grids(self):
        pass
    @property
    def gpd(self):
        if self._gpd is None:
            self.assemble_grid_panels(False)
        return self._gpd
    @property
    def gps(self):
        if self._gps is None:
            self.assemble_grid_panels(False)
        return self._gps
    @property
    def gph(self):
        if self._gph is None:
            self.assemble_grid_horseshoes(False)
        return self._gph
    @property
    def gvd(self):
        if self._gvd is None:
            self.assemble_grid_panels(False)
        return self._gvd
    @property
    def gvs(self):
        if self._gvs is None:
            self.assemble_grid_panels(False)
        return self._gvs
    @property
    def gvh(self):
        if self._gvh is None:
            self.assemble_grid_horseshoes(False)
        return self._gvh
    @property
    def gpm(self):
        if self._gpm is None:
            self._gpm = self.gpd + self.gph
        return self._gpm
    @property
    def gvm(self):
        if self._gvm is None:
            self._gvm = self.gvd + self.gvh
        return self._gvm
    @property
    def sig(self):
        if self._sig is None:
            self._sig = -self.nrms
            # self._sig = solve_matrix_vector(self.ans, -self.nrms)
        return self._sig
    @property
    def mu(self):
        if self._mu is None:
            self.solve_dirichlet_system(time=False)
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
    def assemble_grid_panels(self, time: bool = True):
        if time:
            start = perf_counter()
        shp = (self.numgrd, self.numpnl)
        self._gpd = zeros(shp, dtype=float)
        self._gps = zeros(shp, dtype=float)
        self._gvd = zero_matrix_vector(shp, dtype=float)
        self._gvs = zero_matrix_vector(shp, dtype=float)
        for pnl in self.pnls.values():
            ind = pnl.ind
            self._apd[:, ind], self._aps[:, ind], self._avd[:, ind], self._avs[:, ind] = pnl.influence_coefficients(self.pnts)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Panel assembly time is {elapsed:.3f} seconds.')
    def assemble_grid_horseshoes(self, time: bool = True):
        if time:
            start = perf_counter()
        shp = (self.numgrd, self.numpnl)
        self._gph = zeros(shp, dtype=float)
        self._gvh = zero_matrix_vector(shp, dtype=float)
        for hsv in self.hsvs:
            ind = hsv.ind
            self._gph[:, ind], self._gvh[:, ind] = hsv.doublet_influence_coefficients(self.grdvec)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Horse shoe assembly time is {elapsed:.3f} seconds.')
    def solve_system(self, time: bool = True):
        if time:
            start = perf_counter()
        self.solve_dirichlet_system(time=False)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    def solve_dirichlet_system(self, time: bool = True):
        if time:
            start = perf_counter()
        self._mu = solve_matrix_vector(self.apm, self.bps)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    def solve_neumann_system(self, time: bool = True):
        if time:
            start = perf_counter()
        self._mu = solve_matrix_vector(self.anm, self.bnm)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    
def panelsystem_from_mesh(meshfilepath: str):

    with open(meshfilepath, 'rt') as meshfile:
        data = load(meshfile)
    
    name = data['name']
    bref = data['bref']
    cref = data['cref']
    sref = data['sref']
    xref = data['xref']
    yref = data['yref']
    zref = data['zref']
    rref = Vector(xref, yref, zref)

    grds = {}
    griddata = data['grids']
    for gidstr, gd in griddata.items():
        gid = int(gidstr)
        if 'te' not in gd:
            gd['te'] = False
        grds[gid] = Grid(gid, gd['x'], gd['y'], gd['z'], gd['te'])

    grps = {}
    if 'groups' in data:
        groupdata = data['groups']
        for grpidstr, grpdata in groupdata.items():
            grpid = int(grpidstr)
            grps[grpid] = grpdata
            if 'exclude' not in grps[grpid]:
                grps[grpid]['exclude'] = False
            if 'noload' not in grps[grpid]:
                grps[grpid]['noload'] = False

    pnls = {}
    paneldata = data['panels']
    for pidstr, pd in paneldata.items():
        pid = int(pidstr)
        if 'grpid' in pd:
            grp = grps[pd['grpid']]
            if not grp['exclude']:
                pnls[pid] = Panel(pid, pd['gids'])
                if grp['noload']:
                    pnls[pid].noload = True
        else:
            pnls[pid] = Panel(pid, pd['gids'])
    
    return PanelSystem(name, grds, pnls, bref, cref, sref, rref)

def panelsystem_from_json(jsonfilepath: str):
    with open(jsonfilepath, 'rt') as jsonfile:
        jsondata = load(jsonfile)

    from os.path import dirname, join, exists
    from .panelmesh import panelsurface_from_json

    path = dirname(jsonfilepath)

    for surfdata in jsondata['surfaces']:
        for sectdata in surfdata['sections']:
            if 'airfoil' in sectdata:
                airfoil = sectdata['airfoil']
                if airfoil[-4:] == '.dat':
                    airfoil = join(path, airfoil)
                    if not exists(airfoil):
                        print(f'Airfoil {airfoil} does not exist.')
                        del sectdata['airfoil']
                    else:
                        sectdata['airfoil'] = airfoil
    
    srfcs = []
    for surfdata in jsondata['surfaces']:
        srfcs.append(panelsurface_from_json(surfdata))
    
    name = jsondata['name']
    bref = jsondata['bref']
    cref = jsondata['cref']
    sref = jsondata['sref']
    xref = jsondata['xref']
    yref = jsondata['yref']
    zref = jsondata['zref']
    rref = Vector(xref, yref, zref)

    gid, pid = 0, 0
    for surface in srfcs:
        gid, pid = surface.mesh(gid, pid)

    grds, pnls = {}, {}
    for surface in srfcs:
        for gid, grd in surface.grds.items():
            grds[gid] = grd
        for pid, pnl in surface.pnls.items():
            pnls[pid] = pnl
        
    psys = PanelSystem(name, grds, pnls, bref, cref, sref, rref)
    psys.srfcs = srfcs

    return psys
