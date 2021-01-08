from json import load
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector, MatrixVector
from pygeom.matrix3d import solve_matrix_vector, elementwise_dot_product, elementwise_cross_product
from typing import Dict, List
from .panel import Panel
from .grid import Grid
from .horseshoe import HorseShoe
from numpy.matlib import zeros, matrix, divide, repeat
from time import perf_counter
from matplotlib.pyplot import figure

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
    _numhsv: int = None
    _pnts: MatrixVector = None
    _nrms: MatrixVector = None
    _pnla: matrix = None
    _rrel: MatrixVector = None
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
    _strps: List[object] = None
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
        self._hsvs = None
        for pnl in self.pnls.values():
            pnl.set_horseshoes(diro)
        self._numhsv = None
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
    def numhsv(self):
        if self._numhsv is None:
            self._numhsv = len(self.hsvs)
        return self._numhsv
    @property
    def pnts(self):
        if self._pnts is None:
            self._pnts = zero_matrix_vector((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._pnts[pnl.ind, 0] = pnl.pnto
        return self._pnts
    @property
    def rrel(self):
        if self._rrel is None:
            self._rrel = self.pnts-self.rref
        return self._rrel
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
            self.assemble_panels_full(False)
        return self._avd
    @property
    def aps(self):
        if self._aps is None:
            self.assemble_panels(False)
        return self._aps
    @property
    def avs(self):
        if self._avs is None:
            self.assemble_panels_full(False)
        return self._avs
    @property
    def aph(self):
        if self._aph is None:
            self.assemble_horseshoes(False)
        return self._aph
    @property
    def avh(self):
        if self._avh is None:
            self.assemble_horseshoes_full(False)
        return self._avh
    @property
    def apm(self):
        if self._apm is None:
            self._apm = self.apd.copy()
            for i, hsv in enumerate(self.hsvs):
                ind = hsv.ind
                self._apm[:, ind] = self._apm[:, ind] + self.aph[:, i]
        return self._apm
    @property
    def avm(self):
        if self._avm is None:
            self._avm = self.avd.copy()
            for i, hsv in enumerate(self.hsvs):
                ind = hsv.ind
                self._avm[:, ind] = self._avm[:, ind] + self.avh[:, i]
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
    def sig(self):
        if self._sig is None:
            self._sig = zero_matrix_vector((self.numpnl, 2), dtype=float)
            self._sig[:, 0] = -self.nrms
            self._sig[:, 1] = elementwise_cross_product(self.rrel, self.nrms)
        return self._sig
    @property
    def strps(self):
        if self._strps is None:
            if self.srfcs is not None:
                self._strps = []
                ind = 0
                for srfc in self.srfcs:
                    for strp in srfc.strps:
                        strp.ind = ind
                        self._strps.append(strp)
                        ind += 1
        return self._strps
    @property
    def mu(self):
        if self._mu is None:
            self.solve_dirichlet_system(time=False)
        return self._mu
    def assemble_panels(self, time: bool=True):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        self._apd = zeros(shp, dtype=float)
        self._aps = zeros(shp, dtype=float)
        for pnl in self.pnls.values():
            ind = pnl.ind
            self._apd[:, ind], self._aps[:, ind] = pnl.velocity_potentials(self.pnts)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Panel assembly time is {elapsed:.3f} seconds.')
    def assemble_panels_full(self, time: bool=True):
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
            print(f'Full panel assembly time is {elapsed:.3f} seconds.')
    def assemble_horseshoes(self, time: bool=True):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numhsv)
        self._aph = zeros(shp, dtype=float)
        for i, hsv in enumerate(self.hsvs):
            self._aph[:, i] = hsv.doublet_velocity_potentials(self.pnts)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Horse shoe assembly time is {elapsed:.3f} seconds.')
    def assemble_horseshoes_full(self, time: bool=True):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        self._aph = zeros(shp, dtype=float)
        self._avh = zero_matrix_vector(shp, dtype=float)
        for i, hsv in enumerate(self.hsvs):
            self._aph[:, i], self._avh[:, i] = hsv.doublet_influence_coefficients(self.pnts)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Full horse shoe assembly time is {elapsed:.3f} seconds.')
    def solve_system(self, time: bool=True):
        if time:
            start = perf_counter()
        self.solve_dirichlet_system(time=False)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    def solve_dirichlet_system(self, time: bool=True):
        if time:
            start = perf_counter()
        self._mu = solve_matrix_vector(self.apm, self.bps)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    def solve_neumann_system(self, time: bool=True):
        if time:
            start = perf_counter()
        self._mu = solve_matrix_vector(self.anm, self.bnm)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    def plot_twist_distribution(self, ax=None, axis: str='b', surfaces: list=[]):
        if self.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if len(surfaces) == 0:
                srfcs = [srfc for srfc in self.srfcs]
            else:
                srfcs = []
                for srfc in self.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            for srfc in srfcs:
                t = [prf.twist for prf in srfc.prfs]
                label = srfc.name
                if axis == 'b':
                    b = srfc.prfb
                    if max(b) > min(b):
                        ax.plot(b, t, label=label)
                elif axis == 'y':
                    y = srfc.prfy
                    if max(y) > min(y):
                        ax.plot(y, t, label=label)
                elif axis == 'z':
                    z = srfc.prfz
                    if max(z) > min(z):
                        ax.plot(z, t, label=label)
            ax.legend()
        return ax
    def plot_chord_distribution(self, ax=None, axis: str='b', surfaces: list=[]):
        if self.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if len(surfaces) == 0:
                srfcs = [srfc for srfc in self.srfcs]
            else:
                srfcs = []
                for srfc in self.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            for srfc in srfcs:
                c = [prf.chord for prf in srfc.prfs]
                label = srfc.name
                if axis == 'b':
                    b = srfc.prfb
                    if max(b) > min(b):
                        ax.plot(b, c, label=label)
                elif axis == 'y':
                    y = srfc.prfy
                    if max(y) > min(y):
                        ax.plot(y, c, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(z, c, label=label)
            ax.legend()
        return ax
    def plot_tilt_distribution(self, ax=None, axis: str='b', surfaces: list=[]):
        if self.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if len(surfaces) == 0:
                srfcs = [srfc for srfc in self.srfcs]
            else:
                srfcs = []
                for srfc in self.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            for srfc in srfcs:
                t = [prf.tilt for prf in srfc.prfs]
                label = srfc.name
                if axis == 'b':
                    b = srfc.prfb
                    if max(b) > min(b):
                        ax.plot(b, t, label=label)
                elif axis == 'y':
                    y = srfc.prfy
                    if max(y) > min(y):
                        ax.plot(y, t, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(z, t, label=label)
            ax.legend()
        return ax
    def plot_strip_width_distribution(self, ax=None, axis: str='b', surfaces: list=[]):
        if self.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if len(surfaces) == 0:
                srfcs = [srfc for srfc in self.srfcs]
            else:
                srfcs = []
                for srfc in self.srfcs:
                    if srfc.name in surfaces:
                        srfcs.append(srfc)
            for srfc in srfcs:
                w = [strp.width for strp in srfc.strps]
                label = srfc.name
                if axis == 'b':
                    b = srfc.strpb
                    if max(b) > min(b):
                        ax.plot(b, w, label=label)
                elif axis == 'y':
                    y = srfc.strpy
                    if max(y) > min(y):
                        ax.plot(y, w, label=label)
                elif axis == 'z':
                    z = srfc.strpz
                    if max(z) > min(z):
                        ax.plot(z, w, label=label)
            ax.legend()
        return ax
    def __repr__(self):
        return '<PanelSystem: {:s}>'.format(self.name)
    def __str__(self):
        from py2md.classes import MDTable
        outstr = '# Panel System '+self.name+'\n'
        table = MDTable()
        table.add_column('Name', 's', data=[self.name])
        table.add_column('Sref', 'g', data=[self.sref])
        table.add_column('cref', 'g', data=[self.cref])
        table.add_column('bref', 'g', data=[self.bref])
        table.add_column('xref', '.3f', data=[self.rref.x])
        table.add_column('yref', '.3f', data=[self.rref.y])
        table.add_column('zref', '.3f', data=[self.rref.z])
        outstr += table._repr_markdown_()
        table = MDTable()
        if self.pnls is not None:
            table.add_column('# Panels', 'd', data=[len(self.pnls)])
        if len(table.columns) > 0:
            outstr += table._repr_markdown_()
        return outstr
    def _repr_markdown_(self):
        return self.__str__()
    
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
            grpid = pd['grpid']
            grp = grps[grpid]
            if not grp['exclude']:
                pnls[pid] = Panel(pid, pd['gids'])
                if grp['noload']:
                    pnls[pid].noload = True
                pnls[pid].grp = grpid
        else:
            pnls[pid] = Panel(pid, pd['gids'])
    
    return PanelSystem(name, grds, pnls, bref, cref, sref, rref)

def panelsystem_from_json(jsonfilepath: str):
    with open(jsonfilepath, 'rt') as jsonfile:
        jsondata = load(jsonfile)

    from os.path import dirname, join, exists
    from .panelsurface import panelsurface_from_json

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

    gid, pid = 1, 1
    for surface in srfcs:
        gid = surface.mesh_grids(gid)
        pid = surface.mesh_panels(pid)

    grds, pnls = {}, {}
    for surface in srfcs:
        for grd in surface.grds:
            grds[grd.gid] = grd
        for pnl in surface.pnls:
            pnls[pnl.pid] = pnl
        
    psys = PanelSystem(name, grds, pnls, bref, cref, sref, rref)
    psys.srfcs = srfcs

    return psys
