from json import load
from typing import Dict, List
from time import perf_counter
from os.path import dirname, join, exists
from numpy.matlib import zeros, matrix
from matplotlib.pyplot import figure
from py2md.classes import MDTable
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector, MatrixVector
from pygeom.matrix3d import solve_matrix_vector, elementwise_dot_product, elementwise_cross_product
from .panel import Panel
from .grid import Grid
from .horseshoe import HorseShoe
from .panelsurface import panelsurface_from_json, PanelSurface
from .panelresult import panelresult_from_dict
from ..tools import betm_from_mach

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
    source: str = None
    _hsvs: List[HorseShoe] = None
    _numgrd: int = None
    _numpnl: int = None
    _numhsv: int = None
    _pnts: MatrixVector = None
    _pnla: matrix = None
    _nrms: MatrixVector = None
    _rrel: MatrixVector = None
    _apd: Dict[float, matrix] = None
    _aps: Dict[float, matrix] = None
    _aph: Dict[float, matrix] = None
    _apm: Dict[float, matrix] = None
    _bps: Dict[float, MatrixVector] = None
    _avd: Dict[float, MatrixVector] = None
    _avs: Dict[float, MatrixVector] = None
    _avh: Dict[float, MatrixVector] = None
    _avm: Dict[float, MatrixVector] = None
    _ans: Dict[float, matrix] = None
    _anm: Dict[float, matrix] = None
    _bnm: Dict[float, matrix] = None
    _unsig: Dict[float, MatrixVector] = None
    _unmu: Dict[float, MatrixVector] = None
    _unphi: Dict[float, MatrixVector] = None
    _unnvg: Dict[float, MatrixVector] = None
    _hsvpnts: MatrixVector = None
    _hsvnrms: MatrixVector = None
    _awd: matrix = None
    _aws: matrix = None
    _awh: matrix = None
    _adh: matrix = None
    _ash: matrix = None
    _alh: matrix = None
    _ar: float = None
    _area: float = None
    _strps: List[object] = None
    _phind: Dict[int, List[int]] = None
    def __init__(self, name: str, bref: float, cref: float, sref: float, rref: Vector):
        self.name = name
        self.bref = bref
        self.cref = cref
        self.sref = sref
        self.rref = rref
        self.results = {}
    def set_mesh(self, grds: Dict[int, Grid], pnls: Dict[int, Panel]):
        self.grds = grds
        self.pnls = pnls
        self.update()
    def set_geom(self, srfcs: List[PanelSurface]=None):
        self.srfcs = srfcs
        self.mesh()
        self.update()
    def update(self):
        for ind, grd in enumerate(self.grds.values()):
            grd.set_index(ind)
        for ind, pnl in enumerate(self.pnls.values()):
            pnl.set_index(ind)
    def reset(self):
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None
    def set_horseshoes(self, diro: Vector):
        for pnl in self.pnls.values():
            pnl.set_horseshoes(diro)
        self._hsvs = None
        self._numhsv = None
        self._aph = None
        self._avh = None
        self._apm = None
        self._avm = None
        self._anm = None
        self._hsvpnts = None
        self._hsvnrms = None
        self._awd = None
        self._aws = None
        self._awh = None
        self._adh = None
        self._ash = None
        self._alh = None
        self._phind = None
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
    def phind(self):
        if self._phind is None:
            self._phind = {}
            for i, hsv in enumerate(self.hsvs):
                pind = hsv.ind
                if pind in self._phind:
                    self._phind[pind].append(i)
                else:
                    self._phind[pind] = [i]
        return self._phind
    def bps(self, mach: float=0.0):
        if self._bps is None:
            self._bps = {}
        if mach not in self._bps:
            self._bps[mach] = -1.0*self.aps(mach)*self.unsig(mach)
        return self._bps[mach]
    def bnm(self, mach: float=0.0):
        if self._bnm is None:
            self._bnm = {}
        if mach not in self._bnm:
            self._bnm[mach] = -self.nrms-self.ans(mach)*self.unsig(mach)
        return self._bnm[mach]
    def apd(self, mach: float=0.0):
        if self._apd is None:
            self._apd = {}
        if mach not in self._apd:
            self.assemble_panels(False, mach=mach)
        return self._apd[mach]
    def avd(self, mach: float=0.0):
        if self._avd is None:
            self._avd = {}
        if mach not in self._avd:
            self.assemble_panels_full(False, mach=mach)
        return self._avd[mach]
    def aps(self, mach: float=0.0):
        if self._aps is None:
            self._aps = {}
        if mach not in self._aps:
            self.assemble_panels(False, mach=mach)
        return self._aps[mach]
    def avs(self, mach: float=0.0):
        if self._avs is None:
            self._avs = {}
        if mach not in self._avs:
            self.assemble_panels_full(False, mach=mach)
        return self._avs[mach]
    def aph(self, mach: float=0.0):
        if self._aph is None:
            self._aph = {}
        if mach not in self._aph:
            self.assemble_horseshoes(False, mach=mach)
        return self._aph[mach]
    def avh(self, mach: float=0.0):
        if self._avh is None:
            self._avh = {}
        if mach not in self._avh:
            self.assemble_horseshoes_full(False, mach=mach)
        return self._avh[mach]
    def apm(self, mach: float=0.0):
        if self._apm is None:
            self._apm = {}
        if mach not in self._apm:
            apm = self.apd(mach).copy()
            aph = self.aph(mach)
            for i, hsv in enumerate(self.hsvs):
                ind = hsv.ind
                apm[:, ind] = apm[:, ind] + aph[:, i]
            self._apm[mach] = apm
        return self._apm[mach]
    def avm(self, mach: float=0.0):
        if self._avm is None:
            self._avm = {}
        if self._avm is None:
            avm = self.avd(mach).copy()
            avh = self.avh(mach)
            for i, hsv in enumerate(self.hsvs):
                ind = hsv.ind
                avm[:, ind] = avm[:, ind] + avh[:, i]
            self._avm[mach] = avm
        return self._avm[mach]
    def ans(self, mach: float=0.0):
        if self._ans is None:
            self._ans = {}
        if mach not in self._ans:
            nrms = self.nrms.repeat(self.numpnl, axis=1)
            self._ans[mach] = elementwise_dot_product(nrms, self.avs(mach))
        return self._ans[mach]
    def anm(self, mach: float=0.0):
        if self._anm is None:
            self._anm = {}
        if mach not in self._anm:
            nrms = self.nrms.repeat(self.numpnl, axis=1)
            self._anm[mach] = elementwise_dot_product(nrms, self.avm(mach))
        return self._anm[mach]
    @property
    def hsvpnts(self):
        if self._hsvpnts is None:
            self._hsvpnts = zero_matrix_vector((self.numhsv, 1), dtype=float)
            for i, hsv in enumerate(self.hsvs):
                self._hsvpnts[i, 0] = hsv.pnto
        return self._hsvpnts
    @property
    def hsvnrms(self):
        if self._hsvnrms is None:
            self._hsvnrms = zero_matrix_vector((self.numhsv, 1), dtype=float)
            for i, hsv in enumerate(self.hsvs):
                self._hsvnrms[i, 0] = hsv.nrm
        return self._hsvnrms
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
    def assemble_panels_wash(self, time: bool=True):
        if time:
            start = perf_counter()
        shp = (self.numhsv, self.numpnl)
        self._awd = zeros(shp, dtype=float)
        self._aws = zeros(shp, dtype=float)
        for pnl in self.pnls.values():
            ind = pnl.ind
            _, _, avd, avs = pnl.influence_coefficients(self.hsvpnts)
            self._awd[:, ind] = elementwise_dot_product(avd, self.hsvnrms)
            self._aws[:, ind] = elementwise_dot_product(avs, self.hsvnrms)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Wash matrix assembly time is {elapsed:.3f} seconds.')
    def assemble_horseshoes_wash(self, time: bool=True):
        if time:
            start = perf_counter()
        shp = (self.numhsv, self.numhsv)
        self._awh = zeros(shp, dtype=float)
        for i, hsv in enumerate(self.hsvs):
            avh = hsv.trefftz_plane_velocities(self.hsvpnts)
            self._awh[:, i] = elementwise_dot_product(avh, self.hsvnrms)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Wash horse shoe assembly time is {elapsed:.3f} seconds.')
    # def assemble_horseshoes_wash_v2(self, time: bool=True):
    #     if time:
    #         start = perf_counter()
    #     shp = (self.numhsv, self.numhsv)
    #     self._awh = zeros(shp, dtype=float)
    #     for i, hsvi in enumerate(self.hsvs):
    #         for j, hsvj in enumerate(self.hsvs):
    #             avh = hsvj.trefftz_velocity(hsvi.pnto)
    #             # self._awh[i, j] = avh*hsvi.nrm
    #             if hsvi.nrm.z < 0:
    #                 nrm = Vector(0.0, 0.0, -1.0)
    #             else:
    #                 nrm = Vector(0.0, 0.0, 1.0)
    #             self._awh[i, j] = avh*nrm
    #     if time:
    #         finish = perf_counter()
    #         elapsed = finish - start
    #         print(f'Wash horse shoe assembly time is {elapsed:.3f} seconds.')
    @property
    def awh(self):
        if self._awh is None:
            self.assemble_horseshoes_wash(time=False)
        return self._awh
    @property
    def awd(self):
        if self._awd is None:
            self.assemble_panels_wash(time=False)
        return self._awd
    @property
    def aws(self):
        if self._aws is None:
            self.assemble_panels_wash(time=False)
        return self._aws
    @property
    def adh(self):
        if self._adh is None:
            self._adh = zeros(self.awh.shape, dtype=float)
            for i, hsv in enumerate(self.hsvs):
                self._adh[:, i] = -self._awh[:, i]*hsv.width
        return self._adh
    @property
    def ash(self):
        if self._ash is None:
            self._ash = zeros((self.numhsv, 1), dtype=float)
            for i, hsv in enumerate(self.hsvs):
                self._ash[i, 0] = -hsv.vecab.z
        return self._ash
    @property
    def alh(self):
        if self._alh is None:
            self._alh = zeros((self.numhsv, 1), dtype=float)
            for i, hsv in enumerate(self.hsvs):
                self._alh[i, 0] = hsv.vecab.y
        return self._alh
    def unsig(self, mach: float=0.0):
        if self._unsig is None:
            self._unsig = {}
        if mach not in self._unsig:
            unsig = zero_matrix_vector((self.numpnl, 2), dtype=float)
            unsig[:, 0] = -self.nrms
            unsig[:, 1] = elementwise_cross_product(self.rrel, self.nrms)
            self._unsig[mach] = unsig
        return self._unsig[mach]
    def unmu(self, mach: float=0.0):
        if self._unmu is None:
            self._unmu = {}
        if mach not in self._unmu:
            self.solve_dirichlet_system(time=False, mach=mach)
        return self._unmu[mach]
    def unphi(self, mach: float=0.0):
        if self._unphi is None:
            self._unphi = {}
        if mach not in self._unphi:
            self.solve_dirichlet_system(time=False, mach=mach)
        return self._unphi[mach]
    def assemble_panels(self, time: bool=True, mach=0.0):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        apd = zeros(shp, dtype=float)
        aps = zeros(shp, dtype=float)
        betm = betm_from_mach(mach)
        for pnl in self.pnls.values():
            ind = pnl.ind
            apd[:, ind], aps[:, ind] = pnl.velocity_potentials(self.pnts, betx=betm)
        if self._apd is None:
            self._apd = {}
        self._apd[mach] = apd
        if self._aps is None:
            self._aps = {}
        self._aps[mach] = aps
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Panel assembly time is {elapsed:.3f} seconds.')
    def assemble_panels_full(self, time: bool=True, mach: float=0.0):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        apd = zeros(shp, dtype=float)
        aps = zeros(shp, dtype=float)
        avd = zero_matrix_vector(shp, dtype=float)
        avs = zero_matrix_vector(shp, dtype=float)
        betm = betm_from_mach(mach)
        for pnl in self.pnls.values():
            ind = pnl.ind
            apd[:, ind], aps[:, ind], avd[:, ind], avs[:, ind] = pnl.influence_coefficients(self.pnts, betx=betm)
        if self._apd is None:
            self._apd = {}
        self._apd[mach] = apd
        if self._aps is None:
            self._aps = {}
        self._aps[mach] = aps
        if self._avd is None:
            self._avd = {}
        self._avd[mach] = avd
        if self._avs is None:
            self._avs = {}
        self._avs[mach] = avs
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Full panel assembly time is {elapsed:.3f} seconds.')
    def assemble_horseshoes(self, time: bool=True, mach: float=0.0):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numhsv)
        aph = zeros(shp, dtype=float)
        betm = betm_from_mach(mach)
        for i, hsv in enumerate(self.hsvs):
            aph[:, i] = hsv.doublet_velocity_potentials(self.pnts, betx=betm)
        if self._aph is None:
            self._aph = {}
        self._aph[mach] = aph
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Horse shoe assembly time is {elapsed:.3f} seconds.')
    def assemble_horseshoes_full(self, time: bool=True, mach: float=0.0):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        aph = zeros(shp, dtype=float)
        avh = zero_matrix_vector(shp, dtype=float)
        betm = betm_from_mach(mach)
        for i, hsv in enumerate(self.hsvs):
            aph[:, i], avh[:, i] = hsv.doublet_influence_coefficients(self.pnts, betx=betm)
        if self._aph is None:
            self._aph = {}
        self._aph[mach] = aph
        if self._avh is None:
            self._avh = {}
        self._avh[mach] = avh
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Full horse shoe assembly time is {elapsed:.3f} seconds.')
    def solve_system(self, time: bool=True, mach: float=0.0):
        if time:
            start = perf_counter()
        self.solve_dirichlet_system(time=False, mach=mach)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    def solve_dirichlet_system(self, time: bool=True, mach: float=0.0):
        if time:
            start = perf_counter()
        if self._unmu is None:
            self._unmu = {}
        self._unmu[mach] = solve_matrix_vector(self.apm(mach), self.bps(mach))
        if self._unphi is None:
            self._unphi = {}
        self._unphi[mach] = self.apm(mach)*self.unmu(mach) + self.bps(mach)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    def solve_neumann_system(self, time: bool=True, mach: float=0.0):
        if time:
            start = perf_counter()
        if self._unmu is None:
            self._unmu = {}
        self._unmu[mach] = solve_matrix_vector(self.anm(mach), self.bnm(mach))
        if self._unnvg is None:
            self._unnvg = {}
        self._unnvg[mach] = self.anm(mach)*self.unmu(mach) + self.bnm(mach)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')
    def plot_twist_distribution(self, ax=None, axis: str='b', surfaces: list=None):
        if self.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
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
    def plot_chord_distribution(self, ax=None, axis: str='b', surfaces: list=None):
        if self.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
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
    def plot_tilt_distribution(self, ax=None, axis: str='b', surfaces: list=None):
        if self.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
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
    def plot_strip_width_distribution(self, ax=None, axis: str='b', surfaces: list=None):
        if self.srfcs is not None:
            if ax is None:
                fig = figure(figsize=(12, 8))
                ax = fig.gca()
                ax.grid(True)
            if surfaces is None:
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
    def mesh(self):
        if self.srfcs is not None:
            gid, pid = 1, 1
            for surface in self.srfcs:
                gid = surface.mesh_grids(gid)
                pid = surface.mesh_panels(pid)
            self.grds, self.pnls = {}, {}
            for surface in self.srfcs:
                for grd in surface.grds:
                    self.grds[grd.gid] = grd
                for pnl in surface.pnls:
                    self.pnls[pnl.pid] = pnl
    def __repr__(self):
        return '<PanelSystem: {:s}>'.format(self.name)
    def __str__(self):
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
        if self.grds is not None:
            table.add_column('# Grids', 'd', data=[self.numgrd])
        else:
            table.add_column('# Grids', 'd', data=[0])
        if self.pnls is not None:
            table.add_column('# Panels', 'd', data=[self.numpnl])
        else:
            table.add_column('# Panels', 'd', data=[0])
        if self.hsvs is not None:
            table.add_column('# Horseshoe Vortices', 'd', data=[self.numhsv])
        else:
            table.add_column('# Horseshoe Vortices', 'd', data=[0])
        if len(table.columns) > 0:
            outstr += table._repr_markdown_()
        return outstr
    def _repr_markdown_(self):
        return self.__str__()

def panelsystem_from_json(jsonfilepath: str):

    with open(jsonfilepath, 'rt') as jsonfile:
        sysdct = load(jsonfile)

    sysdct['source'] = jsonfilepath

    filetype = None
    if 'type' in sysdct:
        filetype = sysdct['type']
    elif 'panels' in sysdct and 'grids' in sysdct:
        filetype = 'mesh'
    elif 'surfaces' in sysdct:
        filetype = 'geom'

    if filetype == 'geom':
        psys = panelsystem_from_geom(sysdct)
    elif filetype == 'mesh':
        psys = panelsystem_from_mesh(sysdct)
    else:
        return ValueError('Incorrect file type.')

    psys.source = jsonfilepath

    return psys

def panelsystem_from_mesh(sysdct: Dict[str, any]):

    name = sysdct['name']
    bref = sysdct['bref']
    cref = sysdct['cref']
    sref = sysdct['sref']
    xref = sysdct['xref']
    yref = sysdct['yref']
    zref = sysdct['zref']
    rref = Vector(xref, yref, zref)

    grds = {}
    griddata = sysdct['grids']
    for gidstr, gd in griddata.items():
        gid = int(gidstr)
        if 'te' not in gd:
            gd['te'] = False
        grds[gid] = Grid(gid, gd['x'], gd['y'], gd['z'], gd['te'])

    grps = {}
    if 'groups' in sysdct:
        groupdata = sysdct['groups']
        for grpidstr, grpdata in groupdata.items():
            grpid = int(grpidstr)
            grps[grpid] = grpdata
            if 'exclude' not in grps[grpid]:
                grps[grpid]['exclude'] = False
            if 'noload' not in grps[grpid]:
                grps[grpid]['noload'] = False

    pnls = {}
    paneldata = sysdct['panels']
    for pidstr, pd in paneldata.items():
        pid = int(pidstr)
        if 'grpid' in pd:
            grpid = pd['grpid']
            grp = grps[grpid]
            if not grp['exclude']:
                pnlgrds = [grds[gidi] for gidi in pd['gids']]
                pnls[pid] = Panel(pid, pnlgrds)
                if grp['noload']:
                    pnls[pid].noload = True
                pnls[pid].grp = grpid
        else:
            pnlgrds = [grds[gidi] for gidi in pd['gids']]
            pnls[pid] = Panel(pid, pnlgrds)

    psys = PanelSystem(name, bref, cref, sref, rref)
    psys.set_mesh(grds, pnls)

    if 'cases' in sysdct and sysdct:
        panelresults_from_dict(psys, sysdct['cases'])

    if 'source' in sysdct:
        psys.source = sysdct['source']

    return psys

def panelsystem_from_geom(sysdct: Dict[str, any]):

    if 'source' in sysdct:
        path = dirname(sysdct['source'])

        for surfdata in sysdct['surfaces']:
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
    for surfdata in sysdct['surfaces']:
        srfcs.append(panelsurface_from_json(surfdata))

    name = sysdct['name']
    bref = sysdct['bref']
    cref = sysdct['cref']
    sref = sysdct['sref']
    xref = sysdct['xref']
    yref = sysdct['yref']
    zref = sysdct['zref']
    rref = Vector(xref, yref, zref)

    psys = PanelSystem(name, bref, cref, sref, rref)
    psys.set_geom(srfcs)

    if 'cases' in sysdct and sysdct:
        panelresults_from_dict(psys, sysdct['cases'])

    if 'source' in sysdct:
        psys.source = sysdct['source']

    return psys

def panelresults_from_dict(psys: PanelSystem, cases: dict):

    for i in range(len(cases)):
        resdata = cases[i]
        # if 'trim' in resdata:
        #     latticetrim_from_json(lsys, resdata)
        # else:
        panelresult_from_dict(psys, resdata)
