from json import load
from typing import Dict, List, Tuple
from time import perf_counter
from os.path import dirname, join, exists
from numpy.matlib import zeros, matrix
from matplotlib.pyplot import figure
from py2md.classes import MDTable
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector, MatrixVector
from pygeom.matrix3d import solve_matrix_vector, elementwise_dot_product, elementwise_cross_product
from .grid import Grid, GridNormal
from .trianglepanel import TrianglePanel
from .trailingdoubletpanel import TrailingDoubletPanel
from .panelsurface import panelsurface_from_json, PanelSurface
from .panelresult import panelresult_from_dict
from .paneltrim import paneltrim_from_dict
from ..tools import betm_from_mach
from ..tools.mass import masses_from_json

class PanelSystem(object):
    name: str = None
    grds: Dict[int, Grid] = None
    pnls: Dict[int, TrianglePanel] = None
    bref: float = None
    cref: float = None
    sref: float = None
    rref: float = None
    ctrls: Dict[str, Tuple[int]] = None
    srfcs: List[object] = None
    results: List[object] = None
    masses = None
    source: str = None
    _tdps: List[TrailingDoubletPanel] = None
    _numgrd: int = None
    _numpnl: int = None
    _numnrm: int = None
    _numtdp: int = None
    _numctrl: int = None
    _pnts: MatrixVector = None
    _pnlarea: matrix = None
    _pnlweta: matrix = None
    _pnlpnts: MatrixVector = None
    _pnlrrel: MatrixVector = None
    _pnlnrms: MatrixVector = None
    _grdnrms: List[GridNormal] = None
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
    _tdppnts: MatrixVector = None
    _tdpnrms: MatrixVector = None
    _awd: matrix = None
    _aws: matrix = None
    _awh: matrix = None
    _adh: matrix = None
    _ash: matrix = None
    _alh: matrix = None
    _ar: float = None
    _area: float = None
    _wetarea: float = None
    _strps: List[object] = None
    _phind: Dict[int, List[int]] = None
    def __init__(self, name: str, bref: float, cref: float, sref: float, rref: Vector):
        self.name = name
        self.bref = bref
        self.cref = cref
        self.sref = sref
        self.rref = rref
        self.ctrls = {}
        self.results = {}
    # def set_mesh(self, grds: Dict[int, Grid], pnls: Dict[int, TrianglePanel]):
    #     self.grds = grds
    #     self.pnls = pnls
    #     self.update()
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
    def mesh_trailing_doublet_panels(self, diro: Vector):
        pid = 0
        for pnl in self.pnls.values():
            pid = pnl.mesh_trailing_doublet_panels(pid, diro)
        self._tdps = None
        self._numtdp = None
        self._aph = None
        self._avh = None
        self._apm = None
        self._avm = None
        self._anm = None
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
    def numnrm(self):
        if self._numnrm is None:
            self._numnrm = len(self.grdnrms)
        return self._numnrm
    @property
    def numtdp(self):
        if self._numtdp is None:
            self._numtdp = len(self.tdps)
        return self._numtdp
    @property
    def numctrl(self):
        if self._numctrl is None:
            self._numctrl = len(self.ctrls)
        return self._numctrl
    @property
    def pnts(self):
        if self._pnts is None:
            self._pnts = zero_matrix_vector((self.numgrd, 1), dtype=float)
            for grd in self.grds.values():
                self._pnts[grd.ind, 0] = grd
        return self._pnts
    @property
    def pnlpnts(self):
        if self._pnlpnts is None:
            self._pnlpnts = zero_matrix_vector((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._pnlpnts[pnl.ind, 0] = pnl.pnto
        return self._pnlpnts
    @property
    def pnlrrel(self):
        if self._pnlrrel is None:
            self._pnlrrel = zero_matrix_vector((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._pnlrrel[pnl.ind, 0] = pnl.pnto-self.rref
        return self._pnlrrel
    @property
    def pnlnrms(self):
        if self._pnlnrms is None:
            self._pnlnrms = zero_matrix_vector((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._pnlnrms[pnl.ind, 0] = pnl.dirz
        return self._pnlnrms
    @property
    def rrel(self):
        if self._rrel is None:
            shp = (len(self.grdnrms), 1)
            self._rrel = zero_matrix_vector(shp, dtype=float)
            for grdnrm in self.grdnrms:
                self._rrel[grdnrm.ind, 0] = grdnrm.grd-self.rref
        return self._rrel
    @property
    def grdnrms(self):
        if self._grdnrms is None:
            self._grdnrms = []
            for pnl in self.pnls.values():
                if pnl.nrma not in self._grdnrms:
                    self._grdnrms.append(pnl.nrma)
                if pnl.nrmb not in self._grdnrms:
                    self._grdnrms.append(pnl.nrmb)
                if pnl.nrmc not in self._grdnrms:
                    self._grdnrms.append(pnl.nrmc)
            for ind, grdnrm in enumerate(self._grdnrms):
                grdnrm.set_index(ind)
        return self._grdnrms
    @property
    def nrms(self):
        if self._nrms is None:
            shp = (len(self.grdnrms), 1)
            self._nrms = zero_matrix_vector(shp, dtype=float)
            for grdnrm in self.grdnrms:
                self._nrms[grdnrm.ind, 0] = grdnrm
        return self._nrms
    @property
    def pnlarea(self):
        if self._pnlarea is None:
            self._pnlarea = zeros((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._pnlarea[pnl.ind, 0] = pnl.area
        return self._pnlarea
    @property
    def pnlweta(self):
        if self._pnlweta is None:
            self._pnlweta = zeros((self.numpnl, 1), dtype=float)
            for pnl in self.pnls.values():
                self._pnlweta[pnl.ind, 0] = pnl.wetarea
        return self._pnlweta
    @property
    def tdps(self):
        if self._tdps is None:
            self._tdps = []
            for pnl in self.pnls.values():
                self._tdps = self._tdps + pnl.tdps
        return self._tdps
    @property
    def phind(self):
        if self._phind is None:
            self._phind = {}
            for i, tdp in enumerate(self.tdps):
                pind = tdp.ind
                if pind in self._phind:
                    self._phind[pind].append(i)
                else:
                    self._phind[pind] = [i]
        return self._phind
    def bps(self, mach: float=0.0):
        if self._bps is None:
            self._bps = {}
        if mach not in self._bps:
            self._bps[mach] = -self.aps(mach)*self.unsig(mach)
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
    # def avd(self, mach: float=0.0):
    #     if self._avd is None:
    #         self._avd = {}
    #     if mach not in self._avd:
    #         self.assemble_panels_full(False, mach=mach)
    #     return self._avd[mach]
    def aps(self, mach: float=0.0):
        if self._aps is None:
            self._aps = {}
        if mach not in self._aps:
            self.assemble_panels(False, mach=mach)
        return self._aps[mach]
    # def avs(self, mach: float=0.0):
    #     if self._avs is None:
    #         self._avs = {}
    #     if mach not in self._avs:
    #         self.assemble_panels_full(False, mach=mach)
    #     return self._avs[mach]
    def aph(self, mach: float=0.0):
        if self._aph is None:
            self._aph = {}
        if mach not in self._aph:
            self.assemble_trailing_panels(False, mach=mach)
        return self._aph[mach]
    # def avh(self, mach: float=0.0):
    #     if self._avh is None:
    #         self._avh = {}
    #     if mach not in self._avh:
    #         self.assemble_horseshoes_full(False, mach=mach)
    #     return self._avh[mach]
    def apm(self, mach: float=0.0):
        if self._apm is None:
            self._apm = {}
        if mach not in self._apm:
            apm = self.apd(mach).copy()
            aph = self.aph(mach)
            for i, tdp in enumerate(self.tdps):
                ia = 2*i
                ib = ia+1
                inda, indb = tdp.indd
                apm[:, inda] += aph[:, ia]
                apm[:, indb] += aph[:, ib]
            self._apm[mach] = apm
        return self._apm[mach]
    # def avm(self, mach: float=0.0):
    #     if self._avm is None:
    #         self._avm = {}
    #     if self._avm is None:
    #         avm = self.avd(mach).copy()
    #         avh = self.avh(mach)
    #         for i, tdp in enumerate(self.tdps):
    #             ind = tdp.ind
    #             avm[:, ind] = avm[:, ind] + avh[:, i]
    #         self._avm[mach] = avm
    #     return self._avm[mach]
    # def ans(self, mach: float=0.0):
    #     if self._ans is None:
    #         self._ans = {}
    #     if mach not in self._ans:
    #         nrms = self.nrms.repeat(self.numpnl, axis=1)
    #         self._ans[mach] = elementwise_dot_product(nrms, self.avs(mach))
    #     return self._ans[mach]
    # def anm(self, mach: float=0.0):
    #     if self._anm is None:
    #         self._anm = {}
    #     if mach not in self._anm:
    #         nrms = self.nrms.repeat(self.numpnl, axis=1)
    #         self._anm[mach] = elementwise_dot_product(nrms, self.avm(mach))
    #     return self._anm[mach]
    @property
    def tdppnts(self):
        if self._tdppnts is None:
            self._tdppnts = zero_matrix_vector((self.numtdp, 1), dtype=float)
            for i, tdp in enumerate(self.tdps):
                self._tdppnts[i, 0] = tdp.pnto
        return self._tdppnts
    @property
    def tdpnrms(self):
        if self._tdpnrms is None:
            self._tdpnrms = zero_matrix_vector((self.numtdp, 1), dtype=float)
            for i, tdp in enumerate(self.tdps):
                self._tdpnrms[i, 0] = tdp.nrm
        return self._tdpnrms
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
            for i, tdp in enumerate(self.tdps):
                self._adh[:, i] = -self._awh[:, i]*tdp.width
        return self._adh
    @property
    def ash(self):
        if self._ash is None:
            self._ash = zeros((self.numtdp, 1), dtype=float)
            for i, tdp in enumerate(self.tdps):
                self._ash[i, 0] = -tdp.vecab.z
        return self._ash
    @property
    def alh(self):
        if self._alh is None:
            self._alh = zeros((self.numtdp, 1), dtype=float)
            for i, tdp in enumerate(self.tdps):
                self._alh[i, 0] = tdp.vecab.y
        return self._alh
    def unsig(self, mach: float=0.0):
        if self._unsig is None:
            self._unsig = {}
        if mach not in self._unsig:
            unsig = zero_matrix_vector((self.numnrm, 2+4*self.numctrl), dtype=float)
            unsig[:, 0] = -self.nrms
            unsig[:, 1] = elementwise_cross_product(self.rrel, self.nrms)
            if self.srfcs is not None:
                for srfc in self.srfcs:
                    for sht in srfc.shts:
                        for control in sht.ctrls:
                            ctrl = sht.ctrls[control]
                            ctup = self.ctrls[control]
                            for pnl in ctrl.pnls:
                                inda, indb, indc = pnl.inds
                                rrela = self.rrel[inda, 0]
                                rrelb = self.rrel[indb, 0]
                                rrelc = self.rrel[indc, 0]
                                dndlpa, dndlpb, dndlpc = pnl.dndl(ctrl.posgain, ctrl.uhvec)
                                unsig[inda, ctup[0]] = -dndlpa
                                unsig[inda, ctup[1]] = -rrela**dndlpa
                                unsig[indb, ctup[0]] = -dndlpb
                                unsig[indb, ctup[1]] = -rrelb**dndlpb
                                unsig[indc, ctup[0]] = -dndlpc
                                unsig[indc, ctup[1]] = -rrelc**dndlpc
                                dndlna, dndlnb, dndlnc = pnl.dndl(ctrl.neggain, ctrl.uhvec)
                                unsig[inda, ctup[2]] = -dndlna
                                unsig[inda, ctup[3]] = -rrela**dndlna
                                unsig[indb, ctup[2]] = -dndlnb
                                unsig[indb, ctup[3]] = -rrela**dndlnb
                                unsig[indc, ctup[2]] = -dndlnc
                                unsig[indc, ctup[3]] = -rrela**dndlnc
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
        shp = (self.numgrd, self.numgrd)
        apd = zeros(shp, dtype=float)
        shp = (self.numgrd, self.numnrm)
        aps = zeros(shp, dtype=float)
        betm = betm_from_mach(mach)
        for pnl in self.pnls.values():
            indda, inddb, inddc = pnl.indd
            indsa, indsb, indsc = pnl.inds
            phidabc, phisabc = pnl.linear_phi(self.pnts, betx=betm)
            apd[:, indda] += phidabc[0]
            apd[:, inddb] += phidabc[1]
            apd[:, inddc] += phidabc[2]
            aps[:, indsa] += phisabc[0]
            aps[:, indsb] += phisabc[1]
            aps[:, indsc] += phisabc[2]
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
    def assemble_trailing_panels(self, time: bool=True, mach: float=0.0):
        if time:
            start = perf_counter()
        shp = (self.numgrd, 2*self.numtdp)
        aph = zeros(shp, dtype=float)
        betm = betm_from_mach(mach)
        for i, tdp in enumerate(self.tdps):
            ia = 2*i
            ib = ia+1
            phida, phidb = tdp.linear_doublet_phi(self.pnts, betx=betm)
            aph[:, ia] = phida
            aph[:, ib] = phidb
        if self._aph is None:
            self._aph = {}
        self._aph[mach] = aph
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Trailing doublet assembly time is {elapsed:.3f} seconds.')
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
            ind = 2
            for srfc in self.srfcs:
                for sht in srfc.shts:
                    for control in sht.ctrls:
                        if control not in self.ctrls:
                            self.ctrls[control] = (ind, ind+1, ind+2, ind+3)
                            ind += 4
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
        if self.tdps is not None:
            table.add_column('# Wake Panels', 'd', data=[self.numtdp])
        else:
            table.add_column('# Wake Panels', 'd', data=[0])
        if self.ctrls is not None:
            table.add_column('# Controls', 'd', data=[self.numctrl])
        else:
            table.add_column('# Controls', 'd', data=[0])
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
    # elif filetype == 'mesh':
    #     psys = panelsystem_from_mesh(sysdct)
    else:
        raise ValueError('Incorrect file type.')

    psys.source = jsonfilepath

    return psys

# def panelsystem_from_mesh(sysdct: Dict[str, any]):

#     name = sysdct['name']
#     bref = sysdct['bref']
#     cref = sysdct['cref']
#     sref = sysdct['sref']
#     xref = sysdct['xref']
#     yref = sysdct['yref']
#     zref = sysdct['zref']
#     rref = Vector(xref, yref, zref)

#     grds = {}
#     griddata = sysdct['grids']
#     for gidstr, gd in griddata.items():
#         gid = int(gidstr)
#         if 'te' not in gd:
#             gd['te'] = False
#         grds[gid] = Grid(gid, gd['x'], gd['y'], gd['z'], gd['te'])

#     grps = {}
#     if 'groups' in sysdct:
#         groupdata = sysdct['groups']
#         for grpidstr, grpdata in groupdata.items():
#             grpid = int(grpidstr)
#             grps[grpid] = grpdata
#             if 'exclude' not in grps[grpid]:
#                 grps[grpid]['exclude'] = False
#             if 'noload' not in grps[grpid]:
#                 grps[grpid]['noload'] = False

#     pnls = {}
#     paneldata = sysdct['panels']
#     for pidstr, pd in paneldata.items():
#         pid = int(pidstr)
#         if 'grpid' in pd:
#             grpid = pd['grpid']
#             grp = grps[grpid]
#             if not grp['exclude']:
#                 pnlgrds = [grds[gidi] for gidi in pd['gids']]
#                 pnls[pid] = Panel(pid, pnlgrds)
#                 if grp['noload']:
#                     pnls[pid].noload = True
#                 pnls[pid].grp = grpid
#         else:
#             pnlgrds = [grds[gidi] for gidi in pd['gids']]
#             pnls[pid] = Panel(pid, pnlgrds)

#     psys = PanelSystem(name, bref, cref, sref, rref)
#     psys.set_mesh(grds, pnls)

#     masses = {}
#     if 'masses' in sysdct:
#         if isinstance(sysdct['masses'], list):
#             masses = masses_from_json(sysdct['masses'])
#         # elif isinstance(sysdct['masses'], str):
#         #     if sysdct['masses'][-5:] == '.json':
#         #         massfilename = sysdct['masses']
#         #         massfilepath = join(path, massfilename)
#         #     masses = masses_from_json(massfilepath)
#     psys.masses = masses

#     if 'cases' in sysdct and sysdct:
#         panelresults_from_dict(psys, sysdct['cases'])

#     if 'source' in sysdct:
#         psys.source = sysdct['source']

#     return psys

def panelsystem_from_geom(sysdct: Dict[str, any]):

    if 'source' in sysdct:
        path = dirname(sysdct['source'])

        for surfdata in sysdct['surfaces']:
            for sectdata in surfdata['sections']:
                if 'airfoil' in sectdata:
                    airfoil = sectdata['airfoil']
                    if airfoil is not None:
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

    masses = {}
    if 'masses' in sysdct:
        if isinstance(sysdct['masses'], list):
            masses = masses_from_json(sysdct['masses'])
        elif isinstance(sysdct['masses'], str):
            if sysdct['masses'][-5:] == '.json':
                massfilename = sysdct['masses']
                massfilepath = join(path, massfilename)
            masses = masses_from_json(massfilepath)
    psys.masses = masses

    if 'cases' in sysdct and sysdct:
        panelresults_from_dict(psys, sysdct['cases'])

    if 'source' in sysdct:
        psys.source = sysdct['source']

    return psys

def panelresults_from_dict(psys: PanelSystem, cases: dict):

    for i in range(len(cases)):
        resdata = cases[i]
        if 'trim' in resdata:
            paneltrim_from_dict(psys, resdata)
        else:
            panelresult_from_dict(psys, resdata)
