from json import dump, load
from os.path import dirname, exists, join
from time import perf_counter
from typing import TYPE_CHECKING, Any

from matplotlib.pyplot import figure
from numpy import add, zeros
from numpy.linalg import norm
from py2md.classes import MDTable
from pygeom.geom3d import Vector

from ..tools import betm_from_mach
from ..tools.mass import Mass, masses_from_data, masses_from_json
from .edge import edges_array, edges_from_system
from .grid import Grid
from .panel import Panel
from .panelresult import PanelResult
from .panelsurface import PanelSurface
from .paneltrim import PanelTrim

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..tools.mass import MassCollection
    from .edge import Edge
    from .horseshoedoublet import HorseshoeDoublet
    from .panelstrip import PanelStrip


class PanelSystem():
    name: str = None
    grds: dict[int, Grid] = None
    pnls: dict[int, Panel] = None
    bref: float = None
    cref: float = None
    sref: float = None
    rref: Vector = None
    ctrls: dict[str, tuple[int]] = None
    srfcs: list['PanelSurface'] = None
    results: dict[str, 'PanelResult | PanelTrim'] = None
    masses: dict[str, 'Mass | MassCollection'] = None # Store Mass Options
    mass: 'Mass | MassCollection | None' = None # Mass Object
    source: str = None
    _hsvs: list['HorseshoeDoublet'] = None
    _numgrd: int = None
    _numpnl: int = None
    _numhsv: int = None
    _numctrl: int = None
    _pnts: Vector = None
    _pnla: 'NDArray' = None
    _nrms: Vector = None
    _rrel: Vector = None
    _apd: 'NDArray' = None
    _aps: 'NDArray' = None
    _aph: 'NDArray' = None
    _apm: 'NDArray' = None
    _bps: dict[float, Vector] = None
    _avd: dict[float, Vector] = None
    _avs: dict[float, Vector] = None
    _avh: dict[float, Vector] = None
    _avm: dict[float, Vector] = None
    _ans: 'NDArray' = None
    _anm: 'NDArray' = None
    _bnm: 'NDArray' = None
    _unsig: dict[float, Vector] = None
    _unmu: dict[float, Vector] = None
    _unphi: dict[float, Vector] = None
    _unnvg: dict[float, Vector] = None
    _hsvpnts: Vector = None
    _hsvnrms: Vector = None
    _awd: 'NDArray' = None
    _aws: 'NDArray' = None
    _awh: 'NDArray' = None
    _adh: 'NDArray' = None
    _ash: 'NDArray' = None
    _alh: 'NDArray' = None
    _ar: float = None
    _area: float = None
    _strps: list['PanelStrip'] = None
    _phind: dict[int, list[int]] = None
    _pnldirx: Vector = None
    _pnldiry: Vector = None
    _pnldirz: Vector = None
    _edges: list['Edge'] = None
    _edges_array: 'NDArray' = None
    _triarr: 'NDArray' = None
    _tgrida: Vector = None
    _tgridb: Vector = None
    _tgridc: Vector = None
    _wgrida: Vector = None
    _wgridb: Vector = None
    _wdirl: Vector = None

    def __init__(self, name: str, bref: float, cref: float,
                 sref: float, rref: Vector) -> None:
        self.name = name
        self.bref = bref
        self.cref = cref
        self.sref = sref
        self.rref = rref
        self.ctrls = {}
        self.results = {}

    def set_mesh(self, grds: dict[int, Grid], pnls: dict[int, Panel]) -> None:
        self.grds = grds
        self.pnls = pnls
        self.update()

    def set_geom(self, srfcs: list['PanelSurface']=None) -> None:
        self.srfcs = srfcs
        self.mesh()
        self.update()

    def update(self) -> None:
        for ind, grd in enumerate(self.grds.values()):
            grd.set_index(ind)
        for ind, pnl in enumerate(self.pnls.values()):
            pnl.set_index(ind)

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr[0] == '_':
                self.__dict__[attr] = None

    def set_horseshoes(self, diro: Vector) -> None:
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
    def ar(self) -> float:
        if self._ar is None:
            self._ar = self.bref**2/self.sref
        return self._ar

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = 0.0
            for pnl in self.pnls.values():
                self._area += pnl.area
        return self._area

    @property
    def numgrd(self) -> int:
        if self._numgrd is None:
            self._numgrd = len(self.grds)
        return self._numgrd

    @property
    def numpnl(self) -> int:
        if self._numpnl is None:
            self._numpnl = len(self.pnls)
        return self._numpnl

    @property
    def numhsv(self) -> int:
        if self._numhsv is None:
            self._numhsv = len(self.hsvs)
        return self._numhsv

    @property
    def numctrl(self) -> int:
        if self._numctrl is None:
            self._numctrl = len(self.ctrls)
        return self._numctrl

    @property
    def pnts(self) -> Vector:
        if self._pnts is None:
            self._pnts = Vector.zeros(self.numpnl)
            for pnl in self.pnls.values():
                self._pnts[pnl.ind] = pnl.pnto
        return self._pnts

    @property
    def rrel(self) -> Vector:
        if self._rrel is None:
            self._rrel = self.pnts - self.rref
        return self._rrel

    @property
    def nrms(self) -> Vector:
        if self._nrms is None:
            self._nrms = Vector.zeros(self.numpnl)
            for pnl in self.pnls.values():
                self._nrms[pnl.ind] = pnl.nrm
        return self._nrms

    @property
    def pnla(self) -> 'NDArray':
        if self._pnla is None:
            self._pnla = zeros(self.numpnl)
            for pnl in self.pnls.values():
                self._pnla[pnl.ind] = pnl.area
        return self._pnla

    @property
    def hsvs(self) -> list['HorseshoeDoublet']:
        if self._hsvs is None:
            self._hsvs = []
            for pnl in self.pnls.values():
                self._hsvs = self._hsvs + pnl.hsvs
        return self._hsvs

    @property
    def phind(self) -> dict[int, int]:
        if self._phind is None:
            self._phind = {}
            for i, hsv in enumerate(self.hsvs):
                pind = hsv.ind
                if pind in self._phind:
                    self._phind[pind].append(i)
                else:
                    self._phind[pind] = [i]
        return self._phind

    @property
    def pnldirx(self) -> Vector:
        if self._pnldirx is None:
            self._pnldirx = Vector.zeros(self.numpnl)
            for pnl in self.pnls.values():
                self._pnldirx[pnl.ind] = pnl.crd.dirx
        return self._pnldirx

    @property
    def pnldiry(self) -> Vector:
        if self._pnldiry is None:
            self._pnldiry = Vector.zeros(self.numpnl)
            for pnl in self.pnls.values():
                self._pnldiry[pnl.ind] = pnl.crd.diry
        return self._pnldiry

    @property
    def pnldirz(self) -> Vector:
        if self._pnldirz is None:
            self._pnldirz = Vector.zeros(self.numpnl)
            for pnl in self.pnls.values():
                self._pnldirz[pnl.ind] = pnl.crd.dirz
        return self._pnldirz

    def bps(self, mach: float = 0.0) -> Vector:
        if self._bps is None:
            self._bps = {}
        if mach not in self._bps:
            self._bps[mach] = -self.aps(mach)@self.unsig(mach)
        return self._bps[mach]

    def bnm(self, mach: float = 0.0) -> Vector:
        if self._bnm is None:
            self._bnm = {}
        if mach not in self._bnm:
            self._bnm[mach] = -self.nrms.reshape((-1, 1)) - self.ans(mach)@self.unsig(mach)
        return self._bnm[mach]

    def apd(self, mach: float = 0.0) -> 'NDArray':
        if self._apd is None:
            self._apd = {}
        if mach not in self._apd:
            # self.assemble_panels(False, mach=mach)
            self.assemble_panels_phi(False, mach=mach)
        return self._apd[mach]

    def avd(self, mach: float = 0.0) -> Vector:
        if self._avd is None:
            self._avd = {}
        if mach not in self._avd:
            self.assemble_panels_vel(False, mach=mach)
        return self._avd[mach]

    def aps(self, mach: float = 0.0) -> 'NDArray':
        if self._aps is None:
            self._aps = {}
        if mach not in self._aps:
            # self.assemble_panels(False, mach=mach)
            self.assemble_panels_phi(False, mach=mach)
        return self._aps[mach]

    def avs(self, mach: float = 0.0) -> Vector:
        if self._avs is None:
            self._avs = {}
        if mach not in self._avs:
            self.assemble_panels_vel(False, mach=mach)
        return self._avs[mach]

    def aph(self, mach: float = 0.0) -> 'NDArray':
        if self._aph is None:
            self._aph = {}
        if mach not in self._aph:
            self.assemble_horseshoes_phi(False, mach=mach)
        return self._aph[mach]

    def avh(self, mach: float = 0.0) -> Vector:
        if self._avh is None:
            self._avh = {}
        if mach not in self._avh:
            self.assemble_horseshoes_vel(False, mach=mach)
        return self._avh[mach]

    def apm(self, mach: float = 0.0) -> 'NDArray':
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

    def avm(self, mach: float = 0.0) -> 'NDArray':
        if self._avm is None:
            self._avm = {}
        if mach not in self._avm:
            avm = self.avd(mach).copy()
            avh = self.avh(mach)
            for i, hsv in enumerate(self.hsvs):
                ind = hsv.ind
                avm[:, ind] = avm[:, ind] + avh[:, i]
            self._avm[mach] = avm
        return self._avm[mach]

    def ans(self, mach: float = 0.0) -> 'NDArray':
        if self._ans is None:
            self._ans = {}
        if mach not in self._ans:
            nrms = self.nrms.reshape((-1, 1)).repeat(self.numpnl, axis=1)
            self._ans[mach] = nrms.dot(self.avs(mach))
        return self._ans[mach]

    def anm(self, mach: float = 0.0) -> 'NDArray':
        if self._anm is None:
            self._anm = {}
        if mach not in self._anm:
            nrms = self.nrms.reshape((-1, 1)).repeat(self.numpnl, axis=1)
            self._anm[mach] = nrms.dot(self.avm(mach))
        return self._anm[mach]

    @property
    def hsvpnts(self) -> Vector:
        if self._hsvpnts is None:
            self._hsvpnts = Vector.zeros(self.numhsv)
            for i, hsv in enumerate(self.hsvs):
                self._hsvpnts[i] = hsv.pnto
        return self._hsvpnts

    @property
    def hsvnrms(self) -> Vector:
        if self._hsvnrms is None:
            self._hsvnrms = Vector.zeros(self.numhsv)
            for i, hsv in enumerate(self.hsvs):
                self._hsvnrms[i] = hsv.nrm
        return self._hsvnrms

    @property
    def strps(self) -> list['PanelStrip']:
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

    def assemble_panels_wash(self, time: bool=True) -> None:

        from .. import USE_CUPY

        if USE_CUPY:
            from ..tools.cupy import cupy_ctdsv as ctdsv
        else:
            from ..tools.numpy import numpy_ctdsv as ctdsv

        # if time:
        start = perf_counter()
        shp = (self.numhsv, self.numpnl)
        self._awd = zeros(shp)
        self._aws = zeros(shp)
        for pnl in self.pnls.values():
            ind = pnl.ind
            _, _, avd, avs = pnl.influence_coefficients(self.hsvpnts)
            self._awd[:, ind] = avd.dot(self.hsvnrms)
            self._aws[:, ind] = avs.dot(self.hsvnrms)
        # if time:
        finish = perf_counter()
        elapsed = finish - start
        print(f'Wash array assembly time is {elapsed:.3f} seconds.')

        start = perf_counter()

        hsvpnts = self.hsvpnts.reshape((-1, 1))
        hsvnrms = self.hsvnrms.reshape((-1, 1))

        avdc, avsc = ctdsv(hsvpnts, self.tgrida, self.tgridb, self.tgridc)
        avdc = add.reduceat(avdc, self.triarr, axis=1)
        avsc = add.reduceat(avsc, self.triarr, axis=1)
        awdc = hsvnrms.dot(avd)
        awsc = hsvnrms.dot(avs)

        finish = perf_counter()
        elapsedc = finish - start
        print(f'Wash array assembly time with cupy is {elapsedc:.3f} seconds.')
        print(f'Speedup is {elapsed/elapsedc:.2f}x.')

        diffawd = self._awd - awdc
        diffaws = self._aws - awsc

        normawd = norm(diffawd)
        normaws = norm(diffaws)

        print(f'Difference in awd: {normawd:.12f}')
        print(f'Difference in aws: {normaws:.12f}')

    def assemble_horseshoes_wash(self, time: bool=True) -> None:
        if time:
            start = perf_counter()
        shp = (self.numhsv, self.numhsv)
        self._awh = zeros(shp)
        for i, hsv in enumerate(self.hsvs):
            avh = hsv.trefftz_plane_velocities(self.hsvpnts)
            self._awh[:, i] = avh.dot(self.hsvnrms)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Wash horse shoe assembly time is {elapsed:.3f} seconds.')

        # start = perf_counter()

        # hsvpnts = self.hsvpnts.reshape((-1, 1))
        # hsvnrms = self.hsvnrms.reshape((-1, 1))

        # avhc = cwdv(hsvpnts, self.wgrida, self.wgridb, self.wdirl)

        # awhc = hsvnrms.dot(avhc)

        # finish = perf_counter()
        # elapsedc = finish - start
        # print(f'Wash horse shoe assembly time with cupy is {elapsedc:.3f} seconds.')
        # print(f'Speedup is {elapsed/elapsedc:.2f}x.')

        # diffawh = self._awh - awhc

        # normawh = norm(diffawh)

        # print(f'Difference in awh: {normawh:.12f}')

    @property
    def awh(self) -> 'NDArray':
        if self._awh is None:
            self.assemble_horseshoes_wash(time=False)
        return self._awh

    @property
    def awd(self) -> 'NDArray':
        if self._awd is None:
            self.assemble_panels_wash(time=False)
        return self._awd

    @property
    def aws(self) -> 'NDArray':
        if self._aws is None:
            self.assemble_panels_wash(time=False)
        return self._aws

    @property
    def adh(self) -> 'NDArray':
        if self._adh is None:
            self._adh = zeros(self.awh.shape)
            for i, hsv in enumerate(self.hsvs):
                self._adh[:, i] = -self._awh[:, i]*hsv.width
        return self._adh

    @property
    def ash(self) -> 'NDArray':
        if self._ash is None:
            self._ash = zeros(self.numhsv)
            for i, hsv in enumerate(self.hsvs):
                self._ash[i] = -hsv.vecab.z
        return self._ash

    @property
    def alh(self) -> 'NDArray':
        if self._alh is None:
            self._alh = zeros(self.numhsv)
            for i, hsv in enumerate(self.hsvs):
                self._alh[i] = hsv.vecab.y
        return self._alh

    @property
    def edges(self, time: bool = False) -> list['Edge']:
        if self._edges is None:
            if time:
                start = perf_counter()
            self._edges = edges_from_system(self)
            if time:
                finish = perf_counter()
                elapsed = finish - start
                print(f'Edges assembly time is {elapsed:.3f} seconds.')
        return self._edges

    @property
    def edges_array(self, time: bool = False) -> 'NDArray':
        if self._edges_array is None:
            if time:
                start = perf_counter()
            self._edges_array = edges_array(self.edges)
            if time:
                finish = perf_counter()
                elapsed = finish - start
                print(f'Edges array assembly time is {elapsed:.3f} seconds.')
        return self._edges_array

    def calc_triarr(self) -> 'NDArray':
        numtria = 0
        for panel in self.pnls.values():
            numtria += panel.num
        self._triarr = zeros(self.numpnl, dtype=int)
        self._tgrida = Vector.zeros((1, numtria))
        self._tgridb = Vector.zeros((1, numtria))
        self._tgridc = Vector.zeros((1, numtria))
        k = 0
        for panel in self.pnls.values():
            self._triarr[panel.ind] = k
            for i in range(-1, panel.num-1):
                a, b = i, i + 1
                self._tgrida[0, k] = panel.grds[a]
                self._tgridb[0, k] = panel.grds[b]
                self._tgridc[0, k] = panel.pnto
                k += 1

    @property
    def triarr(self) -> 'NDArray':
        if self._triarr is None:
            self.calc_triarr()
        return self._triarr

    @property
    def tgrida(self) -> Vector:
        if self._tgrida is None:
            self.calc_triarr()
        return self._tgrida

    @property
    def tgridb(self) -> Vector:
        if self._tgridb is None:
            self.calc_triarr()
        return self._tgridb

    @property
    def tgridc(self) -> Vector:
        if self._tgridc is None:
            self.calc_triarr()
        return self._tgridc

    def calc_wpanel(self) -> 'NDArray':
        self._wgrida = Vector.zeros((1, self.numhsv))
        self._wgridb = Vector.zeros((1, self.numhsv))
        self._wdirl = Vector.zeros((1, self.numhsv))
        for i, hsv in enumerate(self.hsvs):
            self._wgrida[0, i] = hsv.grda
            self._wgridb[0, i] = hsv.grdb
            self._wdirl[0, i] = hsv.diro

    @property
    def wgrida(self) -> Vector:
        if self._wgrida is None:
            self.calc_wpanel()
        return self._wgrida

    @property
    def wgridb(self) -> Vector:
        if self._wgridb is None:
            self.calc_wpanel()
        return self._wgridb

    @property
    def wdirl(self) -> Vector:
        if self._wdirl is None:
            self.calc_wpanel()
        return self._wdirl

    def unsig(self, mach: float = 0.0) -> Vector:
        if self._unsig is None:
            self._unsig = {}
        if mach not in self._unsig:
            unsig = Vector.zeros((self.numpnl, 2 + 4*self.numctrl))
            unsig[:, 0] = -self.nrms
            unsig[:, 1] = self.rrel.cross(self.nrms)
            if self.srfcs is not None:
                for srfc in self.srfcs:
                    for sht in srfc.shts:
                        for control in sht.ctrls:
                            ctrl = sht.ctrls[control]
                            ctup = self.ctrls[control]
                            for pnl in ctrl.pnls:
                                ind = pnl.ind
                                rrel = self.rrel[ind]
                                dndlp = pnl.dndl(ctrl.posgain, ctrl.uhvec)
                                unsig[ind, ctup[0]] = -dndlp
                                unsig[ind, ctup[1]] = -rrel.cross(dndlp)
                                dndln = pnl.dndl(ctrl.neggain, ctrl.uhvec)
                                unsig[ind, ctup[2]] = -dndln
                                unsig[ind, ctup[3]] = -rrel.cross(dndln)
            self._unsig[mach] = unsig
        return self._unsig[mach]

    def unmu(self, mach: float = 0.0) -> Vector:
        if self._unmu is None:
            self._unmu = {}
        if mach not in self._unmu:
            self.solve_dirichlet_system(time=False, mach=mach)
            # self.solve_neumann_system(time=False, mach=mach)
        return self._unmu[mach]

    def unphi(self, mach: float = 0.0) -> Vector:
        if self._unphi is None:
            self._unphi = {}
        if mach not in self._unphi:
            self.solve_dirichlet_system(time=False, mach=mach)
            # self.solve_neumann_system(time=False, mach=mach)
        return self._unphi[mach]

    def unnvg(self, mach: float = 0.0) -> Vector:
        if self._unphi is None:
            self._unphi = {}
        if mach not in self._unnvg:
            self.solve_dirichlet_system(time=False, mach=mach)
            # self.solve_neumann_system(time=False, mach=mach)
        return self._unnvg[mach]

    def assemble_panels_phi(self, time: bool=True, mach=0.0) -> None:
        if self._apd is None:
            self._apd = {}
        if self._aps is None:
            self._aps = {}

        betm = betm_from_mach(mach)

        from .. import USE_CUPY

        if USE_CUPY:
            from ..tools.cupy import cupy_ctdsp as ctdsp
        else:
            from ..tools.numpy import numpy_ctdsp as ctdsp

        # # if time:
        # start = perf_counter()
        # shp = (self.numpnl, self.numpnl)
        # apd = zeros(shp)
        # aps = zeros(shp)
        # for pnl in self.pnls.values():
        #     ind = pnl.ind
        #     apd[:, ind], aps[:, ind] = pnl.velocity_potentials(self.pnts, betx=betm)
        # # if time:
        # finish = perf_counter()
        # elapsed = finish - start
        # print(f'Panel assembly time is {elapsed:.6f} seconds.')

        # start = perf_counter()

        pnts = self.pnts.reshape((-1, 1))

        apdc, apsc = ctdsp(pnts, self.tgrida, self.tgridb, self.tgridc,
                           betx=betm, cond=-1.0)

        apdc = add.reduceat(apdc, self.triarr, axis=1)
        apsc = add.reduceat(apsc, self.triarr, axis=1)

        # finish = perf_counter()
        # elapsedc = finish - start
        # print(f'Panel assembly time with cupy is {elapsedc:.6f} seconds.')
        # print(f'Speedup is {elapsed/elapsedc:.2f}x.')

        # diffapd = apd - apdc
        # diffaps = aps - apsc
        # normapd = norm(diffapd)
        # normaps = norm(diffaps)
        # print(f'Difference in apd: {normapd:.12f}')
        # print(f'Difference in aps: {normaps:.12f}')

        self._apd[mach] = apdc
        self._aps[mach] = apsc

    def assemble_panels_vel(self, time: bool=True, mach: float = 0.0) -> None:
        if self._apd is None:
            self._apd = {}
        if self._aps is None:
            self._aps = {}
        if self._avd is None:
            self._avd = {}
        if self._avs is None:
            self._avs = {}

        betm = betm_from_mach(mach)

        from .. import USE_CUPY

        if USE_CUPY:
            from ..tools.cupy import cupy_ctdsv as ctdsv
        else:
            from ..tools.numpy import numpy_ctdsv as ctdsv

        # # if time:
        # start = perf_counter()
        # shp = (self.numpnl, self.numpnl)
        # apd = zeros(shp)
        # aps = zeros(shp)
        # avd = Vector.zeros(shp)
        # avs = Vector.zeros(shp)
        # for pnl in self.pnls.values():
        #     ind = pnl.ind
        #     apd[:, ind], aps[:, ind], avd[:, ind], avs[:, ind] = pnl.influence_coefficients(self.pnts, betx=betm)
        # # if time:
        # finish = perf_counter()
        # elapsed = finish - start
        # print(f'Full panel assembly time is {elapsed:.3f} seconds.')

        # start = perf_counter()

        pnts = self.pnts.reshape((-1, 1))

        avdc, avsc = ctdsv(pnts, self.tgrida, self.tgridb, self.tgridc,
                           betx=betm, cond=1.0)

        # apdc = add.reduceat(apdc, self.triarr, axis=1)
        avdc = Vector(add.reduceat(avdc.x, self.triarr, axis=1),
                      add.reduceat(avdc.y, self.triarr, axis=1),
                      add.reduceat(avdc.z, self.triarr, axis=1))
        # apsc = add.reduceat(apsc, self.triarr, axis=1)
        avsc = Vector(add.reduceat(avsc.x, self.triarr, axis=1),
                      add.reduceat(avsc.y, self.triarr, axis=1),
                      add.reduceat(avsc.z, self.triarr, axis=1))

        # finish = perf_counter()
        # elapsedc = finish - start
        # print(f'Full panel assembly time with cupy is {elapsedc:.3f} seconds.')
        # print(f'Speedup is {elapsed/elapsedc:.2f}x.')

        # diffapd = apd - apdc
        # diffaps = aps - apsc
        # diffavd = avd - avdc
        # diffavs = avs - avsc
        # normapd = norm(diffapd)
        # normaps = norm(diffaps)
        # normavdx = norm(diffavd.x)
        # normavdy = norm(diffavd.y)
        # normavdz = norm(diffavd.z)
        # normavsx = norm(diffavs.x)
        # normavsy = norm(diffavs.y)
        # normavsz = norm(diffavs.z)
        # print(f'Difference in apd: {normapd:.12f}')
        # print(f'Difference in aps: {normaps:.12f}')
        # print(f'Difference in avd.x: {normavdx:.12f}')
        # print(f'Difference in avd.y: {normavdy:.12f}')
        # print(f'Difference in avd.z: {normavdz:.12f}')
        # print(f'Difference in avs.x: {normavsx:.12f}')
        # print(f'Difference in avs.y: {normavsy:.12f}')
        # print(f'Difference in avs.z: {normavsz:.12f}')

        # self._diffapd = diffapd
        # self._diffaps = diffaps
        # self._diffavdx = diffavd.x
        # self._diffavdy = diffavd.y
        # self._diffavdz = diffavd.z
        # self._diffavsx = diffavs.x
        # self._diffavsy = diffavs.y
        # self._diffavsz = diffavs.z

        # self._avdc = avdc
        # self._avsc = avsc

        # self._apd[mach] = apd
        # self._aps[mach] = aps
        # self._avd[mach] = avd
        # self._avs[mach] = avs

        # self._apd[mach] = apdc
        # self._aps[mach] = apsc
        self._avd[mach] = avdc
        self._avs[mach] = avsc

    def assemble_horseshoes_phi(self, time: bool=True, mach: float = 0.0) -> None:
        if self._aph is None:
            self._aph = {}

        betm = betm_from_mach(mach)

        from .. import USE_CUPY

        if USE_CUPY:
            from ..tools.cupy import cupy_cwdp as cwdp
        else:
            from ..tools.numpy import numpy_cwdp as cwdp

        # # if time:
        # start = perf_counter()
        # shp = (self.numpnl, self.numhsv)
        # aph = zeros(shp)
        # for i, hsv in enumerate(self.hsvs):
        #     aph[:, i] = hsv.doublet_velocity_potentials(self.pnts, betx=betm)
        # # if time:
        # finish = perf_counter()
        # elapsed = finish - start
        # print(f'Horse shoe assembly time is {elapsed:.3f} seconds.')

        # start = perf_counter()

        pnts = self.pnts.reshape((-1, 1))

        aphc = cwdp(pnts, self.wgrida, self.wgridb, self.wdirl,
                    betx=betm)

        # finish = perf_counter()
        # elapsedc = finish - start
        # print(f'Horse shoe assembly time with cupy is {elapsedc:.3f} seconds.')
        # print(f'Speedup is {elapsed/elapsedc:.2f}x.')

        # diffaph = aph - aphc
        # normaph = norm(diffaph)
        # print(f'Difference in aph: {normaph:.12f}')

        self._aph[mach] = aphc

    def assemble_horseshoes_vel(self, time: bool=True, mach: float = 0.0):
        if self._aph is None:
            self._aph = {}
        if self._avh is None:
            self._avh = {}

        betm = betm_from_mach(mach)

        from .. import USE_CUPY

        if USE_CUPY:
            from ..tools.cupy import cupy_cwdv as cwdv
        else:
            from ..tools.numpy import numpy_cwdv as cwdv

        # # if time:
        # start = perf_counter()
        # shp = (self.numpnl, self.numhsv)
        # aph = zeros(shp)
        # avh = Vector.zeros(shp)
        # for i, hsv in enumerate(self.hsvs):
        #     aph[:, i], avh[:, i] = hsv.doublet_influence_coefficients(self.pnts, betx=betm)
        # # if time:
        # finish = perf_counter()
        # elapsed = finish - start
        # print(f'Full horse shoe assembly time is {elapsed:.3f} seconds.')

        # start = perf_counter()

        pnts = self.pnts.reshape((-1, 1))

        avhc = cwdv(pnts, self.wgrida, self.wgridb, self.wdirl,
                    betx=betm)

        # finish = perf_counter()
        # elapsedc = finish - start

        # print(f'Full horse shoe assembly time with cupy is {elapsedc:.3f} seconds.')
        # print(f'Speedup is {elapsed/elapsedc:.2f}x.')

        # diffaph = aph - aphc
        # diffavh = avh - avhc
        # normaph = norm(diffaph)
        # normavh = norm(diffavh.x) + norm(diffavh.y) + norm(diffavh.z)
        # print(f'Difference in aph: {normaph:.12f}')
        # print(f'Difference in avh: {normavh:.12f}')

        # self._aph[mach] = aphc
        self._avh[mach] = avhc

    def solve_system(self, time: bool=True, mach: float = 0.0) -> None:
        if time:
            start = perf_counter()
        self.solve_dirichlet_system(time=False, mach=mach)
        # self.solve_neumann_system(time=False, mach=mach)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')

    def solve_dirichlet_system(self, time: bool=True, mach: float = 0.0) -> None:
        if time:
            start = perf_counter()
        if self._unmu is None:
            self._unmu = {}
        self._unmu[mach] = self.bps(mach).solve(self.apm(mach))
        if self._unphi is None:
            self._unphi = {}
        self._unphi[mach] = self.apm(mach)@self.unmu(mach) - self.bps(mach)
        if self._unnvg is None:
            self._unnvg = {}
        self._unnvg[mach] = Vector.zeros(self._unphi[mach].shape)
        # if self._anm is not None and self._bnm is not None:
        #     if mach in self._anm and mach in self._bnm:
        # self._unnvg[mach] = self.anm(mach)@self.unmu(mach) - self.bnm(mach)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')

    def solve_neumann_system(self, time: bool=True, mach: float = 0.0) -> None:
        if time:
            start = perf_counter()
        if self._unmu is None:
            self._unmu = {}
        self._unmu[mach] = self.bnm(mach).solve(self.anm(mach))
        if self._unnvg is None:
            self._unnvg = {}
        self._unnvg[mach] = self.anm(mach)@self.unmu(mach) - self.bnm(mach)
        if self._unphi is None:
            self._unphi = {}
        self._unphi[mach] = Vector.zeros(self._unnvg[mach].shape)
        # if self._apm is not None and self._bps is not None:
        #     if mach in self._apm and mach in self._bps:
        # self._unphi[mach] = self.apm(mach)@self.unmu(mach) - self.bps(mach)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'System solution time is {elapsed:.3f} seconds.')

    def plot_twist_distribution(self, ax: 'Axes'=None, axis: str='b',
                                surfaces: list['PanelSurface']=None) -> 'Axes':
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

    def plot_chord_distribution(self, ax: 'Axes'=None, axis: str='b', surfaces: list=None):
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

    def plot_tilt_distribution(self, ax: 'Axes'=None, axis: str='b', surfaces: list=None):
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

    def plot_strip_width_distribution(self, ax: 'Axes'=None, axis: str='b', surfaces: list=None):
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

    def mesh(self) -> None:
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

    def trim(self) -> None:
        for result in self.results.values():
            if isinstance(result, PanelTrim):
                result.trim()

    @classmethod
    def from_json(cls, jsonfilepath: str,
                  trim: bool = True) -> 'PanelSystem':
        """Create a PanelSystem from a JSON file."""

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
            sys = cls.from_geom(sysdct, trim=False)
        elif filetype == 'mesh':
            sys = cls.from_mesh(sysdct, trim=False)
        else:
            raise ValueError('Incorrect file type.')

        sys.source = jsonfilepath

        sys.load_initial_state(sys.source)

        if trim:
            sys.trim()

        return sys

    @classmethod
    def from_mesh(cls, sysdct: dict[str, Any],
                  trim: bool = True) -> 'PanelSystem':

        name = sysdct['name']
        bref = sysdct['bref']
        cref = sysdct['cref']
        sref = sysdct['sref']
        xref = sysdct['xref']
        yref = sysdct['yref']
        zref = sysdct['zref']
        rref = Vector(xref, yref, zref)

        grds: dict[int, Grid] = {}
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

        pnls: dict[int, Panel] = {}
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

        sys = cls(name, bref, cref, sref, rref)
        sys.set_mesh(grds, pnls)

        masses = {}
        if 'masses' in sysdct:
            if isinstance(sysdct['masses'], list):
                masses = masses_from_json(sysdct['masses'])
        sys.masses = masses

        if 'cases' in sysdct and sysdct:
            sys.results_from_dict(sysdct['cases'], trim = False)

        if 'source' in sysdct:
            sys.source = sysdct['source']

        sys.load_initial_state(sys.source)

        if trim:
            sys.trim()

        return sys

    @classmethod
    def from_geom(cls, sysdct: dict[str, any],
                  trim: bool = True) -> 'PanelSystem':

        jsonfilepath = sysdct.get('source', '.')

        path = dirname(jsonfilepath)

        surfsdata: list[dict[str, Any]] = sysdct.get('surfaces', [])

        for surfdata in surfsdata:
            if 'defaults' in surfdata:
                if 'airfoil' in surfdata['defaults']:
                    airfoil = surfdata['defaults']['airfoil']
                    if airfoil[-4:] == '.dat':
                        airfoil = join(path, airfoil)
                        if not exists(airfoil):
                            print(f'Airfoil {airfoil} does not exist.')
                            del surfdata['defaults']['airfoil']
                        else:
                            surfdata['defaults']['airfoil'] = airfoil
            sectsdata: list[dict[str, Any]] = surfdata.get('sections', [])
            for sectdata in sectsdata:
                if 'airfoil' in sectdata:
                    airfoil = sectdata['airfoil']
                    if airfoil[-4:] == '.dat':
                        airfoil = join(path, airfoil)
                        if not exists(airfoil):
                            print(f'Airfoil {airfoil} does not exist.')
                            del sectdata['airfoil']
                        else:
                            sectdata['airfoil'] = airfoil

        name = sysdct['name']
        srfcs = []
        for surfdata in sysdct['surfaces']:
            srfc = PanelSurface.from_dict(surfdata)
            srfcs.append(srfc)
        bref = sysdct['bref']
        cref = sysdct['cref']
        sref = sysdct['sref']
        xref = sysdct['xref']
        yref = sysdct['yref']
        zref = sysdct['zref']
        rref = Vector(xref, yref, zref)
        sys = cls(name, bref, cref, sref, rref)
        sys.set_geom(srfcs)

        masses = {}
        if 'masses' in sysdct:
            if isinstance(sysdct['masses'], dict):
                masses = masses_from_data(sysdct['masses'])
            elif isinstance(sysdct['masses'], str):
                if sysdct['masses'][-5:] == '.json':
                    massfilename = sysdct['masses']
                    massfilepath = join(path, massfilename)
                masses = masses_from_json(massfilepath)
        sys.masses = masses
        mass = sysdct.get('mass', None)
        if isinstance(mass, float):
            sys.mass = Mass(sys.name, mass = mass, xcm = sys.rref.x,
                        ycm = sys.rref.y, zcm = sys.rref.z)
        elif isinstance(mass, str):
            sys.mass = masses[mass]
        else:
            sys.mass = Mass(sys.name, mass = 1.0, xcm = sys.rref.x,
                        ycm = sys.rref.y, zcm = sys.rref.z)

        if 'cases' in sysdct and sysdct:
            sys.results_from_dict(sysdct['cases'], trim = False)

        sys.source = jsonfilepath

        sys.load_initial_state(sys.source)

        if trim:
            sys.trim()

        return sys

    def results_from_dict(self, cases: dict[str, Any],
                          trim: bool = True) -> 'PanelResult':

        for i in range(len(cases)):
            resdata = cases[i]
            if 'trim' in resdata:
                PanelTrim.from_dict(self, resdata, trim=trim)
            else:
                PanelResult.from_dict(self, resdata)

    def save_initial_state(self, infilepath: str,
                           outfilepath: str | None = None,
                           tolerance: float = 1e-10) -> None:

        if not exists(infilepath):
            raise FileNotFoundError(f"Input file {infilepath} does not exist.")

        with open(infilepath, 'r') as jsonfile:
            data = load(jsonfile)

        data['state'] = {}
        for resname, result in self.results.items():
            data['state'][resname] = {}
            if abs(result.alpha) > tolerance:
                data['state'][resname]['alpha'] = result.alpha
            if abs(result.beta) > tolerance:
                data['state'][resname]['beta'] = result.beta
            if abs(result.pbo2v) > tolerance:
                data['state'][resname]['pbo2v'] = result.pbo2v
            if abs(result.qco2v) > tolerance:
                data['state'][resname]['qco2v'] = result.qco2v
            if abs(result.rbo2v) > tolerance:
                data['state'][resname]['rbo2v'] = result.rbo2v
            for control in self.ctrls:
                if abs(result.ctrls[control]) > tolerance:
                    data['state'][resname][control] = result.ctrls[control]

        if outfilepath is None:
            outfilepath = infilepath

        with open(outfilepath, 'w') as jsonfile:
            dump(data, jsonfile, indent=4)

    def load_initial_state(self, infilepath: str) -> None:

        if exists(infilepath):

            with open(infilepath, 'r') as jsonfile:
                data: dict[str, Any] = load(jsonfile)

            state: dict[str, Any] = data.get('state', {})

            for result in self.results.values():
                resdata: dict[str, Any] = state.get(result.name, {})
                result.alpha = resdata.get('alpha', result.alpha)
                result.beta = resdata.get('beta', result.beta)
                result.pbo2v = resdata.get('pbo2v', result.pbo2v)
                result.qco2v = resdata.get('qco2v', result.qco2v)
                result.rbo2v = resdata.get('rbo2v', result.rbo2v)
                for control in self.ctrls:
                    value = result.ctrls[control]
                    result.ctrls[control] = resdata.get(control, value)

    def __repr__(self) -> str:
        return '<PanelSystem: {:s}>'.format(self.name)

    def __str__(self) -> str:
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
        if self.ctrls is not None:
            table.add_column('# Controls', 'd', data=[self.numctrl])
        else:
            table.add_column('# Controls', 'd', data=[0])
        if len(table.columns) > 0:
            outstr += table._repr_markdown_()
        return outstr

    def _repr_markdown_(self) -> str:
        return self.__str__()
