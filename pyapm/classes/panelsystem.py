from time import perf_counter
from typing import TYPE_CHECKING, Any

from matplotlib.pyplot import figure
from numpy import zeros
from py2md.classes import MDTable
from pygeom.geom3d import Vector

from ..tools import betm_from_mach
from ..tools.mass import Mass, masses_from_data, masses_from_json
from .grid import Grid
from .panel import Panel
from .panelresult import panelresult_from_dict
from .panelsurface import panelsurface_from_json
from .paneltrim import paneltrim_from_dict

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..tools.mass import MassCollection
    from .horseshoedoublet import HorseshoeDoublet
    from .panelresult import PanelResult
    from .panelstrip import PanelStrip
    from .panelsurface import PanelSurface
    from .paneltrim import PanelTrim


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
    def pnts(self) -> int:
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
            self._bnm[mach] = -self.nrms - self.ans(mach)@self.unsig(mach)
        return self._bnm[mach]

    def apd(self, mach: float = 0.0) -> 'NDArray':
        if self._apd is None:
            self._apd = {}
        if mach not in self._apd:
            self.assemble_panels(False, mach=mach)
        return self._apd[mach]

    def avd(self, mach: float = 0.0) -> 'NDArray':
        if self._avd is None:
            self._avd = {}
        if mach not in self._avd:
            self.assemble_panels_full(False, mach=mach)
        return self._avd[mach]

    def aps(self, mach: float = 0.0) -> 'NDArray':
        if self._aps is None:
            self._aps = {}
        if mach not in self._aps:
            self.assemble_panels(False, mach=mach)
        return self._aps[mach]

    def avs(self, mach: float = 0.0) -> 'NDArray':
        if self._avs is None:
            self._avs = {}
        if mach not in self._avs:
            self.assemble_panels_full(False, mach=mach)
        return self._avs[mach]

    def aph(self, mach: float = 0.0) -> 'NDArray':
        if self._aph is None:
            self._aph = {}
        if mach not in self._aph:
            self.assemble_horseshoes(False, mach=mach)
        return self._aph[mach]

    def avh(self, mach: float = 0.0) -> 'NDArray':
        if self._avh is None:
            self._avh = {}
        if mach not in self._avh:
            self.assemble_horseshoes_full(False, mach=mach)
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
        if self._avm is None:
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
            nrms = self.nrms.repeat(self.numpnl, axis=1)
            self._ans[mach] = nrms.dot(self.avs(mach))
        return self._ans[mach]

    def anm(self, mach: float = 0.0) -> 'NDArray':
        if self._anm is None:
            self._anm = {}
        if mach not in self._anm:
            nrms = self.nrms.repeat(self.numpnl, axis=1)
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
        if time:
            start = perf_counter()
        shp = (self.numhsv, self.numpnl)
        self._awd = zeros(shp)
        self._aws = zeros(shp)
        for pnl in self.pnls.values():
            ind = pnl.ind
            _, _, avd, avs = pnl.influence_coefficients(self.hsvpnts)
            self._awd[:, ind] = avd.dot(self.hsvnrms)
            self._aws[:, ind] = avs.dot(self.hsvnrms)
        if time:
            finish = perf_counter()
            elapsed = finish - start
            print(f'Wash array assembly time is {elapsed:.3f} seconds.')

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
        return self._unmu[mach]

    def unphi(self, mach: float = 0.0) -> Vector:
        if self._unphi is None:
            self._unphi = {}
        if mach not in self._unphi:
            self.solve_dirichlet_system(time=False, mach=mach)
        return self._unphi[mach]

    def assemble_panels(self, time: bool=True, mach=0.0) -> None:
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        apd = zeros(shp)
        aps = zeros(shp)
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

    def assemble_panels_full(self, time: bool=True, mach: float = 0.0) -> None:
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        apd = zeros(shp)
        aps = zeros(shp)
        avd = Vector.zeros(shp)
        avs = Vector.zeros(shp)
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

    def assemble_horseshoes(self, time: bool=True, mach: float = 0.0) -> None:
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numhsv)
        aph = zeros(shp)
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

    def assemble_horseshoes_full(self, time: bool=True, mach: float = 0.0):
        if time:
            start = perf_counter()
        shp = (self.numpnl, self.numpnl)
        aph = zeros(shp)
        avh = Vector.zeros(shp)
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

    def solve_system(self, time: bool=True, mach: float = 0.0) -> None:
        if time:
            start = perf_counter()
        self.solve_dirichlet_system(time=False, mach=mach)
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
        self._unphi[mach] = self.apm(mach)@self.unmu(mach) + self.bps(mach)
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
        self._unnvg[mach] = self.anm(mach)@self.unmu(mach) + self.bnm(mach)
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

def panelsystem_from_json(jsonfilepath: str,
                          trim: bool = True) -> PanelSystem:
    from json import load

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
        sys = panelsystem_from_geom(sysdct, trim=trim)
    elif filetype == 'mesh':
        sys = panelsystem_from_mesh(sysdct, trim=trim)
    else:
        raise ValueError('Incorrect file type.')

    sys.source = jsonfilepath

    return sys

def panelsystem_from_mesh(sysdct: dict[str, any], trim: bool = True) -> PanelSystem:

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

    sys = PanelSystem(name, bref, cref, sref, rref)
    sys.set_mesh(grds, pnls)

    masses = {}
    if 'masses' in sysdct:
        if isinstance(sysdct['masses'], list):
            masses = masses_from_json(sysdct['masses'])
    sys.masses = masses

    if 'cases' in sysdct and sysdct:
        panelresults_from_dict(sys, sysdct['cases'], trim = trim)

    if 'source' in sysdct:
        sys.source = sysdct['source']

    return sys

def panelsystem_from_geom(sysdct: dict[str, any], trim: bool = True) -> PanelSystem:

    from os.path import dirname, exists, join

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
        srfc = panelsurface_from_json(surfdata)
        srfcs.append(srfc)
    bref = sysdct['bref']
    cref = sysdct['cref']
    sref = sysdct['sref']
    xref = sysdct['xref']
    yref = sysdct['yref']
    zref = sysdct['zref']
    rref = Vector(xref, yref, zref)
    sys = PanelSystem(name, bref, cref, sref, rref)
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
        panelresults_from_dict(sys, sysdct['cases'], trim = trim)

    sys.source = jsonfilepath

    return sys

def panelresults_from_dict(sys: PanelSystem, cases: dict[str, Any],
                           trim: bool = True) -> 'PanelResult':

    for i in range(len(cases)):
        resdata = cases[i]
        if 'trim' in resdata:
            paneltrim_from_dict(sys, resdata, trim = trim)
        else:
            panelresult_from_dict(sys, resdata)
