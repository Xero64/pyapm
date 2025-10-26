from json import dump, load
from os.path import dirname, exists, join
# from time import perf_counter
from typing import TYPE_CHECKING, Any

# from matplotlib.pyplot import figure
from numpy import zeros, eye
from numpy.linalg import inv
from py2md.classes import MDTable
from pygeom.geom3d import Vector
from pygeom.geom2d import Vector2D

from ..tools import betm_from_mach
from ..tools.mass import MassObject, masses_from_data, masses_from_json
from .edge import edges_array, edges_from_system
from .grid import Grid
from .panel import Panel
from .panelcontrol import PanelControl
from .panelgroup import PanelGroup
from .panelresult import PanelResult
from .panelsurface import PanelSurface
from .paneltrim import PanelTrim

if TYPE_CHECKING:
    # from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..tools.mass import MassCollection
    from .edge import Edge
    from .face import Face
    from .panelstrip import PanelStrip
    from .wakepanel import WakePanel


class PanelSystem():
    name: str = None
    grids: dict[int, Grid] = None
    dpanels: dict[int, Panel] = None
    wpanels: dict[int, 'WakePanel'] = None
    bref: float = None
    cref: float = None
    sref: float = None
    rref: Vector = None
    controls: dict[str, tuple[int]] = None
    surfaces: list['PanelSurface'] = None
    groups: dict[int, 'PanelGroup'] = None
    results: dict[str, 'PanelResult | PanelTrim'] = None
    masses: dict[str, 'MassObject | MassCollection'] = None # Store Mass Options
    mass: 'MassObject | MassCollection | None' = None # Mass Object
    source: str = None
    _num_grids: int = None
    _num_dpanels: int = None
    _num_wpanels: int = None
    _num_controls: int = None
    _points: Vector = None
    _panel_area: 'NDArray' = None
    _normals: Vector = None
    _rrel: Vector = None
    _aphdd: dict[float, 'NDArray'] = None
    _aphsd: dict[float, 'NDArray'] = None
    _aphdw: dict[float, 'NDArray'] = None
    # _aph: 'NDArray' = None
    # _apm: 'NDArray' = None
    _bphs: dict[float, Vector] = None
    _ainv: dict[float, 'NDArray'] = None
    _cmat: 'NDArray' = None
    # _avd: dict[float, Vector] = None
    # _avs: dict[float, Vector] = None
    # _avh: dict[float, Vector] = None
    # _avm: dict[float, Vector] = None
    # _ans: 'NDArray' = None
    # _anm: 'NDArray' = None
    # _bnm: 'NDArray' = None
    _unsig: dict[float, Vector] = None
    _unmud: dict[float, Vector] = None
    _unmuw: dict[float, Vector] = None
    _unphi: dict[float, Vector] = None
    # _unnvg: dict[float, Vector] = None
    # _hsvpnts: Vector = None
    # _hsvnrms: Vector = None
    # _awd: 'NDArray' = None
    # _aws: 'NDArray' = None
    # _awh: 'NDArray' = None
    # _adh: 'NDArray' = None
    # _ash: 'NDArray' = None
    # _alh: 'NDArray' = None
    _ar: float = None
    _area: float = None
    _strips: list['PanelStrip'] = None
    _phind: dict[int, list[int]] = None
    # _pnldirx: Vector = None
    # _pnldiry: Vector = None
    # _pnldirz: Vector = None
    _edges: list['Edge'] = None
    _edges_array: 'NDArray' = None
    # _triarr: 'NDArray' = None
    # _tgrida: Vector = None
    # _tgridb: Vector = None
    # _tgridc: Vector = None
    # _wgrida: Vector = None
    # _wgridb: Vector = None
    # _wdirl: Vector = None
    _dfaces: list['Face'] = None
    _num_dfaces: int = None
    _dface_pnt: Vector = None
    _dface_indg: 'NDArray' = None
    _dface_indp: 'NDArray' = None
    _dface_velg: Vector2D = None
    _dface_velp: Vector2D = None
    _dface_dirx: Vector2D = None
    _dface_diry: Vector2D = None
    _dface_dirz: Vector2D = None
    _dface_area: 'NDArray' = None

    def __init__(self, name: str, bref: float, cref: float,
                 sref: float, rref: Vector) -> None:
        self.name = name
        self.bref = bref
        self.cref = cref
        self.sref = sref
        self.rref = rref
        self.controls = {}
        self.results = {}

    def set_mesh(self, grids: dict[int, Grid],
                 dpanels: dict[int, Panel],
                 wpanels: dict[int, 'WakePanel']) -> None:
        self.grids = grids
        self.dpanels = dpanels
        self.wpanels = wpanels
        self.update()

    def set_geom(self, surfaces: list['PanelSurface']=None) -> None:
        self.surfaces = surfaces
        self.mesh()
        self.update()

    def update(self) -> None:
        for ind, grid in enumerate(self.grids.values()):
            grid.ind = ind
        for ind, panel in enumerate(self.dpanels.values()):
            panel.ind = ind
        for ind, panel in enumerate(self.wpanels.values()):
            panel.ind = ind

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                self.__dict__[attr] = None

    @property
    def ar(self) -> float:
        if self._ar is None:
            self._ar = self.bref**2/self.sref
        return self._ar

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = 0.0
            for panel in self.dpanels.values():
                self._area += panel.area
        return self._area

    @property
    def num_grids(self) -> int:
        if self._num_grids is None:
            self._num_grids = len(self.grids)
        return self._num_grids

    @property
    def num_dpanels(self) -> int:
        if self._num_dpanels is None:
            self._num_dpanels = len(self.dpanels)
        return self._num_dpanels

    @property
    def num_wpanels(self) -> int:
        if self._num_wpanels is None:
            self._num_wpanels = len(self.wpanels)
        return self._num_wpanels

    @property
    def num_controls(self) -> int:
        if self._num_controls is None:
            self._num_controls = len(self.controls)
        return self._num_controls

    @property
    def points(self) -> Vector:
        if self._points is None:
            self._points = Vector.zeros(self.num_dpanels)
            for panel in self.dpanels.values():
                self._points[panel.ind] = panel.pnto
        return self._points

    @property
    def rrel(self) -> Vector:
        if self._rrel is None:
            self._rrel = self.points - self.rref
        return self._rrel

    @property
    def normals(self) -> Vector:
        if self._normals is None:
            self._normals = Vector.zeros(self.num_dpanels)
            for panel in self.dpanels.values():
                self._normals[panel.ind] = panel.nrm
        return self._normals

    @property
    def panel_area(self) -> 'NDArray':
        if self._panel_area is None:
            self._panel_area = zeros(self.num_dpanels)
            for panel in self.dpanels.values():
                self._panel_area[panel.ind] = panel.area
        return self._panel_area

    # @property
    # def pnldirx(self) -> Vector:
    #     if self._pnldirx is None:
    #         self._pnldirx = Vector.zeros(self.num_dpanels)
    #         for panel in self.dpanels.values():
    #             self._pnldirx[panel.ind] = panel.crd.dirx
    #     return self._pnldirx

    # @property
    # def pnldiry(self) -> Vector:
    #     if self._pnldiry is None:
    #         self._pnldiry = Vector.zeros(self.num_dpanels)
    #         for panel in self.dpanels.values():
    #             self._pnldiry[panel.ind] = panel.crd.diry
    #     return self._pnldiry

    # @property
    # def pnldirz(self) -> Vector:
    #     if self._pnldirz is None:
    #         self._pnldirz = Vector.zeros(self.num_dpanels)
    #         for panel in self.dpanels.values():
    #             self._pnldirz[panel.ind] = panel.crd.dirz
    #     return self._pnldirz

    def bphs(self, mach: float = 0.0) -> Vector:
        if self._bphs is None:
            self._bphs = {}
        if mach not in self._bphs:
            self._bphs[mach] = -self.aphsd(mach)@self.unsig(mach)
        return self._bphs[mach]

    # def bnm(self, mach: float = 0.0) -> Vector:
    #     if self._bnm is None:
    #         self._bnm = {}
    #     if mach not in self._bnm:
    #         self._bnm[mach] = -self.normals.reshape((-1, 1)) - self.ans(mach)@self.unsig(mach)
    #     return self._bnm[mach]

    def aphdd(self, mach: float = 0.0) -> 'NDArray':
        if self._aphdd is None:
            self._aphdd = {}
        if mach not in self._aphdd:
            self.assemble_panels_phi(mach = mach)
        return self._aphdd[mach]

    # def avd(self, mach: float = 0.0) -> Vector:
    #     if self._avd is None:
    #         self._avd = {}
    #     if mach not in self._avd:
    #         self.assemble_panels_vel(mach = mach)
    #     return self._avd[mach]

    def aphsd(self, mach: float = 0.0) -> 'NDArray':
        if self._aphsd is None:
            self._aphsd = {}
        if mach not in self._aphsd:
            self.assemble_panels_phi(mach = mach)
        return self._aphsd[mach]

    def aphdw(self, mach: float = 0.0) -> 'NDArray':
        if self._aphdw is None:
            self._aphdw = {}
        if mach not in self._aphdw:
            self.assemble_panels_phi(mach = mach)
        return self._aphdw[mach]

    # def avs(self, mach: float = 0.0) -> Vector:
    #     if self._avs is None:
    #         self._avs = {}
    #     if mach not in self._avs:
    #         self.assemble_panels_vel(mach = mach)
    #     return self._avs[mach]

    # def aph(self, mach: float = 0.0) -> 'NDArray':
    #     if self._aph is None:
    #         self._aph = {}
    #     if mach not in self._aph:
    #         self.assemble_horseshoes_phi(mach = mach)
    #     return self._aph[mach]

    # def avh(self, mach: float = 0.0) -> Vector:
    #     if self._avh is None:
    #         self._avh = {}
    #     if mach not in self._avh:
    #         self.assemble_horseshoes_vel(mach = mach)
    #     return self._avh[mach]

    # def apm(self, mach: float = 0.0) -> 'NDArray':
    #     if self._apm is None:
    #         self._apm = {}
    #     if mach not in self._apm:
    #         apm = self.apd(mach).copy()
    #         aph = self.aph(mach)
    #         for i, hsv in enumerate(self.hsvs):
    #             ind = hsv.ind
    #             apm[:, ind] = apm[:, ind] + aph[:, i]
    #         self._apm[mach] = apm
    #     return self._apm[mach]

    # def avm(self, mach: float = 0.0) -> 'NDArray':
    #     if self._avm is None:
    #         self._avm = {}
    #     if mach not in self._avm:
    #         avm = self.avd(mach).copy()
    #         avh = self.avh(mach)
    #         for i, hsv in enumerate(self.hsvs):
    #             ind = hsv.ind
    #             avm[:, ind] = avm[:, ind] + avh[:, i]
    #         self._avm[mach] = avm
    #     return self._avm[mach]

    # def ans(self, mach: float = 0.0) -> 'NDArray':
    #     if self._ans is None:
    #         self._ans = {}
    #     if mach not in self._ans:
    #         nrms = self.nrms.reshape((-1, 1)).repeat(self.numpnl, axis=1)
    #         self._ans[mach] = nrms.dot(self.avs(mach))
    #     return self._ans[mach]

    # def anm(self, mach: float = 0.0) -> 'NDArray':
    #     if self._anm is None:
    #         self._anm = {}
    #     if mach not in self._anm:
    #         nrms = self.nrms.reshape((-1, 1)).repeat(self.numpnl, axis=1)
    #         self._anm[mach] = nrms.dot(self.avm(mach))
    #     return self._anm[mach]

    # @property
    # def hsvpnts(self) -> Vector:
    #     if self._hsvpnts is None:
    #         self._hsvpnts = Vector.zeros(self.numhsv)
    #         for i, hsv in enumerate(self.hsvs):
    #             self._hsvpnts[i] = hsv.pnto
    #     return self._hsvpnts

    # @property
    # def hsvnrms(self) -> Vector:
    #     if self._hsvnrms is None:
    #         self._hsvnrms = Vector.zeros(self.numhsv)
    #         for i, hsv in enumerate(self.hsvs):
    #             self._hsvnrms[i] = hsv.nrm
    #     return self._hsvnrms

    @property
    def strips(self) -> list['PanelStrip']:
        if self._strips is None:
            if self.surfaces is not None:
                self._strips = []
                ind = 0
                for surface in self.surfaces:
                    for strip in surface.strips:
                        strip.ind = ind
                        self._strips.append(strip)
                        ind += 1
        return self._strips

    # def assemble_panels_wash(self) -> None:

    #     # from .. import USE_CUPY

    #     # if USE_CUPY:
    #     #     from ..tools.cupy import cupy_ctdsv as ctdsv
    #     # else:
    #     #     from ..tools.numpy import numpy_ctdsv as ctdsv

    #     start = perf_counter()
    #     shp = (self.numhsv, self.numpnl)
    #     self._awd = zeros(shp)
    #     self._aws = zeros(shp)
    #     for pnl in self.pnls.values():
    #         ind = pnl.ind
    #         _, _, avd, avs = pnl.influence_coefficients(self.hsvpnts)
    #         self._awd[:, ind] = avd.dot(self.hsvnrms)
    #         self._aws[:, ind] = avs.dot(self.hsvnrms)
    #     finish = perf_counter()
    #     elapsed = finish - start
    #     print(f'Wash array assembly time is {elapsed:.3f} seconds.')

    #     # start = perf_counter()

    #     # hsvpnts = self.hsvpnts.reshape((-1, 1))
    #     # hsvnrms = self.hsvnrms.reshape((-1, 1))

    #     # avdc, avsc = ctdsv(hsvpnts, self.tgrida, self.tgridb, self.tgridc)
    #     # avdc = add.reduceat(avdc, self.triarr, axis=1)
    #     # avsc = add.reduceat(avsc, self.triarr, axis=1)
    #     # awdc = hsvnrms.dot(avd)
    #     # awsc = hsvnrms.dot(avs)

    #     # finish = perf_counter()
    #     # elapsedc = finish - start
    #     # print(f'Wash array assembly time with cupy is {elapsedc:.3f} seconds.')
    #     # print(f'Speedup is {elapsed/elapsedc:.2f}x.')

    #     # diffawd = self._awd - awdc
    #     # diffaws = self._aws - awsc

    #     # normawd = norm(diffawd)
    #     # normaws = norm(diffaws)

    #     # print(f'Difference in awd: {normawd:.12f}')
    #     # print(f'Difference in aws: {normaws:.12f}')

    # def assemble_horseshoes_wash(self) -> None:
    #     start = perf_counter()
    #     shp = (self.numhsv, self.numhsv)
    #     self._awh = zeros(shp)
    #     for i, hsv in enumerate(self.hsvs):
    #         avh = hsv.trefftz_plane_velocities(self.hsvpnts)
    #         self._awh[:, i] = avh.dot(self.hsvnrms)
    #     finish = perf_counter()
    #     elapsed = finish - start
    #     print(f'Wash horse shoe assembly time is {elapsed:.3f} seconds.')

    # @property
    # def awh(self) -> 'NDArray':
    #     if self._awh is None:
    #         self.assemble_horseshoes_wash()
    #     return self._awh

    # @property
    # def awd(self) -> 'NDArray':
    #     if self._awd is None:
    #         self.assemble_panels_wash()
    #     return self._awd

    # @property
    # def aws(self) -> 'NDArray':
    #     if self._aws is None:
    #         self.assemble_panels_wash()
    #     return self._aws

    # @property
    # def adh(self) -> 'NDArray':
    #     if self._adh is None:
    #         self._adh = zeros(self.awh.shape)
    #         for i, hsv in enumerate(self.hsvs):
    #             self._adh[:, i] = -self._awh[:, i]*hsv.width
    #     return self._adh

    # @property
    # def ash(self) -> 'NDArray':
    #     if self._ash is None:
    #         self._ash = zeros(self.numhsv)
    #         for i, hsv in enumerate(self.hsvs):
    #             self._ash[i] = -hsv.vecab.z
    #     return self._ash

    # @property
    # def alh(self) -> 'NDArray':
    #     if self._alh is None:
    #         self._alh = zeros(self.numhsv)
    #         for i, hsv in enumerate(self.hsvs):
    #             self._alh[i] = hsv.vecab.y
    #     return self._alh

    @property
    def edges(self) -> list['Edge']:
        if self._edges is None:
            self._edges = edges_from_system(self)
        return self._edges

    @property
    def edges_array(self) -> 'NDArray':
        if self._edges_array is None:
            self._edges_array = edges_array(self.edges)
        return self._edges_array

    # def calc_triarr(self) -> 'NDArray':
    #     numtria = 0
    #     for panel in self.pnls.values():
    #         if panel.num == 3:
    #             numtria += 1
    #         else:
    #             numtria += panel.num
    #     self._triarr = zeros(self.numpnl, dtype=int)
    #     self._tgrida = Vector.zeros((1, numtria))
    #     self._tgridb = Vector.zeros((1, numtria))
    #     self._tgridc = Vector.zeros((1, numtria))
    #     k = 0
    #     for panel in self.pnls.values():
    #         self._triarr[panel.ind] = k
    #         if panel.num == 3:
    #             self._tgrida[0, k] = panel.grds[0]
    #             self._tgridb[0, k] = panel.grds[1]
    #             self._tgridc[0, k] = panel.grds[2]
    #             k += 1
    #         else:
    #             for i in range(-1, panel.num-1):
    #                 a, b = i, i + 1
    #                 self._tgrida[0, k] = panel.grds[a]
    #                 self._tgridb[0, k] = panel.grds[b]
    #                 self._tgridc[0, k] = panel.pnto
    #                 k += 1

    # @property
    # def triarr(self) -> 'NDArray':
    #     if self._triarr is None:
    #         self.calc_triarr()
    #     return self._triarr

    # @property
    # def tgrida(self) -> Vector:
    #     if self._tgrida is None:
    #         self.calc_triarr()
    #     return self._tgrida

    # @property
    # def tgridb(self) -> Vector:
    #     if self._tgridb is None:
    #         self.calc_triarr()
    #     return self._tgridb

    # @property
    # def tgridc(self) -> Vector:
    #     if self._tgridc is None:
    #         self.calc_triarr()
    #     return self._tgridc

    # def calc_wpanel(self) -> 'NDArray':
    #     self._wgrida = Vector.zeros((1, self.numhsv))
    #     self._wgridb = Vector.zeros((1, self.numhsv))
    #     self._wdirl = Vector.zeros((1, self.numhsv))
    #     for i, hsv in enumerate(self.hsvs):
    #         self._wgrida[0, i] = hsv.grda
    #         self._wgridb[0, i] = hsv.grdb
    #         self._wdirl[0, i] = hsv.diro

    # @property
    # def wgrida(self) -> Vector:
    #     if self._wgrida is None:
    #         self.calc_wpanel()
    #     return self._wgrida

    # @property
    # def wgridb(self) -> Vector:
    #     if self._wgridb is None:
    #         self.calc_wpanel()
    #     return self._wgridb

    # @property
    # def wdirl(self) -> Vector:
    #     if self._wdirl is None:
    #         self.calc_wpanel()
    #     return self._wdirl

    def unsig(self, mach: float = 0.0) -> Vector:
        if self._unsig is None:
            self._unsig = {}
        if mach not in self._unsig:
            unsig = Vector.zeros((self.num_dpanels, 2 + 4*self.num_controls))
            unsig[:, 0] = -self.normals
            unsig[:, 1] = self.rrel.cross(self.normals)
            if self.surfaces is not None:
                for surface in self.surfaces:
                    for sheet in surface.sheets:
                        for control in sheet.controls.values():
                            ctuple = self.controls[control.name]
                            for panel in control.panels:
                                ind = panel.ind
                                rrel = self.rrel[ind]
                                dndlp = panel.dndl(control.posgain, control.uhvec)
                                unsig[ind, ctuple[0]] = -dndlp
                                unsig[ind, ctuple[1]] = -rrel.cross(dndlp)
                                dndln = panel.dndl(control.neggain, control.uhvec)
                                unsig[ind, ctuple[2]] = -dndln
                                unsig[ind, ctuple[3]] = -rrel.cross(dndln)
            elif self.groups is not None:
                for group in self.groups.values():
                    for control in group.controls.values():
                        ctuple = self.controls[control.name]
                        for panel in control.panels:
                            ind = panel.ind
                            rrel = self.rrel[ind]
                            dndlp = panel.dndl(control.posgain, control.uhvec)
                            unsig[ind, ctuple[0]] = -dndlp
                            unsig[ind, ctuple[1]] = -rrel.cross(dndlp)
                            dndln = panel.dndl(control.neggain, control.uhvec)
                            unsig[ind, ctuple[2]] = -dndln
                            unsig[ind, ctuple[3]] = -rrel.cross(dndln)
            self._unsig[mach] = unsig
        return self._unsig[mach]

    def unmud(self, mach: float = 0.0) -> Vector:
        if self._unmud is None:
            self._unmud = {}
        if mach not in self._unmud:
            self.solve_dirichlet_system(mach = mach)
            # self.solve_neumann_system(mach = mach)
        return self._unmud[mach]

    def unmuw(self, mach: float = 0.0) -> Vector:
        if self._unmuw is None:
            self._unmuw = {}
        if mach not in self._unmuw:
            self.solve_dirichlet_system(mach = mach)
            # self.solve_neumann_system(mach = mach)
        return self._unmuw[mach]

    def unphi(self, mach: float = 0.0) -> Vector:
        if self._unphi is None:
            self._unphi = {}
        if mach not in self._unphi:
            self.solve_dirichlet_system(mach = mach)
            # self.solve_neumann_system(mach = mach)
        return self._unphi[mach]

    # def unnvg(self, mach: float = 0.0) -> Vector:
    #     if self._unphi is None:
    #         self._unphi = {}
    #     if mach not in self._unnvg:
    #         self.solve_dirichlet_system(mach = mach)
    #         # self.solve_neumann_system(mach = mach)
    #     return self._unnvg[mach]

    def assemble_panels_phi(self, *, mach: float = 0.0) -> None:

        betm = betm_from_mach(mach)

        if self._aphdd is None:
            self._aphdd = {}
        if self._aphsd is None:
            self._aphsd = {}

        aphdd = zeros((self.num_dpanels, self.num_dpanels))
        aphsd = zeros((self.num_dpanels, self.num_dpanels))

        for panel in self.dpanels.values():
            aphdp, aphsp = panel.constant_doublet_source_phi(self.points, betx = betm)
            aphdd[:, panel.ind] = aphdp
            aphsd[:, panel.ind] = aphsp

        self._aphdd[mach] = aphdd
        self._aphsd[mach] = aphsd

        if self._aphdw is None:
            self._aphdw = {}

        aphdw = zeros((self.num_dpanels, self.num_wpanels))

        for panel in self.wpanels.values():
            aphdp = panel.constant_doublet_phi(self.points, betx = betm)
            aphdw[:, panel.ind] += aphdp

        self._aphdw[mach] = aphdw

    def ainv(self, mach: float = 0.0) -> 'NDArray':
        if self._ainv is None:
            self._ainv = {}
        if mach not in self._ainv:
            self._ainv[mach] = inv(self.aphdd(mach))
        return self._ainv[mach]

    def bmat(self, mach: float = 0.0) -> 'NDArray':
        return self.aphdw(mach)

    @property
    def cmat(self) -> 'NDArray':
        if self._cmat is None:
            self._cmat = zeros((self.num_wpanels, self.num_dpanels))
            for wpanel in self.wpanels.values():
                dpanel = wpanel.adjpanels[0]
                self._cmat[wpanel.ind, dpanel.ind] = -1.0
                if len(wpanel.adjpanels) == 2:
                    dpanel = wpanel.adjpanels[1]
                    self._cmat[wpanel.ind, dpanel.ind] = 1.0
        return self._cmat

    @property
    def dfaces(self) -> list['Face']:
        if self._dfaces is None:
            self._dfaces = []
            for panel in self.dpanels.values():
                for face in panel.faces:
                    self._dfaces.append(face)
        return self._dfaces

    @property
    def num_dfaces(self) -> int:
        if self._num_dfaces is None:
            self._num_dfaces = len(self.dfaces)
        return self._num_dfaces

    def assemble_dfaces(self) -> None:
        self._dface_pnts = Vector.zeros(self.num_dfaces)
        self._dface_dirx = Vector.zeros(self.num_dfaces)
        self._dface_diry = Vector.zeros(self.num_dfaces)
        self._dface_dirz = Vector.zeros(self.num_dfaces)
        self._dface_area = zeros(self.num_dfaces)
        self._dface_indg = zeros((self.num_dfaces, 2), dtype=int)
        self._dface_velg = Vector2D.zeros((self.num_dfaces, 2))
        self._dface_indp = zeros(self.num_dfaces, dtype=int)
        self._dface_velp = Vector2D.zeros(self.num_dfaces)

        for i, face in enumerate(self.dfaces):
            self._dface_pnts[i] = face.cord.pnt
            self._dface_dirx[i] = face.cord.dirx
            self._dface_diry[i] = face.cord.diry
            self._dface_dirz[i] = face.cord.dirz
            self._dface_area[i] = face.area
            self._dface_indg[i, ...] = face.indg
            self._dface_velg[i, ...] = face.velg
            self._dface_indp[i] = face.indp
            self._dface_velp[i] = face.velp

    @property
    def dface_pnts(self) -> Vector:
        if self._dface_pnts is None:
            self.assemble_dfaces()
        return self._dface_pnts

    @property
    def dface_indg(self) -> 'NDArray':
        if self._dface_indg is None:
            self.assemble_dfaces()
        return self._dface_indg

    @property
    def dface_indp(self) -> 'NDArray':
        if self._dface_indp is None:
            self.assemble_dfaces()
        return self._dface_indp

    @property
    def dface_velg(self) -> Vector2D:
        if self._dface_velg is None:
            self.assemble_dfaces()
        return self._dface_velg

    @property
    def dface_velp(self) -> Vector2D:
        if self._dface_velp is None:
            self.assemble_dfaces()
        return self._dface_velp

    @property
    def dface_dirx(self) -> Vector:
        if self._dface_dirx is None:
            self.assemble_dfaces()
        return self._dface_dirx

    @property
    def dface_diry(self) -> Vector:
        if self._dface_diry is None:
            self.assemble_dfaces()
        return self._dface_diry

    @property
    def dface_dirz(self) -> Vector:
        if self._dface_dirz is None:
            self.assemble_dfaces()
        return self._dface_dirz

    @property
    def dface_area(self) -> 'NDArray':
        if self._dface_area is None:
            self.assemble_dfaces()
        return self._dface_area

    # def assemble_panels_vel(self, *, mach: float = 0.0) -> None:
    #     if self._apd is None:
    #         self._apd = {}
    #     if self._aps is None:
    #         self._aps = {}
    #     if self._avd is None:
    #         self._avd = {}
    #     if self._avs is None:
    #         self._avs = {}

    #     betm = betm_from_mach(mach)

    #     from .. import USE_CUPY

    #     if USE_CUPY:
    #         from ..tools.cupy import cupy_ctdsv as ctdsv
    #     else:
    #         from ..tools.numpy import numpy_ctdsv as ctdsv

    #     pnts = self.pnts.reshape((-1, 1))

    #     avdc = Vector.zeros((1, self.numpnl))
    #     avsc = Vector.zeros((1, self.numpnl))

    #     for pnl in self.pnls.values():
    #         if pnl.num == 3:
    #             grda = pnl.grds[0]
    #             grdb = pnl.grds[1]
    #             grdc = pnl.grds[2]
    #         else:
    #             grda = Vector.zeros((1, pnl.num))
    #             grdb = Vector.zeros((1, pnl.num))
    #             grdc = Vector.zeros((1, pnl.num))
    #             for i in range(-1, pnl.num-1):
    #                 grda[i+1] = pnl.grds[i]
    #                 grdb[i+1] = pnl.grds[i+1]
    #                 grdc[i+1] = pnl.pnto
    #         avdcp, avscp = ctdsv(pnts, grda, grdb, grdc,
    #                              betx=betm, cond=1.0)
    #         avdc[:, pnl.ind] += avdcp.sum(axis=1)
    #         avsc[:, pnl.ind] += avscp.sum(axis=1)

    #     # avdc, avsc = ctdsv(pnts, self.tgrida, self.tgridb, self.tgridc,
    #     #                    betx=betm, cond=1.0)

    #     # # apdc = add.reduceat(apdc, self.triarr, axis=1)
    #     # avdc = Vector(add.reduceat(avdc.x, self.triarr, axis=1),
    #     #               add.reduceat(avdc.y, self.triarr, axis=1),
    #     #               add.reduceat(avdc.z, self.triarr, axis=1))
    #     # # apsc = add.reduceat(apsc, self.triarr, axis=1)
    #     # avsc = Vector(add.reduceat(avsc.x, self.triarr, axis=1),
    #     #               add.reduceat(avsc.y, self.triarr, axis=1),
    #     #               add.reduceat(avsc.z, self.triarr, axis=1))

    #     self._avd[mach] = avdc
    #     self._avs[mach] = avsc

    # def assemble_horseshoes_phi(self, *, mach: float = 0.0) -> None:
    #     if self._aph is None:
    #         self._aph = {}

    #     betm = betm_from_mach(mach)

    #     from .. import USE_CUPY

    #     if USE_CUPY:
    #         from ..tools.cupy import cupy_cwdp as cwdp
    #     else:
    #         from ..tools.numpy import numpy_cwdp as cwdp

    #     pnts = self.pnts.reshape((-1, 1))

    #     aphc = cwdp(pnts, self.wgrida, self.wgridb, self.wdirl,
    #                 betx=betm)

    #     self._aph[mach] = aphc

    # def assemble_horseshoes_vel(self, *, mach: float = 0.0) -> None:
    #     if self._aph is None:
    #         self._aph = {}
    #     if self._avh is None:
    #         self._avh = {}

    #     betm = betm_from_mach(mach)

    #     from .. import USE_CUPY

    #     if USE_CUPY:
    #         from ..tools.cupy import cupy_cwdv as cwdv
    #     else:
    #         from ..tools.numpy import numpy_cwdv as cwdv

    #     pnts = self.pnts.reshape((-1, 1))

    #     avhc = cwdv(pnts, self.wgrida, self.wgridb, self.wdirl,
    #                 betx=betm)

    #     self._avh[mach] = avhc

    def solve_system(self, *, mach: float = 0.0) -> None:
        self.solve_dirichlet_system(mach = mach)
        # self.solve_neumann_system(mach = mach)

    def solve_dirichlet_system(self, mach: float = 0.0) -> None:

        if self._unmud is None:
            self._unmud = {}
        if self._unmuw is None:
            self._unmuw = {}

        Ai = self.ainv(mach)
        Bm = self.bmat(mach)
        Cm = self.cmat
        Dm = eye(self.num_wpanels)
        Em = self.bphs(mach)
        Fm = Vector.zeros((self.num_wpanels, 2 + 4*self.num_controls))

        Km = Cm@Ai

        Gm = Dm - Km@Bm
        Gi = inv(Gm)

        Lm = Bm@Gi

        Hm = Km@Em# - Fm
        Im = Ai@Em + Ai@Lm@Hm
        Jm = Gi@Fm - Gi@Km@Em

        self._unmud[mach] = Im
        self._unmuw[mach] = Jm

        if self._unphi is None:
            self._unphi = {}
        self._unphi[mach] = self.aphdd(mach)@self.unmud(mach)
        self._unphi[mach] += self.aphdw(mach)@self.unmuw(mach)
        self._unphi[mach] -= self.bphs(mach)

    # def solve_neumann_system(self, mach: float = 0.0) -> None:
    #     if self._unmu is None:
    #         self._unmu = {}
    #     self._unmu[mach] = self.bnm(mach).solve(self.anm(mach))
    #     if self._unnvg is None:
    #         self._unnvg = {}
    #     self._unnvg[mach] = self.anm(mach)@self.unmu(mach) - self.bnm(mach)
    #     if self._unphi is None:
    #         self._unphi = {}
    #     self._unphi[mach] = Vector.zeros(self._unnvg[mach].shape)

    # def plot_twist_distribution(self, ax: 'Axes'=None, axis: str='b',
    #                             surfaces: list['PanelSurface']=None) -> 'Axes':
    #     if self.srfcs is not None:
    #         if ax is None:
    #             fig = figure(figsize=(12, 8))
    #             ax = fig.gca()
    #             ax.grid(True)
    #         if surfaces is None:
    #             srfcs = [srfc for srfc in self.srfcs]
    #         else:
    #             srfcs = []
    #             for srfc in self.srfcs:
    #                 if srfc.name in surfaces:
    #                     srfcs.append(srfc)
    #         for srfc in srfcs:
    #             t = [prf.twist for prf in srfc.prfs]
    #             label = srfc.name
    #             if axis == 'b':
    #                 b = srfc.prfb
    #                 if max(b) > min(b):
    #                     ax.plot(b, t, label=label)
    #             elif axis == 'y':
    #                 y = srfc.prfy
    #                 if max(y) > min(y):
    #                     ax.plot(y, t, label=label)
    #             elif axis == 'z':
    #                 z = srfc.prfz
    #                 if max(z) > min(z):
    #                     ax.plot(z, t, label=label)
    #         ax.legend()
    #     return ax

    # def plot_chord_distribution(self, ax: 'Axes'=None, axis: str='b', surfaces: list=None):
    #     if self.srfcs is not None:
    #         if ax is None:
    #             fig = figure(figsize=(12, 8))
    #             ax = fig.gca()
    #             ax.grid(True)
    #         if surfaces is None:
    #             srfcs = [srfc for srfc in self.srfcs]
    #         else:
    #             srfcs = []
    #             for srfc in self.srfcs:
    #                 if srfc.name in surfaces:
    #                     srfcs.append(srfc)
    #         for srfc in srfcs:
    #             c = [prf.chord for prf in srfc.prfs]
    #             label = srfc.name
    #             if axis == 'b':
    #                 b = srfc.prfb
    #                 if max(b) > min(b):
    #                     ax.plot(b, c, label=label)
    #             elif axis == 'y':
    #                 y = srfc.prfy
    #                 if max(y) > min(y):
    #                     ax.plot(y, c, label=label)
    #             elif axis == 'z':
    #                 z = srfc.strpz
    #                 if max(z) > min(z):
    #                     ax.plot(z, c, label=label)
    #         ax.legend()
    #     return ax

    # def plot_tilt_distribution(self, ax: 'Axes'=None, axis: str='b', surfaces: list=None):
    #     if self.srfcs is not None:
    #         if ax is None:
    #             fig = figure(figsize=(12, 8))
    #             ax = fig.gca()
    #             ax.grid(True)
    #         if surfaces is None:
    #             srfcs = [srfc for srfc in self.srfcs]
    #         else:
    #             srfcs = []
    #             for srfc in self.srfcs:
    #                 if srfc.name in surfaces:
    #                     srfcs.append(srfc)
    #         for srfc in srfcs:
    #             t = [prf.tilt for prf in srfc.prfs]
    #             label = srfc.name
    #             if axis == 'b':
    #                 b = srfc.prfb
    #                 if max(b) > min(b):
    #                     ax.plot(b, t, label=label)
    #             elif axis == 'y':
    #                 y = srfc.prfy
    #                 if max(y) > min(y):
    #                     ax.plot(y, t, label=label)
    #             elif axis == 'z':
    #                 z = srfc.strpz
    #                 if max(z) > min(z):
    #                     ax.plot(z, t, label=label)
    #         ax.legend()
    #     return ax

    # def plot_strip_width_distribution(self, ax: 'Axes'=None, axis: str='b', surfaces: list=None):
    #     if self.srfcs is not None:
    #         if ax is None:
    #             fig = figure(figsize=(12, 8))
    #             ax = fig.gca()
    #             ax.grid(True)
    #         if surfaces is None:
    #             srfcs = [srfc for srfc in self.srfcs]
    #         else:
    #             srfcs = []
    #             for srfc in self.srfcs:
    #                 if srfc.name in surfaces:
    #                     srfcs.append(srfc)
    #         for srfc in srfcs:
    #             w = [strp.width for strp in srfc.strps]
    #             label = srfc.name
    #             if axis == 'b':
    #                 b = srfc.strpb
    #                 if max(b) > min(b):
    #                     ax.plot(b, w, label=label)
    #             elif axis == 'y':
    #                 y = srfc.strpy
    #                 if max(y) > min(y):
    #                     ax.plot(y, w, label=label)
    #             elif axis == 'z':
    #                 z = srfc.strpz
    #                 if max(z) > min(z):
    #                     ax.plot(z, w, label=label)
    #         ax.legend()
    #     return ax

    def mesh(self) -> None:
        if self.surfaces is not None:
            gid, pid = 1, 1
            for surface in self.surfaces:
                gid = surface.mesh_grids(gid)
                pid = surface.mesh_panels(pid)
            self.grids = {}
            self.dpanels = {}
            self.wpanels = {}
            for surface in self.surfaces:
                for grid in surface.grids:
                    self.grids[grid.gid] = grid
                for panel in surface.dpanels:
                    self.dpanels[panel.pid] = panel
                for panel in surface.wpanels:
                    self.wpanels[panel.pid] = panel
            ind = 2
            for surface in self.surfaces:
                for sheet in surface.sheets:
                    for control in sheet.controls.values():
                        if control.name not in self.controls:
                            self.controls[control.name] = (ind, ind+1, ind+2, ind+3)
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
        griddata: dict[str, Any] = sysdct['grids']
        for gidstr, gd in griddata.items():
            gid = int(gidstr)
            if 'te' not in gd:
                gd['te'] = False
            grds[gid] = Grid(gid, gd['x'], gd['y'], gd['z'])

        grps: dict[int, PanelGroup] = {}
        groupdata: dict[int, dict[str, Any]] = sysdct.get('groups', {})
        for grpidstr, grpdata in groupdata.items():
            grpid = int(grpidstr)
            grpname = grpdata.get('name', f'Group {grpid}')
            grps[grpid] = PanelGroup(grpname)
            grps[grpid].exclude = grpdata.get('exclude', False)
            grps[grpid].noload = grpdata.get('noload', False)
            grps[grpid].nohsv = grpdata.get('nohsv', False)

        pnls: dict[int, Panel] = {}
        paneldata: dict[int, dict[str, Any]] = sysdct['panels']
        for pidstr, pd in paneldata.items():
            pid = int(pidstr)
            if 'grpid' in pd:
                grpid = pd['grpid']
                grp = grps[grpid]
                if not grp.exclude:
                    pnlgrds = [grds[gidi] for gidi in pd['gids']]
                    pnl = Panel(pid, pnlgrds)
                    pnls[pid] = pnl
                    if grp.noload:
                        pnls[pid].noload = True
                    pnls[pid].grp = grp
                grp.add_panel(pnl)
            else:
                pnlgrds = [grds[gidi] for gidi in pd['gids']]
                pnls[pid] = Panel(pid, pnlgrds)

        sys = cls(name, bref, cref, sref, rref)
        sys.set_mesh(grds, pnls)
        sys.groups = grps

        ind = 2
        ctrldata: dict[str, dict[str, Any]] = sysdct.get('controls', {})
        for desc, ctrldct in ctrldata.items():
            ctrlname = ctrldct.get('name', desc)
            posgain = ctrldct.get('posgain', 1.0)
            neggain = ctrldct.get('neggain', 1.0)
            hvec = Vector.from_dict(ctrldct.get('hvec', {'x': 0.0, 'y': 0.0, 'z': 0.0}))
            ctrl = PanelControl(ctrlname, posgain, neggain)
            ctrl.desc = desc
            ctrl.set_hinge_vector(hvec)
            grpid = ctrldct.get('group', None)
            if grpid is not None:
                grpid = int(grpid)
                grp = sys.grps[grpid]
                grp.add_control(ctrl)
            if ctrlname not in sys.ctrls:
                sys.ctrls[ctrlname] = (ind, ind+1, ind+2, ind+3)
                ind += 4

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
            sys.mass = MassObject(sys.name, mass = mass,
                                  xcm = sys.rref.x,
                                  ycm = sys.rref.y,
                                  zcm = sys.rref.z)
        elif isinstance(mass, str):
            sys.mass = masses[mass]
        else:
            sys.mass = MassObject(sys.name, mass = 1.0,
                                  xcm = sys.rref.x,
                                  ycm = sys.rref.y,
                                  zcm = sys.rref.z)

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
                for control in self.controls:
                    value = result.controls[control]
                    result.controls[control] = resdata.get(control, value)

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
        if self.grids is not None:
            table.add_column('# Grids', 'd', data=[self.num_grids])
        else:
            table.add_column('# Grids', 'd', data=[0])
        if self.dpanels is not None:
            table.add_column('# Dirichlet Panels', 'd', data=[self.num_dpanels])
        else:
            table.add_column('# Dirichlet Panels', 'd', data=[0])
        if self.wpanels is not None:
            table.add_column('# Wake Panels', 'd', data=[self.num_wpanels])
        else:
            table.add_column('# Wake Panels', 'd', data=[0])
        if self.controls is not None:
            table.add_column('# Controls', 'd', data=[self.num_controls])
        else:
            table.add_column('# Controls', 'd', data=[0])
        if len(table.columns) > 0:
            outstr += table._repr_markdown_()
        return outstr

    def _repr_markdown_(self) -> str:
        return self.__str__()
