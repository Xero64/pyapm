from json import dump, load
from os.path import exists
from typing import TYPE_CHECKING, Any

from numpy import add, eye, zeros
from numpy.linalg import inv
from py2md.classes import MDReport
from pygeom.geom3d import Vector

from .constantcontrol import ConstantControl
from .constantgeometry import ConstantGeometry
from .constantgrid import ConstantGrid
from .constantpanel import ConstantPanel
from .constantwakepanel import ConstantWakePanel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...tools.mass import MassObject
    from .constantresult import ConstantResult

DISPLAY = False


class ConstantSystem():
    name: str
    geometry: ConstantGeometry
    dpanels: list[ConstantPanel]
    npanels: list[ConstantPanel]
    wpanels: list[ConstantWakePanel]
    ctrls: dict[str, ConstantControl]
    rref: Vector
    bref: float
    cref: float
    sref: float
    CDo: float
    mass: 'MassObject'
    _ar: float
    _grids: list[ConstantGrid]
    _num_grids: int
    _dpoints: Vector
    _npoints: Vector
    _dnormal: Vector
    _nnormal: Vector
    _drel: Vector
    _nrel: Vector
    _avnd: Vector
    _avns: Vector
    _avnn: Vector
    _avnw: Vector
    _avdd: Vector
    _avds: Vector
    _avdn: Vector
    _avdw: Vector
    _amdd: 'NDArray'
    _amds: 'NDArray'
    _amdn: 'NDArray'
    _amdw: 'NDArray'
    _amnd: 'NDArray'
    _amns: 'NDArray'
    _amnn: 'NDArray'
    _amnw: 'NDArray'
    _amwd: 'NDArray'
    _amwn: 'NDArray'
    _amww: 'NDArray'
    _amat: 'NDArray'
    _bmat: 'NDArray'
    _cmat: 'NDArray'
    _dmat: 'NDArray'
    _unsig: Vector
    _evecd: Vector
    _evecn: Vector
    _evec: Vector
    _fvec: Vector
    _ainv: 'NDArray'
    _kmat: 'NDArray'
    _hvec: Vector
    _unmud: Vector
    _unmun: Vector
    _unmuw: Vector
    _gridvec: Vector
    _gridarea: 'NDArray'
    _avgd: Vector
    _avgs: Vector
    _avgn: Vector
    _avgw: Vector
    _blgd: Vector
    _blgn: Vector
    _blgw: Vector
    _results: dict[str, 'ConstantResult']
    _dtriarr: 'NDArray'
    _dgrida: Vector
    _dgridb: Vector
    _dgridc: Vector
    _ntriarr: 'NDArray'
    _ngrida: Vector
    _ngridb: Vector
    _ngridc: Vector
    _wgrida: Vector
    _wgridb: Vector
    _wdirl: Vector

    __slots__ = tuple(__annotations__)

    def __init__(self, name: str, dpanels: list[ConstantPanel],
                 npanels: list[ConstantPanel], wpanels: list[ConstantWakePanel]) -> None:
        self.name = name
        self.geometry = ConstantGeometry()
        self.dpanels = dpanels
        self.npanels = npanels
        self.wpanels = wpanels
        self.ctrls = {}
        self.rref = Vector(0.0, 0.0, 0.0)
        self.bref = 1.0
        self.cref = 1.0
        self.sref = 1.0
        self.CDo = 0.0
        self.reset()
        self.update_indices()

    def reset(self) -> None:
        for attr in self.__slots__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def update_indices(self) -> None:
        self.reset()
        indo = 0
        for panel in self.dpanels:
            panel.indo = indo
            indo += 1
        for panel in self.npanels:
            panel.indo = indo
            indo += 1
        indo = 0
        for panel in self.wpanels:
            panel.indo = indo
            indo += 1

    def update_ctrl_indices(self) -> None:
        for i, ctrl in enumerate(self.ctrls.values()):
            ctrl.index = (4*i+2, 4*i+3, 4*i+4, 4*i+5)

    @property
    def ar(self) -> float:
        if self._ar is None:
            self._ar = self.bref**2/self.sref
        return self._ar

    @ar.setter
    def ar(self, ar: float) -> None:
        self._ar = ar

    @property
    def num_ctrls(self) -> int:
        return len(self.ctrls)

    @property
    def num_dpanels(self) -> int:
        return len(self.dpanels)

    @property
    def num_npanels(self) -> int:
        return len(self.npanels)

    @property
    def num_wpanels(self) -> int:
        return len(self.wpanels)

    @property
    def num_panels(self) -> int:
        return self.num_dpanels + self.num_npanels

    @property
    def results(self) -> dict[str, 'ConstantResult']:
        if self._results is None:
            self._results = {}
        return self._results

    @property
    def grids(self) -> list[ConstantGrid]:
        if self._grids is None:
            grids = set()
            for panel in self.dpanels:
                for grid in panel.grids:
                    grids.add(grid)
            for panel in self.npanels:
                for grid in panel.grids:
                    grids.add(grid)
            for panel in self.wpanels:
                for grid in panel.grids:
                    grids.add(grid)
            self._grids = list(grids)
            for ind, grid in enumerate(self._grids):
                grid.ind = ind
        return self._grids

    @property
    def num_grids(self) -> int:
        if self._num_grids is None:
            self._num_grids = len(self.grids)
        return self._num_grids

    @property
    def dpoints(self) -> Vector:
        if self._dpoints is None:
            self._dpoints = Vector.zeros(self.num_dpanels)
            for i, panel in enumerate(self.dpanels):
                self._dpoints[i] = panel.point
        return self._dpoints

    @property
    def npoints(self) -> Vector:
        if self._npoints is None:
            self._npoints = Vector.zeros(self.num_npanels)
            for i, panel in enumerate(self.npanels):
                self._npoints[i] = panel.point
        return self._npoints

    @property
    def dnormal(self) -> Vector:
        if self._dnormal is None:
            self._dnormal = Vector.zeros(self.num_dpanels)
            for i, panel in enumerate(self.dpanels):
                self._dnormal[i] = panel.normal
        return self._dnormal

    @property
    def nnormal(self) -> Vector:
        if self._nnormal is None:
            self._nnormal = Vector.zeros(self.num_npanels)
            for i, panel in enumerate(self.npanels):
                self._nnormal[i] = panel.normal
        return self._nnormal

    @property
    def drel(self) -> Vector:
        if self._drel is None:
            self._drel = self.dpoints - self.rref
        return self._drel

    @property
    def nrel(self) -> Vector:
        if self._nrel is None:
            self._nrel = self.npoints - self.rref
        return self._nrel

    def calc_dpanel(self) -> 'NDArray':
        numtria = 0
        for panel in self.dpanels:
            numtria += panel.num
        self._dtriarr = zeros(self.num_dpanels, dtype=int)
        self._dgrida = Vector.zeros((1, numtria))
        self._dgridb = Vector.zeros((1, numtria))
        self._dgridc = Vector.zeros((1, numtria))
        k = 0
        for i, panel in enumerate(self.dpanels):
            self._dtriarr[i] = k
            for i in range(-1, panel.num-1):
                a, b = i, i + 1
                self._dgrida[0, k] = panel.grids[a]
                self._dgridb[0, k] = panel.grids[b]
                self._dgridc[0, k] = panel.point
                k += 1

    @property
    def dtriarr(self) -> 'NDArray':
        if self._dtriarr is None:
            self.calc_dpanel()
        return self._dtriarr

    @property
    def dgrida(self) -> Vector:
        if self._dgrida is None:
            self.calc_dpanel()
        return self._dgrida

    @property
    def dgridb(self) -> Vector:
        if self._dgridb is None:
            self.calc_dpanel()
        return self._dgridb

    @property
    def dgridc(self) -> Vector:
        if self._dgridc is None:
            self.calc_dpanel()
        return self._dgridc

    def calc_npanel(self) -> 'NDArray':
        numtria = 0
        for panel in self.npanels:
            numtria += panel.num
        self._ntriarr = zeros(self.num_npanels, dtype=int)
        self._ngrida = Vector.zeros((1, numtria))
        self._ngridb = Vector.zeros((1, numtria))
        self._ngridc = Vector.zeros((1, numtria))
        k = 0
        for i, panel in enumerate(self.npanels):
            self._ntriarr[i] = k
            for i in range(-1, panel.num-1):
                a, b = i, i + 1
                self._ngrida[0, k] = panel.grids[a]
                self._ngridb[0, k] = panel.grids[b]
                self._ngridc[0, k] = panel.point
                k += 1

    @property
    def ntriarr(self) -> 'NDArray':
        if self._ntriarr is None:
            self.calc_npanel()
        return self._ntriarr

    @property
    def ngrida(self) -> Vector:
        if self._ngrida is None:
            self.calc_npanel()
        return self._ngrida

    @property
    def ngridb(self) -> Vector:
        if self._ngridb is None:
            self.calc_npanel()
        return self._ngridb

    @property
    def ngridc(self) -> Vector:
        if self._ngridc is None:
            self.calc_npanel()
        return self._ngridc

    def calc_wpanel(self) -> 'NDArray':
        self._wgrida = Vector.zeros((1, self.num_wpanels))
        self._wgridb = Vector.zeros((1, self.num_wpanels))
        self._wdirl = Vector.zeros((1, self.num_wpanels))
        for panel in self.wpanels:
            self._wgrida[0, panel.indo] = panel.grida
            self._wgridb[0, panel.indo] = panel.gridb
            self._wdirl[0, panel.indo] = panel.dirl

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

    def calc_amdd_and_amds(self):

        from . import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdsp as ctdsp
        else:
            from pyapm.tools.numpy import numpy_ctdsp as ctdsp

        dpoints = self.dpoints.reshape((self.num_dpanels, 1))

        amddt, amdst = ctdsp(dpoints, self.dgrida, self.dgridb, self.dgridc,
                             cond=-1.0)
        self._amdd = add.reduceat(amddt, self.dtriarr, axis=1)
        self._amds = add.reduceat(amdst, self.dtriarr, axis=1)

    @property
    def amdd(self) -> 'NDArray':
        if self._amdd is None:
            self.calc_amdd_and_amds()
        return self._amdd

    @property
    def amds(self) -> 'NDArray':
        if self._amds is None:
            self.calc_amdd_and_amds()
        return self._amds

    @property
    def amdn(self) -> 'NDArray':
        if self._amdn is None:

            from . import USE_CUPY

            if USE_CUPY:
                from pyapm.tools.cupy import cupy_ctdp as ctdp
            else:
                from pyapm.tools.numpy import numpy_ctdp as ctdp

            dpoints = self.dpoints.reshape((self.num_dpanels, 1))

            amdnt = ctdp(dpoints, self.ngrida, self.ngridb, self.ngridc)
            self._amdn = add.reduceat(amdnt, self.ntriarr, axis=1)

        return self._amdn

    @property
    def amdw(self) -> 'NDArray':
        if self._amdw is None:

            from . import USE_CUPY

            if USE_CUPY:
                from pyapm.tools.cupy import cupy_cwdp as cwdp
            else:
                from pyapm.tools.numpy import numpy_cwdp as cwdp

            dpoints = self.dpoints.reshape((self.num_dpanels, 1))

            self._amdw = cwdp(dpoints, self.wgrida, self.wgridb, self.wdirl)

        return self._amdw

    def calc_avnd_and_avns(self):

        from . import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdsv as ctdsv
        else:
            from pyapm.tools.numpy import numpy_ctdsv as ctdsv

        npoints = self.npoints.reshape((self.num_npanels, 1))

        avndt, avnst = ctdsv(npoints, self.dgrida, self.dgridb, self.dgridc,
                             cond=-1.0)

        self._avnd = Vector(add.reduceat(avndt.x, self.dtriarr, axis=1),
                            add.reduceat(avndt.y, self.dtriarr, axis=1),
                            add.reduceat(avndt.z, self.dtriarr, axis=1))
        self._avns = Vector(add.reduceat(avnst.x, self.dtriarr, axis=1),
                            add.reduceat(avnst.y, self.dtriarr, axis=1),
                            add.reduceat(avnst.z, self.dtriarr, axis=1))

    @property
    def avnd(self) -> Vector:
        if self._avnd is None:
            self.calc_avnd_and_avns()
        return self._avnd

    @property
    def avns(self) -> Vector:
        if self._avns is None:
            self.calc_avnd_and_avns()
        return self._avns

    @property
    def avnn(self) -> Vector:
        if self._avnn is None:

            from . import USE_CUPY

            if USE_CUPY:
                from pyapm.tools.cupy import cupy_ctdv as ctdv
            else:
                from pyapm.tools.numpy import numpy_ctdv as ctdv

            npoints = self.npoints.reshape((self.num_npanels, 1))

            avnnt = ctdv(npoints, self.ngrida, self.ngridb, self.ngridc)
            self._avnn = Vector(add.reduceat(avnnt.x, self.ntriarr, axis=1),
                                add.reduceat(avnnt.y, self.ntriarr, axis=1),
                                add.reduceat(avnnt.z, self.ntriarr, axis=1))

        return self._avnn

    @property
    def avnw(self) -> Vector:
        if self._avnw is None:

            from . import USE_CUPY

            if USE_CUPY:
                from pyapm.tools.cupy import cupy_cwdv as cwdv
            else:
                from pyapm.tools.numpy import numpy_cwdv as cwdv

            npoints = self.npoints.reshape((self.num_npanels, 1))

            self._avnw = cwdv(npoints, self.wgrida, self.wgridb, self.wdirl)

        return self._avnw

    def calc_avdd_and_avds(self):

        from . import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdsv as ctdsv
        else:
            from pyapm.tools.numpy import numpy_ctdsv as ctdsv

        dpoints = self.dpoints.reshape((self.num_dpanels, 1))

        avddt, avdst = ctdsv(dpoints, self.dgrida, self.dgridb, self.dgridc,
                             cond=-1.0)
        self._avdd = Vector(add.reduceat(avddt.x, self.dtriarr, axis=1),
                            add.reduceat(avddt.y, self.dtriarr, axis=1),
                            add.reduceat(avddt.z, self.dtriarr, axis=1))
        self._avds = Vector(add.reduceat(avdst.x, self.dtriarr, axis=1),
                            add.reduceat(avdst.y, self.dtriarr, axis=1),
                            add.reduceat(avdst.z, self.dtriarr, axis=1))

    @property
    def avdd(self) -> Vector:
        if self._avdd is None:
            self.calc_avdd_and_avds()
        return self._avdd

    @property
    def avds(self) -> Vector:
        if self._avds is None:
            self.calc_avdd_and_avds()
        return self._avds

    @property
    def avdn(self) -> Vector:
        if self._avdn is None:

            from . import USE_CUPY

            if USE_CUPY:
                from pyapm.tools.cupy import cupy_ctdv as ctdv
            else:
                from pyapm.tools.numpy import numpy_ctdv as ctdv

            dpoints = self.dpoints.reshape((self.num_dpanels, 1))

            avdnt = ctdv(dpoints, self.ngrida, self.ngridb, self.ngridc)
            self._avdn = Vector(add.reduceat(avdnt.x, self.ntriarr, axis=1),
                                add.reduceat(avdnt.y, self.ntriarr, axis=1),
                                add.reduceat(avdnt.z, self.ntriarr, axis=1))

        return self._avdn

    @property
    def avdw(self) -> Vector:
        if self._avdw is None:

            from . import USE_CUPY

            if USE_CUPY:
                from pyapm.tools.cupy import cupy_cwdv as cwdv
            else:
                from pyapm.tools.numpy import numpy_cwdv as cwdv

            dpoints = self.dpoints.reshape((self.num_dpanels, 1))

            self._avdw = cwdv(dpoints, self.wgrida, self.wgridb, self.wdirl)

        return self._avdw

    @property
    def amnd(self) -> 'NDArray':
        if self._amnd is None:
            self._amnd = zeros((self.num_npanels, self.num_dpanels))
            for i in range(self.num_dpanels):
                self._amnd[:, i] = self.avnd[:, i].dot(self.nnormal)
        return self._amnd

    @property
    def amns(self) -> 'NDArray':
        if self._amns is None:
            self._amns = zeros((self.num_npanels, self.num_dpanels))
            for i in range(self.num_dpanels):
                self._amns[:, i] = self.avns[:, i].dot(self.nnormal)
        return self._amns

    @property
    def amnn(self) -> 'NDArray':
        if self._amnn is None:
            self._amnn = zeros((self.num_npanels, self.num_npanels))
            for i in range(self.num_npanels):
                self._amnn[:, i] = self.avnn[:, i].dot(self.nnormal)
        return self._amnn

    @property
    def amnw(self) -> 'NDArray':
        if self._amnw is None:
            self._amnw = zeros((self.num_npanels, self.num_wpanels))
            for i in range(self.num_wpanels):
                self._amnw[:, i] = self.avnw[:, i].dot(self.nnormal)
        return self._amnw

    @property
    def amat(self) -> 'NDArray':
        if self._amat is None:
            self._amat = zeros((self.num_panels, self.num_panels))
            self._amat[:self.num_dpanels, :self.num_dpanels] = self.amdd
            self._amat[:self.num_dpanels, self.num_dpanels:] = self.amdn
            self._amat[self.num_dpanels:, :self.num_dpanels] = self.amnd
            self._amat[self.num_dpanels:, self.num_dpanels:] = self.amnn
        return self._amat

    @property
    def bmat(self) -> 'NDArray':
        if self._bmat is None:
            self._bmat = zeros((self.num_panels, self.num_wpanels))
            self._bmat[:self.num_dpanels, :] = self.amdw
            self._bmat[self.num_dpanels:, :] = self.amnw
        return self._bmat

    @property
    def cmat(self) -> 'NDArray':
        if self._cmat is None:
            self._cmat = zeros((self.num_wpanels, self.num_panels))
            for i, panel in enumerate(self.wpanels):
                for adjpanel, adjbool in panel.panels.items():
                    if adjbool:
                        self._cmat[i, adjpanel.indo] = 1.0
                    else:
                        self._cmat[i, adjpanel.indo] = -1.0
        return self._cmat

    @property
    def dmat(self) -> 'NDArray':
        if self._dmat is None:
            self._dmat = eye(self.num_wpanels)
        return self._dmat

    @property
    def unsig(self) -> Vector:
        if self._unsig is None:
            self.update_ctrl_indices()
            self._unsig = Vector.zeros((self.num_dpanels, 2 + 4*self.num_ctrls))
            self._unsig[:, 0] = -self.dnormal
            self._unsig[:, 1] = -self.dnormal.cross(self.drel)
            for ctrl in self.ctrls.values():
                ind1, ind2, ind3, ind4 = ctrl.index
                pos_nrml = ctrl.normal_change_approx[:self.num_dpanels, 0]
                neg_nrml = ctrl.normal_change_approx[:self.num_dpanels, 1]
                self._unsig[:, ind1] = -pos_nrml
                self._unsig[:, ind2] = -pos_nrml.cross(self.drel)
                self._unsig[:, ind3] = -neg_nrml
                self._unsig[:, ind4] = -neg_nrml.cross(self.drel)
        return self._unsig

    @property
    def evecd(self) -> Vector:
        if self._evecd is None:
            self._evecd = -self.amds@self.unsig
        return self._evecd

    @property
    def evecn(self) -> Vector:
        if self._evecn is None:
            self.update_ctrl_indices()
            self._evecn = Vector.zeros((self.num_npanels, 2 + 4*self.num_ctrls))
            self._evecn[:, 0] = -self.nnormal
            self._evecn[:, 1] = -self.nnormal.cross(self.nrel)
            for ctrl in self.ctrls.values():
                ind1, ind2, ind3, ind4 = ctrl.index
                pos_nrml = ctrl.normal_change_approx[self.num_dpanels:, 0]
                neg_nrml = ctrl.normal_change_approx[self.num_dpanels:, 1]
                self._evecn[:, ind1] = -pos_nrml
                self._evecn[:, ind2] = -pos_nrml.cross(self.nrel)
                self._evecn[:, ind3] = -neg_nrml
                self._evecn[:, ind4] = -neg_nrml.cross(self.nrel)
            self._evecn += self.amns@self.unsig
        return self._evecn

    @property
    def evec(self) -> Vector:
        if self._evec is None:
            self._evec = Vector.zeros((self.num_panels, 2 + 4*self.num_ctrls))
            self._evec[:self.num_dpanels, :] = self.evecd
            self._evec[self.num_dpanels:, :] = self.evecn
        return self._evec

    @property
    def fvec(self) -> Vector:
        if self._fvec is None:
            self._fvec = Vector.zeros((self.num_wpanels, 2 + 4*self.num_ctrls))
        return self._fvec

    @property
    def ainv(self) -> 'NDArray':
        if self._ainv is None:
            self._ainv = inv(self.amat)
        return self._ainv

    @property
    def kmat(self) -> 'NDArray':
        if self._kmat is None:
            self._kmat = self.cmat@self.ainv
        return self._kmat

    @property
    def hvec(self) -> Vector:
        if self._hvec is None:
            self._hvec = self.kmat@self.evec# - self.fvec # fvec is zero
        return self._hvec

    def solve_system(self, bmat: 'NDArray | None' = None,
                     evec: 'Vector | None' = None) -> None:

        if bmat is None:
            bmat = self.bmat

        if evec is None:
            evec = self.evec
            hvec = self.hvec
        else:
            hvec = self.kmat@evec# - self.fvec # fvec is zero

        gmat = self.dmat - self.kmat@bmat
        ginv = inv(gmat)

        lmat = self.bmat@ginv

        ivec = self.ainv@evec + self.ainv@lmat@hvec
        jvec = -ginv@self.kmat@evec# + ginv@self.fvec # fvec is zero

        self._unmud = ivec[:self.num_dpanels, :]
        self._unmun = ivec[self.num_dpanels:, :]
        self._unmuw = jvec

    @property
    def unmud(self) -> Vector:
        if self._unmud is None:
            self.solve_system()
        return self._unmud

    @property
    def unmun(self) -> Vector:
        if self._unmun is None:
            self.solve_system()
        return self._unmun

    @property
    def unmuw(self) -> Vector:
        if self._unmuw is None:
            self.solve_system()
        return self._unmuw

    @property
    def gridvec(self) -> Vector:
        if self._gridvec is None:
            self._gridvec = Vector.zeros(self.num_grids)
            for i, grid in enumerate(self.grids):
                self._gridvec[i] = grid
        return self._gridvec

    @property
    def gridarea(self) -> 'NDArray':
        if self._gridarea is None:
            self._gridarea = zeros(self.num_grids)
            for dpanel in self.dpanels:
                self._gridarea[dpanel.grid_index, ...] += dpanel.area/dpanel.num
            for npanel in self.npanels:
                self._gridarea[npanel.grid_index, ...] += npanel.area/npanel.num
        return self._gridarea

    def calc_avgd_and_avgs(self):

        from . import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdsv as ctdsv
        else:
            from pyapm.tools.numpy import numpy_ctdsv as ctdsv

        gridvec = self.gridvec.reshape((self.num_grids, 1))

        avgdt, avgst = ctdsv(gridvec, self.dgrida, self.dgridb, self.dgridc,
                             cond=-1.0)
        self._avgd = Vector(add.reduceat(avgdt.x, self.dtriarr, axis=1),
                            add.reduceat(avgdt.y, self.dtriarr, axis=1),
                            add.reduceat(avgdt.z, self.dtriarr, axis=1))
        self._avgs = Vector(add.reduceat(avgst.x, self.dtriarr, axis=1),
                            add.reduceat(avgst.y, self.dtriarr, axis=1),
                            add.reduceat(avgst.z, self.dtriarr, axis=1))

    @property
    def avgd(self) -> Vector:
        if self._avgd is None:
            self.calc_avgd_and_avgs()
        return self._avgd

    @property
    def avgs(self) -> Vector:
        if self._avgs is None:
            self.calc_avgd_and_avgs()
        return self._avgs

    @property
    def avgn(self) -> Vector:
        if self._avgn is None:

            from . import USE_CUPY

            if USE_CUPY:
                from pyapm.tools.cupy import cupy_ctdv as ctdv
            else:
                from pyapm.tools.numpy import numpy_ctdv as ctdv

            gridvec = self.gridvec.reshape((self.num_grids, 1))

            avgnt = ctdv(gridvec, self.ngrida, self.ngridb, self.ngridc)
            self._avgn = Vector(add.reduceat(avgnt.x, self.ntriarr, axis=1),
                                add.reduceat(avgnt.y, self.ntriarr, axis=1),
                                add.reduceat(avgnt.z, self.ntriarr, axis=1))

        return self._avgn

    @property
    def avgw(self) -> Vector:
        if self._avgw is None:

            from . import USE_CUPY

            if USE_CUPY:
                from pyapm.tools.cupy import cupy_cwdv as cwdv
            else:
                from pyapm.tools.numpy import numpy_cwdv as cwdv

            gridvec = self.gridvec.reshape((self.num_grids, 1))

            self._avgw = cwdv(gridvec, self.wgrida, self.wgridb, self.wdirl)

        return self._avgw

    @property
    def blgd(self) -> Vector:
        if self._blgd is None:
            self._blgd = Vector.zeros((self.num_grids, self.num_dpanels))
            for i, panel in enumerate(self.dpanels):
                self._blgd[panel.grid_index, i] += panel.grid_force
        return self._blgd

    @property
    def blgn(self) -> Vector:
        if self._blgn is None:
            self._blgn = Vector.zeros((self.num_grids, self.num_npanels))
            for i, panel in enumerate(self.npanels):
                self._blgn[panel.grid_index, i] += panel.grid_force
        return self._blgn

    @property
    def blgw(self) -> Vector:
        if self._blgw is None:
            self._blgw = Vector.zeros((self.num_grids, self.num_wpanels))
            for i, panel in enumerate(self.wpanels):
                self._blgw[panel.grid_index, i] += panel.grid_force
        return self._blgw

    def to_mdobj(self) -> MDReport:

        from . import get_unit_string

        lstr = get_unit_string('length')
        Astr = get_unit_string('area')

        report = MDReport()
        report.add_heading(f'Constant System {self.name}', 1)

        table = report.add_table()
        table.add_column('Name', 's', data=[self.name])
        table.add_column(f'S<sub>ref</sub>{Astr:s}', 'g', data=[self.sref])
        table.add_column(f'c<sub>ref</sub>{lstr:s}', 'g', data=[self.cref])
        table.add_column(f'b<sub>ref</sub>{lstr:s}', 'g', data=[self.bref])
        table.add_column(f'x<sub>ref</sub>{lstr:s}', '.3f', data=[self.rref.x])
        table.add_column(f'y<sub>ref</sub>{lstr:s}', '.3f', data=[self.rref.y])
        table.add_column(f'z<sub>ref</sub>{lstr:s}', '.3f', data=[self.rref.z])

        table = report.add_table()
        table.add_column('# D-Panels', 'd', data=[self.num_dpanels])
        table.add_column('# N-Panels', 'd', data=[self.num_npanels])
        table.add_column('# W-Panels', 'd', data=[self.num_wpanels])
        table.add_column('# Controls', 'd', data=[self.num_ctrls])

        return report

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

    def __str__(self) -> str:
        return self.to_mdobj().__str__()

    def _repr_markdown_(self) -> str:
        return self.to_mdobj()._repr_markdown_()

    def __repr__(self) -> str:
        return f'ConstantSystem({self.name:s})'
