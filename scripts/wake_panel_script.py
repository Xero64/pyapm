#%%
# Import Dependencies
from typing import TYPE_CHECKING
from pyapm.classes import Grid
from pyapm.core.flow import Flow
from pygeom.geom3d import Vector
from numpy import ones
from numpy.linalg import norm

if TYPE_CHECKING:
    from numpy.typing import NDArray

from pyapm.tools.mesh import point_mesh_xy
from pyapm.tools.plot import point_contourf_xy

#%%
# Create Wake Panel Class
class WakePanel:
    pid: int = None
    grdas: list[Grid] = None
    grdbs: list[Grid] = None
    dirw: Vector = None
    indo: int = None
    _num: int = None
    _pntos: Vector = None
    _vecas: Vector = None
    _vecbs: Vector = None
    _veccs: Vector = None
    _veca: Vector = None
    _vecb: Vector = None

    def __init__(self, pid: int, grdas: list[Grid], grdbs: list[Grid], dirw: Vector) -> None:
        self.pid = pid
        if len(grdas) != len(grdbs):
            raise ValueError('The len(grdas) must equal the len(grdbs).')
        self.grdas = grdas
        self.grdbs = grdbs
        self.dirw = dirw

    @property
    def num(self) -> int:
        if self._num is None:
            if len(self.grdas) != len(self.grdbs):
                raise ValueError('The len(grdas) must equal the len(grdbs).')
            self._num = len(self.grdas) -1
        return self._num

    @property
    def pntos(self) -> Vector:
        if self._pntos is None:
            self._pntos = Vector.zeros(self.num)
            for i in range(self.num):
                ip1 = i + 1
                grdai = self.grdas[i]
                grdaip1 = self.grdas[ip1]
                grdbi = self.grdas[i]
                grdbip1 = self.grdbs[ip1]
                self.pntos[i] = (grdai + grdaip1 + grdbi + grdbip1)/4
        return self._pntos

    @property
    def vecas(self) -> Vector:
        if self._vecas is None:
            self._vecas = Vector.zeros(4*self.num)
            for i in range(self.num):
                fouri = 4*i
                ip1 = i + 1
                grdai = self.grdas[i]
                grdaip1 = self.grdas[ip1]
                grdbi = self.grdbs[i]
                grdbip1 = self.grdbs[ip1]
                self._vecas[fouri] = grdai
                self._vecas[fouri + 1] = grdbi
                self._vecas[fouri + 2] = grdbip1
                self._vecas[fouri + 3] = grdaip1
        return self._vecas

    @property
    def vecbs(self) -> Vector:
        if self._vecbs is None:
            self._vecbs = Vector.zeros(4*self.num)
            for i in range(self.num):
                fouri = 4*i
                ip1 = i + 1
                grdai = self.grdas[i]
                grdaip1 = self.grdas[ip1]
                grdbi = self.grdbs[i]
                grdbip1 = self.grdbs[ip1]
                self._vecbs[fouri] = grdbi
                self._vecbs[fouri + 1] = grdbip1
                self._vecbs[fouri + 2] = grdaip1
                self._vecbs[fouri + 3] = grdai
        return self._vecbs

    @property
    def veccs(self) -> Vector:
        if self._veccs is None:
            self._veccs = Vector.zeros(4*self.num)
            for i in range(self.num):
                fouri = 4*i
                self._veccs[fouri:fouri + 4] = self.pntos[i]
        return self._veccs

    @property
    def veca(self) -> Vector:
        if self._veca is None:
            self._veca = Vector.from_obj(self.grdas[-1])
        return self._veca

    @property
    def vecb(self) -> Vector:
        if self._vecb is None:
            self._vecb = Vector.from_obj(self.grdbs[-1])
        return self._vecb

    def constant_doublet_phi(self, pnts: Vector, **kwargs: dict[str, float]) -> 'NDArray':

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdp as ctdp
            from pyapm.tools.cupy import cupy_cwdp as cwdp
        else:
            from pyapm.tools.numpy import numpy_ctdp as ctdp
            from pyapm.tools.numpy import numpy_cwdp as cwdp

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), 4*self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        aphi = ctdp(pnts, vecas, vecbs, veccs, **kwargs).sum(axis=-1)

        pnts = pnts.reshape(shp)

        aphi += cwdp(pnts, self.veca, self.vecb, self.dirw, **kwargs)

        return aphi

    def constant_doublet_vel(self, pnts: Vector, **kwargs: dict[str, float]) -> Vector:

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdv as ctdv
            from pyapm.tools.cupy import cupy_cwdv as cwdv
        else:
            from pyapm.tools.numpy import numpy_ctdv as ctdv
            from pyapm.tools.numpy import numpy_cwdv as cwdv

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), 4*self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        avel = ctdv(pnts, vecas, vecbs, veccs, **kwargs).sum(axis=-1)

        pnts = pnts.reshape(shp)

        avel += cwdv(pnts, self.veca, self.vecb, self.dirw, **kwargs)

        return avel

    def constant_doublet_flow(self, pnts: Vector, **kwargs: dict[str, float]) -> Flow:

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdf as ctdf
            from pyapm.tools.cupy import cupy_cwdf as cwdf
        else:
            from pyapm.tools.numpy import numpy_ctdf as ctdf
            from pyapm.tools.numpy import numpy_cwdf as cwdf

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), 4*self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        aflw = ctdf(pnts, vecas, vecbs, veccs, **kwargs).sum(axis=-1)

        pnts = pnts.reshape(shp)

        aflw += cwdf(pnts, self.veca, self.vecb, self.dirw, **kwargs)

        return aflw

#%%
# Create Wake Panel
grdas = [
    Grid(1, -0.5, 0.5, 0.0),
    Grid(2, -0.25, 0.55, 0.0),
    Grid(3, 0.0, 0.45, 0.0),
    Grid(4, 0.25, 0.5, 0.0)
    ]
grdbs = [
    Grid(5, -0.5, -0.5, 0.0),
    Grid(6, -0.25, -0.45, 0.0),
    Grid(7, 0.0, -0.55, 0.0),
    Grid(8, 0.25, -0.5, 0.0)
]
dirw = Vector(1.0, -0.25, 0.0)

wpnl = WakePanel(1, grdas, grdbs, dirw)

#%%
# Mesh Points and Calculate Influence
pnts = point_mesh_xy(0.0, 0.0, -0.01, 200, 200, 1.0, 1.0)

flww = wpnl.constant_doublet_flow(pnts, cond = -1.0)

phiw = wpnl.constant_doublet_phi(pnts, cond = -1.0)

velw = wpnl.constant_doublet_vel(pnts)

#%%
# Plot Results
axph, cfph = point_contourf_xy(pnts, flww.phi)

axvx, cfvx = point_contourf_xy(pnts, flww.vel.x)

axvy, cfvy = point_contourf_xy(pnts, flww.vel.y)

axvz, cfvz = point_contourf_xy(pnts, flww.vel.z)

#%%
# Print Outs

diff_flw = flww - Flow(phiw, velw)

print(f'{norm(diff_flw.phi) = :.12f}')
print(f'{norm(diff_flw.vel.x) = :.12f}')
print(f'{norm(diff_flw.vel.y) = :.12f}')
print(f'{norm(diff_flw.vel.z) = :.12f}')
