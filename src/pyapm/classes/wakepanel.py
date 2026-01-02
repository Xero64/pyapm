#%%
# Import Dependencies
from typing import TYPE_CHECKING

from numpy import asarray, ones
from pyapm.classes import Grid
from pyapm.core.flow import Flow
from pygeom.geom3d import Vector

from .edge import BoundEdge, VortexEdge, BluntEdge
from .panel import Panel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .panelgroup import PanelGroup
    from .panelsection import PanelSection
    from .panelsheet import PanelSheet
    from .panelsurface import PanelSurface


class TrailingPanel(Panel):

    _adj_inds: 'NDArray' = None
    _adj_facs: 'NDArray' = None

    def __init__(self, pid: int, grids: list[Grid]) -> None:
        self.pid = pid
        self.grids = grids

    def calc_adjacents(self) -> None:
        adj_inds = []
        adj_facs = []
        for mesh_edge in self.mesh_edges:
            if isinstance(mesh_edge, BluntEdge):
                panel = mesh_edge.panel
                length = mesh_edge.panel_edge.length
                adj_inds.append(panel.ind)
                adj_facs.append(length)
        self._adj_inds = asarray(adj_inds, dtype=int)
        self._adj_facs = asarray(adj_facs, dtype=float)
        self._adj_facs /= self._adj_facs.sum()

    @property
    def adj_inds(self) -> 'NDArray':
        if self._adj_inds is None:
            self.calc_adjacents()
        return self._adj_inds

    @property
    def adj_facs(self) -> 'NDArray':
        if self._adj_facs is None:
            self.calc_adjacents()
        return self._adj_facs

    def constant_doublet_phi(self, pnts: Vector,
                             **kwargs: dict[str, float]) -> 'NDArray':

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdp as ctdp
        else:
            from pyapm.tools.numpy import numpy_ctdp as ctdp

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        aphidi = ctdp(pnts, vecas, vecbs, veccs, **kwargs)

        aphid = aphidi.sum(axis=-1)

        return aphid

    def constant_doublet_vel(self, pnts: Vector,
                             **kwargs: dict[str, float]) -> Vector:

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdv as ctdv
        else:
            from pyapm.tools.numpy import numpy_ctdv as ctdv

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), 4*self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        aveldi = ctdv(pnts, vecas, vecbs, veccs, **kwargs)

        aveld = aveldi.sum(axis=-1)

        return aveld

    def constant_doublet_flow(self, pnts: Vector,
                              **kwargs: dict[str, float]) -> 'Flow':

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdf as ctdf
        else:
            from pyapm.tools.numpy import numpy_ctdf as ctdf

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), 4*self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        aflwdi = ctdf(pnts, vecas, vecbs, veccs, **kwargs)

        afldw = aflwdi.sum(axis=-1)

        return afldw

    def __str__(self) -> str:
        return f'TrailingPanel({self.pid:d})'

    def __repr__(self) -> str:
        return self.__str__()


class WakePanel():
    pid: int = None
    gridas: list[Grid] = None
    gridbs: list[Grid] = None
    dirw: Vector = None
    ind: int = None
    group: 'PanelGroup' = None
    sheet: 'PanelSheet' = None
    section: 'PanelSection' = None
    surface: 'PanelSurface' = None
    _bound_edge: 'BoundEdge' = None
    _vortex_edges: list['VortexEdge'] = None
    _grids: list[Grid] = None
    _panel_edges: list['BoundEdge | VortexEdge'] = None
    _num: int = None
    _pntos: Vector = None
    _vecas: Vector = None
    _vecbs: Vector = None
    _veccs: Vector = None
    _veca: Vector = None
    _vecb: Vector = None
    # _adjpanels: list[Panel] = None
    _adj_inds: 'NDArray' = None
    _adj_facs: 'NDArray' = None

    def __init__(self, pid: int, gridas: list[Grid], gridbs: list[Grid],
                 dirw: Vector = None) -> None:
        self.pid = pid
        if len(gridas) != len(gridbs):
            raise ValueError('The len(gridas) must equal the len(gridbs).')
        self.gridas = gridas
        self.gridbs = gridbs
        self.dirw = dirw

    @property
    def num(self) -> int:
        if self._num is None:
            if len(self.gridas) != len(self.gridbs):
                raise ValueError('The len(gridas) must equal the len(gridbs).')
            self._num = len(self.gridas) -1
        return self._num

    @property
    def pntos(self) -> Vector:
        if self._pntos is None:
            self._pntos = Vector.zeros(self.num)
            for i in range(self.num):
                ip1 = i + 1
                gridai = self.gridas[i]
                gridaip1 = self.gridas[ip1]
                gridbi = self.gridbs[i]
                gridbip1 = self.gridbs[ip1]
                self.pntos[i] = (gridai + gridaip1 + gridbi + gridbip1)/4
        return self._pntos

    @property
    def vecas(self) -> Vector:
        if self._vecas is None:
            self._vecas = Vector.zeros(4*self.num)
            for i in range(self.num):
                fouri = 4*i
                ip1 = i + 1
                gridai = self.gridas[i]
                gridaip1 = self.gridas[ip1]
                gridbi = self.gridbs[i]
                gridbip1 = self.gridbs[ip1]
                self._vecas[fouri] = gridai
                self._vecas[fouri + 1] = gridbi
                self._vecas[fouri + 2] = gridbip1
                self._vecas[fouri + 3] = gridaip1
        return self._vecas

    @property
    def vecbs(self) -> Vector:
        if self._vecbs is None:
            self._vecbs = Vector.zeros(4*self.num)
            for i in range(self.num):
                fouri = 4*i
                ip1 = i + 1
                gridai = self.gridas[i]
                gridaip1 = self.gridas[ip1]
                gridbi = self.gridbs[i]
                gridbip1 = self.gridbs[ip1]
                self._vecbs[fouri] = gridbi
                self._vecbs[fouri + 1] = gridbip1
                self._vecbs[fouri + 2] = gridaip1
                self._vecbs[fouri + 3] = gridai
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
            self._veca = Vector.from_obj(self.gridas[-1])
        return self._veca

    @property
    def vecb(self) -> Vector:
        if self._vecb is None:
            self._vecb = Vector.from_obj(self.gridbs[-1])
        return self._vecb

    @property
    def bound_edge(self) -> 'BoundEdge':
        if self._bound_edge is None:
            grida = self.gridas[0]
            gridb = self.gridbs[0]
            self._bound_edge = BoundEdge(grida, gridb, self)
        return self._bound_edge

    @property
    def vortex_edges(self) -> list['VortexEdge']:
        if self._vortex_edges is None:
            self._vortex_edges = []
            for i in range(self.num):
                grida = self.gridas[self.num - i]
                gridb = self.gridas[self.num - i - 1]
                self._vortex_edges.append(VortexEdge(grida, gridb, self))
            for i in range(self.num):
                grida = self.gridbs[i]
                gridb = self.gridbs[i + 1]
                self._vortex_edges.append(VortexEdge(grida, gridb, self))
        return self._vortex_edges

    @property
    def grids(self) -> list[Grid]:
        if self._grids is None:
            self._grids = []
            self._grids.extend(reversed(self.gridas))
            self._grids.extend(self.gridbs)
        return self._grids

    @property
    def panel_edges(self) -> list['BoundEdge | VortexEdge']:
        if self._panel_edges is None:
            self._panel_edges = []
            self._panel_edges.extend(self.vortex_edges[:self.num])
            self._panel_edges.append(self.bound_edge)
            self._panel_edges.extend(self.vortex_edges[self.num:])
        return self._panel_edges

    def calc_adjacents(self) -> None:
        adj_inds = []
        adj_facs = []
        panel_edgea = self.bound_edge.mesh_edge.panel_edge
        panel_edgeb = self.bound_edge.mesh_edge.adjacent_edge
        if self.bound_edge.grida is panel_edgea.grida and \
            self.bound_edge.gridb is panel_edgea.gridb:
                panel_edgea, panel_edgeb = panel_edgeb, panel_edgea
        panela = panel_edgea.panel
        panelb = panel_edgeb.panel
        faca = 1.0
        facb = -faca
        if not isinstance(panela, TrailingPanel):
            adj_inds.append(panela.ind)
            adj_facs.append(faca)
        else:
            adj_inds += panela.adj_inds.tolist()
            adj_facs += (faca*panela.adj_facs).tolist()
        if not isinstance(panelb, TrailingPanel):
            adj_inds.append(panelb.ind)
            adj_facs.append(facb)
        else:
            adj_inds += panelb.adj_inds.tolist()
            adj_facs += (facb*panelb.adj_facs).tolist()
        self._adj_inds = asarray(adj_inds, dtype=int)
        self._adj_facs = asarray(adj_facs, dtype=float)

    @property
    def adj_inds(self) -> 'NDArray':
        if self._adj_inds is None:
            self.calc_adjacents()
        return self._adj_inds

    @property
    def adj_facs(self) -> 'NDArray':
        if self._adj_facs is None:
            self.calc_adjacents()
        return self._adj_facs

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

        if self.dirw is not None:
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

        if self.dirw is not None:
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

        if self.dirw is not None:
            pnts = pnts.reshape(shp)
            aflw += cwdf(pnts, self.veca, self.vecb, self.dirw, **kwargs)

        return aflw

    def __repr__(self):
        return f'WakePanel({self.pid})'

    def __str__(self):
        return self.__repr__()
