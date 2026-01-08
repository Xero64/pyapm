from typing import TYPE_CHECKING

from numpy import (arange, bincount, full, minimum, ones, pi, sqrt, unique,
                   zeros, concatenate)
from pygeom.geom2d import Vector2D
from pygeom.geom3d import IHAT, KHAT, Coordinate, Vector
from pygeom.geom3d.tools import angle_between_vectors

from .edge import InternalEdge, PanelEdge
from .face import Face
from .grid import Grid, Vertex

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.flow import Flow
    from .edge import MeshEdge
    from .panelgroup import PanelGroup
    from .panelsection import PanelSection
    from .panelsheet import PanelSheet
    from .panelsurface import PanelSurface

OOR2 = 1/sqrt(2.0)
ANGTOL = pi/4


class Panel():
    pid: int = None
    grids: list[Grid] = None
    group: 'PanelGroup' = None
    sheet: 'PanelSheet' = None
    section: 'PanelSection' = None
    surface: 'PanelSurface' = None
    ind: int = None
    _gridvec: Vector = None
    _num: int = None
    _pnto: Vector = None
    _grdrel: Vector = None
    _veca: Vector = None
    _vecb: Vector = None
    _vecas: Vector = None
    _vecbs: Vector = None
    _veccs: Vector = None
    _vecaxb: Vector = None
    _sumaxb: Vector = None
    _area: float = None
    _nrm: Vector = None
    _crd: Coordinate = None
    # _faces: list[Face] = None
    _panel_edges: list['PanelEdge'] = None
    _mesh_edges: list['MeshEdge'] = None
    # _panel_gradient: 'NDArray' = None
    _facets: list[Face] = None
    _vertices: list[Vertex] = None
    # _hsvs: list[HorseshoeDoublet] = None
    _grdpnls: list[list['Panel']] = None
    _grdinds: list[list[int]] = None
    _grdfacs: list[list[float]] = None
    _edge_velg: 'NDArray' = None
    _edge_velp: Vector2D = None
    _edge_indg: 'NDArray' = None
    _edge_indp: 'NDArray' = None
    _edge_indps: dict[int, 'NDArray'] = None
    _edge_velps: dict[int, Vector2D] = None
    _vert_indps: dict[int, 'NDArray'] = None
    _vert_facps: dict[int, 'NDArray'] = None

    def __init__(self, pid: int, grids: list[Grid]) -> None:
        self.pid = pid
        self.grids = grids
        self.link()

    def link(self) -> None:
        for grid in self.grids:
            grid.panels.add(self)

    def unlink(self) -> None:
        for grid in self.grids:
            grid.panels.remove(self)

    def dndl(self, gain: float, hvec: Vector) -> Vector:
        return gain*hvec.cross(self.nrm)

    @property
    def gridvec(self) -> Vector:
        if self._gridvec is None:
            self._gridvec = Vector.from_iter(self.grids)
        return self._gridvec

    @property
    def num(self) -> int:
        if self._num is None:
            self._num = self.gridvec.size
        return self._num

    @property
    def pnto(self) -> Vector:
        if self._pnto is None:
            self._pnto = self.gridvec.sum()/self.num
        return self._pnto

    @property
    def gridrel(self) -> Vector:
        if self._gridrel is None:
            self._gridrel = self.gridvec - self.pnto
        return self._gridrel

    @property
    def veca(self) -> Vector:
        if self._veca is None:
            inda = arange(-1, self.num-1)
            self._veca = self.gridvec[inda]
        return self._veca

    @property
    def vecb(self) -> Vector:
        if self._vecb is None:
            indb = arange(0, self.num)
            self._vecb = self.gridvec[indb]
        return self._vecb

    @property
    def vecab(self):
        if self._vecab is None:
            self._vecab = self.vecb - self.veca
        return self._vecab

    @property
    def vecaxb(self):
        if self._vecaxb is None:
            self._vecaxb = self.veca.cross(self.vecb)
        return self._vecaxb

    @property
    def sumaxb(self) -> Vector:
        if self._sumaxb is None:
            self._sumaxb = self.vecaxb.sum()
        return self._sumaxb

    @property
    def area(self):
        if self._area is None:
            if self.no_load:
                self._area = 0.0
            else:
                self._area = self.sumaxb.return_magnitude()/2
        return self._area

    @property
    def nrm(self) -> Vector:
        if self._nrm is None:
            self._nrm = self.sumaxb.to_unit()
        return self._nrm

    @property
    def no_load(self) -> bool:
        no_load = False
        if self.sheet is not None:
            no_load = self.sheet.no_load
        if self.section is not None:
            no_load = self.section.no_load
        if self.group is not None:
            no_load = self.group.no_load
        return no_load

    @property
    def nohsv(self) -> bool:
        nohsv = False
        if self.sheet is not None:
            nohsv = self.sheet.nohsv
        if self.section is not None:
            nohsv = self.section.nohsv
        if self.group is not None:
            nohsv = self.group.nohsv
        return nohsv

    @property
    def vecas(self) -> Vector:
        if self._vecas is None:
            self._vecas = Vector.zeros(self.num)
            for i in range(self.num):
                self._vecas[i] = self.grids[i - 1]
        return self._vecas

    @property
    def vecbs(self) -> Vector:
        if self._vecbs is None:
            self._vecbs = Vector.zeros(self.num)
            for i in range(self.num):
                self._vecbs[i] = self.grids[i]
        return self._vecbs

    @property
    def veccs(self) -> Vector:
        if self._veccs is None:
            self._veccs = Vector.zeros(self.num)
            for i in range(self.num):
                self._veccs[i] = self.pnto
        return self._veccs

    def check_panel(self, pnl: 'Panel') -> tuple[bool, bool, bool]:
        if pnl.group is not None and self.group is not None:
            grpchk = pnl.group == self.group
        else:
            grpchk = False
        if pnl.surface is not None and self.surface is not None:
            srfchk = pnl.surface == self.surface
        else:
            srfchk = False
        if srfchk:
            if pnl.sheet is not None and self.sheet is not None:
                typchk = True
            elif pnl.section is not None and self.section is not None:
                typchk = True
            else:
                typchk = False
        else:
            typchk = False
        return grpchk, srfchk, typchk

    def check_angle(self, pnl: 'Panel') -> bool:
        ang = angle_between_vectors(pnl.crd.dirz, self.crd.dirz)
        angchk = abs(ang) < ANGTOL
        return angchk

    def check_edge(self, pnl: 'Panel', grda: Grid, grdb: Grid) -> bool:
        edgchk = False
        if grda in pnl.grids and grdb in pnl.grids:
            edgchk = True
        return edgchk

    @property
    def crd(self) -> Coordinate:
        if self._crd is None:
            dirz = self.nrm
            vecy = dirz.cross(IHAT)
            magy = vecy.return_magnitude()
            if magy < OOR2:
                vecy = dirz.cross(KHAT)
            vecx = vecy.cross(dirz)
            pntc = self.pnto
            self._crd = Coordinate(pntc, vecx, vecy)
        return self._crd

    @property
    def panel_edges(self) -> list['PanelEdge']:
        if self._panel_edges is None:
            self._panel_edges = []
            for i in range(self.num):
                a = i - 1
                b = i
                grida = self.grids[a]
                gridb = self.grids[b]
                edge = PanelEdge(grida, gridb, self)
                self._panel_edges.append(edge)
        return self._panel_edges

    @property
    def mesh_edges(self) -> list['MeshEdge']:
        if self._mesh_edges is None:
            self._mesh_edges = []
            for panel_edge in self.panel_edges:
                self._mesh_edges.append(panel_edge.mesh_edge)
        return self._mesh_edges

    def calc_edge_gradient(self) -> None:
        intedges: dict[int, InternalEdge] = dict()
        for i, mesh_edge in enumerate(self.mesh_edges):
            if isinstance(mesh_edge, InternalEdge):
                intedges[i] = mesh_edge
        numintedge = len(intedges)
        self._edge_velp = Vector2D.zeros(numintedge + 1)
        self._edge_indp = zeros(numintedge + 1, dtype=int)
        self._edge_indp[0] = self.ind
        self._edge_indps: dict[int, 'NDArray'] = dict()
        self._edge_velps: dict[int, Vector2D] = dict()
        for i, (key, intedge) in enumerate(intedges.items()):
            vecg = intedge.edge_point - self.pnto
            if intedge.panela is self:
                face = intedge.facea
                vecl = face.cord.vector_to_local(vecg)
                dirl = Vector2D.from_obj(vecl).to_unit()
                self._edge_indp[0] = intedge.panela.ind
                self._edge_velp[0] += dirl*intedge.panelb_fac
                self._edge_indp[i+1] = intedge.panelb.ind
                self._edge_velp[i+1] += dirl*intedge.panela_fac
                self._edge_indps[key] = zeros(2, dtype=int)
                self._edge_indps[key][0] = intedge.panela.ind
                self._edge_indps[key][1] = intedge.panelb.ind
                self._edge_velps[key] = Vector2D.zeros(2)
                self._edge_velps[key][0] = dirl*intedge.panelb_fac
                self._edge_velps[key][1] = dirl*intedge.panela_fac
            elif intedge.panelb is self:
                face = intedge.faceb
                vecl = face.cord.vector_to_local(vecg)
                dirl = Vector2D.from_obj(vecl).to_unit()
                self._edge_indp[0] = intedge.panelb.ind
                self._edge_velp[0] += dirl*intedge.panela_fac
                self._edge_indp[i+1] = intedge.panela.ind
                self._edge_velp[i+1] += dirl*intedge.panelb_fac
                self._edge_indps[key] = zeros(2, dtype=int)
                self._edge_indps[key][0] = intedge.panelb.ind
                self._edge_indps[key][1] = intedge.panela.ind
                self._edge_velps[key] = Vector2D.zeros(2)
                self._edge_velps[key][0] = dirl*intedge.panela_fac
                self._edge_velps[key][1] = dirl*intedge.panelb_fac
        if numintedge > 0:
            self._edge_velp /= numintedge
        for i in range(self.num):
            if i not in intedges:
                self._edge_indps[i] = self._edge_indp
                self._edge_velps[i] = self._edge_velp

    def calc_vertex_gradient(self) -> None:

        self._vert_indps: dict[int, 'NDArray'] = dict()
        self._vert_facps: dict[int, 'NDArray'] = dict()

        for indv in range(self.num):

            vertex = self.vertices[indv]

            inda = indv
            indb = indv + 1
            if indv + 1 == self.num:
                indb = 0

            mesh_edgea = self.mesh_edges[inda]
            mesh_edgeb = self.mesh_edges[indb]
            indpa = mesh_edgea.indps
            indpb = mesh_edgeb.indps
            facpa = mesh_edgea.facps
            facpb = mesh_edgeb.facps

            veca = self.crd.point_to_local(mesh_edgea.edge_point)
            vecb = self.crd.point_to_local(mesh_edgeb.edge_point)
            vecv = self.crd.point_to_local(vertex)
            dira = Vector2D.from_obj(veca)
            dirb = Vector2D.from_obj(vecb)
            dirv = Vector2D.from_obj(vecv)

            denom = dira.cross(dirb)
            muafac = facpa * dirv.cross(dirb) / denom
            mubfac = facpb * dira.cross(dirv) / denom
            mupfac = denom + dirv.cross(dira) + dirb.cross(dirv)
            mupfac = mupfac / denom


            conindps = concatenate(([self.ind], indpa, indpb))
            confacps = concatenate(([mupfac], muafac, mubfac))

            uniqindps, invinds = unique(conindps, return_inverse=True)
            uniqfacps = bincount(invinds, weights=confacps)

            self._vert_indps[indv] = uniqindps
            self._vert_facps[indv] = uniqfacps

    @property
    def edge_velp(self) -> Vector2D:
        if self._edge_velp is None:
            self.calc_edge_gradient()
        return self._edge_velp

    @property
    def edge_indp(self) -> 'NDArray':
        if self._edge_indp is None:
            self.calc_edge_gradient()
        return self._edge_indp

    @property
    def edge_indps(self) -> dict[int, 'NDArray']:
        if self._edge_indps is None:
            self.calc_edge_gradient()
        return self._edge_indps

    @property
    def edge_velps(self) -> dict[int, Vector2D]:
        if self._edge_velps is None:
            self.calc_edge_gradient()
        return self._edge_velps

    @property
    def vert_indps(self) -> dict[int, 'NDArray']:
        if self._vert_indps is None:
            self.calc_vertex_gradient()
        return self._vert_indps

    @property
    def vert_facps(self) -> dict[int, 'NDArray']:
        if self._vert_facps is None:
            self.calc_vertex_gradient()
        return self._vert_facps

    @property
    def facets(self) -> list[Face]:
        if self._facets is None:
            self._facets = []
            for panel_edge in self.panel_edges:
                facet = Face(panel_edge.grida, panel_edge.edge_point, self)
                facet.set_dirl(self.crd.dirx)
                facet.edge = panel_edge
                self._facets.append(facet)
                facet = Face(panel_edge.edge_point, panel_edge.gridb, self)
                facet.set_dirl(self.crd.dirx)
                facet.edge = panel_edge
                self._facets.append(facet)
        return self._facets

    @property
    def vertices(self) -> list[Vertex]:
        if self._vertices is None:
            self._vertices = []
            for grid in self.grids:
                for vertex in grid.vertices:
                    if self in vertex.panels:
                        self._vertices.append(vertex)
                        break
        return self._vertices

    def grids_before_and_after_grid(self, grid: Grid) -> tuple[Grid, Grid]:
        ind = self.grids.index(grid)
        ind_bef = ind - 1
        ind_aft = ind + 1
        if ind_aft >= self.num:
            ind_aft -= self.num
        grida = self.grids[ind_bef]
        gridb = self.grids[ind_aft]
        return grida, gridb

    def edges_before_and_after_grid(self, grid: Grid) -> tuple[PanelEdge,
                                                               PanelEdge]:
        ind = self.grids.index(grid)
        ind_bef = ind
        ind_aft = ind + 1
        if ind_aft >= self.num:
            ind_aft -= self.num
        edgea = self.panel_edges[ind_bef]
        edgeb = self.panel_edges[ind_aft]
        return edgea, edgeb

    # @property
    # def panel_gradient(self) -> 'NDArray':
    #     if self._panel_gradient is None:
    #         n = 1.0
    #         sum_x = 0.0
    #         sum_y = 0.0
    #         sum_xx = 0.0
    #         sum_xy = 0.0
    #         sum_yy = 0.0
    #         x_lst = [0.0]
    #         y_lst = [0.0]
    #         o_lst = [1.0]
    #         for edge in self.edges:
    #             if edge.panel is None:
    #                 pnte = self.crd.point_to_local(edge.edge_point)
    #                 xe = pnte.x
    #                 ye = pnte.y
    #                 n += 1.0
    #                 sum_x += xe
    #                 sum_y += ye
    #                 sum_xx += xe*xe
    #                 sum_xy += xe*ye
    #                 sum_yy += ye*ye
    #                 x_lst.append(xe)
    #                 y_lst.append(ye)
    #                 o_lst.append(1.0)
    #         amat = asarray([[sum_xx, sum_xy, sum_x],
    #                         [sum_xy, sum_yy, sum_y],
    #                         [sum_x, sum_y, n]])
    #         bmat = asarray([x_lst, y_lst, o_lst])
    #         cmat = solve(amat, bmat)
    #         self._panel_gradient = cmat
    #     return self._panel_gradient

    def diff_mu(self, mud: 'NDArray', mue: 'NDArray', muv: 'NDArray') -> Vector2D:
        qjac = Vector2D(0.0, 0.0)
        jac = 0.0
        i = 0
        for panel_edge in self.panel_edges:
            i += 1
            qxJ = panel_edge.face.face_qxJ(mud, mue, muv)
            qjac += qxJ
            jac += panel_edge.face.jac
        q = qjac/jac
        return q

    @property
    def grdpnls(self) -> list[list['Panel']]:
        if self._grdpnls is None:
            self._grdpnls = []
            for i, grid in enumerate(self.grids):
                self._grdpnls.append([])
                for pnl in grid.panels:
                    grpchk, srfchk, typchk = self.check_panel(pnl)
                    if grpchk:
                        angchk = self.check_angle(pnl)
                        if angchk:
                            self._grdpnls[i].append(pnl)
                    elif srfchk and typchk:
                        self._grdpnls[i].append(pnl)
        return self._grdpnls

    @property
    def grdinds(self) -> list[list[int]]:
        if self._grdinds is None:
            self._grdinds = []
            for i in range(self.num):
                self._grdinds.append([])
                for pnl in self.grdpnls[i]:
                    self._grdinds[i].append(pnl.ind)
        return self._grdinds

    @property
    def grdfacs(self) -> list[list[float]]:
        if self._grdfacs is None:
            self._grdfacs = []
            for i, grid in enumerate(self.grids):
                pnldist = [(grid - pnl.pnto).return_magnitude() for pnl in self.grdpnls[i]]
                pnlinvd = [1 / dist for dist in pnldist]
                suminvd = sum(pnlinvd)
                self._grdfacs.append([invd/suminvd for invd in pnlinvd])
        return self._grdfacs

    def grid_res(self, pnlres: 'NDArray') -> list[float]:
        grdres = []
        for i in range(self.num):
            grdres.append(0.0)
            for ind, fac in zip(self.grdinds[i], self.grdfacs[i]):
                grdres[i] += pnlres[ind] * fac
        return grdres

    def within_and_absz_ttol(self, pnts: Vector, ttol: float=0.1) -> tuple['NDArray', 'NDArray']:
        nttol = -ttol
        wint = full(pnts.shape, False)
        absz = full(pnts.shape, float('inf'))
        for face in self.faces:
            mintf, abszf = face.mint_and_absz(pnts)
            chk = mintf > nttol
            wint[chk] = True
            absz = minimum(absz, abszf)
        return wint, absz

    def point_res(self, pnlres: 'NDArray', pnt: Vector, ttol: float=0.1) -> float:
        gres = self.grid_res(pnlres)
        pres = pnlres[self.ind]
        r = 0.0
        for i, face in enumerate(self.faces):
            t123, _ = face.t123_and_absz(pnt)
            ta, tb, tc = t123[0], t123[1], t123[2]
            mint = min(ta, tb, tc)
            if mint > -ttol:
                ra, rb, rc = gres[i - 1], gres[i], pres
                r = ra*ta + rb*tb + rc*tc
                break
        return r

    def constant_doublet_source_phi(self, pnts: Vector,
                                    **kwargs: dict[str, float]) -> tuple['NDArray':, 'NDArray']:

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdsp as ctdsp
        else:
            from pyapm.tools.numpy import numpy_ctdsp as ctdsp

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        aphidi, aphisi = ctdsp(pnts, vecas, vecbs, veccs, **kwargs)

        aphid = aphidi.sum(axis=-1)
        aphis = aphisi.sum(axis=-1)

        return aphid, aphis

    def constant_doublet_source_vel(self, pnts: Vector,
                                    **kwargs: dict[str, float]) -> tuple[Vector, Vector]:

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdsv as ctdsv
        else:
            from pyapm.tools.numpy import numpy_ctdsv as ctdsv

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), 4*self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        aveldi, avelsi = ctdsv(pnts, vecas, vecbs, veccs, **kwargs)

        aveld = aveldi.sum(axis=-1)
        avels = avelsi.sum(axis=-1)

        return aveld, avels

    def constant_doublet_source_flow(self, pnts: Vector,
                                     **kwargs: dict[str, float]) -> tuple['Flow', 'Flow']:

        from pyapm import USE_CUPY

        if USE_CUPY:
            from pyapm.tools.cupy import cupy_ctdsf as ctdsf
        else:
            from pyapm.tools.numpy import numpy_ctdsf as ctdsf

        shp = pnts.shape
        ndm = pnts.ndim

        pntshp = (*shp, 1)
        pnts = pnts.reshape(pntshp)

        vecshp = (*ones(ndm, dtype=int), 4*self.num)
        vecas = self.vecas.reshape(vecshp)
        vecbs = self.vecbs.reshape(vecshp)
        veccs = self.veccs.reshape(vecshp)

        aflwdi, aflwsi = ctdsf(pnts, vecas, vecbs, veccs, **kwargs)

        afldw = aflwdi.sum(axis=-1)
        aflds = aflwsi.sum(axis=-1)

        return afldw, aflds

    def __str__(self) -> str:
        return f'Panel({self.pid:d})'

    def __repr__(self) -> str:
        return self.__str__()
