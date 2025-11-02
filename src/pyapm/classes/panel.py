from typing import TYPE_CHECKING

from numpy import arange, asarray, full, minimum, ones, pi, sqrt, zeros
from numpy.linalg import solve
from pygeom.geom2d import Vector2D
from pygeom.geom3d import IHAT, KHAT, Coordinate, Vector
from pygeom.geom3d.tools import angle_between_vectors

from .face import Face
from .grid import Grid

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.flow import Flow
    from .edge import InternalEdge
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
    _faces: list[Face] = None
    _edges: list['InternalEdge'] = None
    _panel_gradient: 'NDArray' = None
    _facets: list[Face] = None
    # _hsvs: list[HorseshoeDoublet] = None
    _grdpnls: list[list['Panel']] = None
    _grdinds: list[list[int]] = None
    _grdfacs: list[list[float]] = None
    _edge_velg: 'NDArray' = None
    _edge_velp: Vector2D = None
    _edge_indg: 'NDArray' = None
    _edge_indp: 'NDArray' = None

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
            if self.noload:
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
    def noload(self) -> bool:
        noload = False
        if self.sheet is not None:
            noload = self.sheet.noload
        if self.section is not None:
            noload = self.section.noload
        if self.group is not None:
            noload = self.group.noload
        return noload

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
            for i, face in enumerate(self.faces):
                self._vecas[i] = face.grida
        return self._vecas

    @property
    def vecbs(self) -> Vector:
        if self._vecbs is None:
            self._vecbs = Vector.zeros(self.num)
            for i, face in enumerate(self.faces):
                self._vecbs[i] = face.gridb
        return self._vecbs

    @property
    def veccs(self) -> Vector:
        if self._veccs is None:
            self._veccs = Vector.zeros(self.num)
            for i, face in enumerate(self.faces):
                self._veccs[i] = face.gridc
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
    def edges(self) -> list['InternalEdge']:
        if self._edges is None:
            self._edges = []
        return self._edges

    @property
    def faces(self) -> list[Face]:
        if self._faces is None:
            self._faces = []
            for i in range(self.num):
                a = i - 1
                b = i
                grda = self.grids[a]
                grdb = self.grids[b]
                face = Face(i, grda, grdb, self)
                face.set_dirl(self.crd.dirx)
                self._faces.append(face)
        return self._faces

    def calc_edge_gradient(self) -> None:
        conedge = [edge for edge in self.edges if edge.panel is None]
        numconedge = len(conedge)
        self._edge_velp = Vector2D.zeros(numconedge + 1)
        self._edge_indp = zeros(numconedge + 1, dtype=int)
        self._edge_indp[0] = self.ind
        for i, edge in enumerate(conedge):
            vecg = edge.edge_point - self.pnto
            if edge.panela is self:
                face = edge.facea
                vecl = face.cord.vector_to_local(vecg)
                dirl = Vector2D.from_obj(vecl).to_unit()
                self._edge_indp[0] = edge.panela.ind
                self._edge_velp[0] += dirl*edge.panelb_fac
                self._edge_indp[i+1] = edge.panelb.ind
                self._edge_velp[i+1] += dirl*edge.panela_fac
            else:
                face = edge.faceb
                vecl = face.cord.vector_to_local(vecg)
                dirl = Vector2D.from_obj(vecl).to_unit()
                self._edge_indp[0] = edge.panelb.ind
                self._edge_velp[0] += dirl*edge.panela_fac
                self._edge_indp[i+1] = edge.panela.ind
                self._edge_velp[i+1] += dirl*edge.panelb_fac
        self._edge_velp /= numconedge

        # self._edge_velg = Vector2D.zeros(self.num)
        # self._edge_velp = Vector2D.zeros()
        # self._edge_indg = asarray([grid.ind for grid in self.grids])
        # self._edge_indp = self.ind
        # edge_count = 0
        # for i, face in enumerate(self.faces):
        #     found = False
        #     for edge in self.edges:
        #         if face.grida is edge.grida and face.gridb is edge.gridb:
        #             found = True
        #             break
        #         elif face.grida is edge.gridb and face.gridb is edge.grida:
        #             found = True
        #             break
        #     if edge.panel is None and found:
        #         edge_count += 1
        #         a = i - 1
        #         b = i
        #         self._edge_velg[a] += face.velg[0]
        #         self._edge_velg[b] += face.velg[1]
        #         self._edge_velp += face.velp
        # self._edge_velp /= edge_count
        # self._edge_velg /= edge_count
        # if edge_count == 1:
        #     print(f'{self}: edge_count = {edge_count:d}')
        #     # print(f'{self}: edge_velg = {self._edge_velg}')
        #     print(f'{self}: edge_velp = {self._edge_velp}')

    # @property
    # def edge_velg(self) -> Vector2D:
    #     if self._edge_velg is None:
    #         self.calc_edge_gradient()
    #     return self._edge_velg

    @property
    def edge_velp(self) -> Vector2D:
        if self._edge_velp is None:
            self.calc_edge_gradient()
        return self._edge_velp

    # @property
    # def edge_indg(self) -> 'NDArray':
    #     if self._edge_indg is None:
    #         self.calc_edge_gradient()
    #     return self._edge_indg

    @property
    def edge_indp(self) -> 'NDArray':
        if self._edge_indp is None:
            self.calc_edge_gradient()
        return self._edge_indp

    @property
    def facets(self) -> list[Face]:
        if self._facets is None:
            self._facets = []
            # edges: list['InternalEdge'] = []
            # for face in self.faces:
            #     for edge in self.edges:
            #         if edge.grida is face.grida and edge.gridb is face.gridb:
            #             edges.append(edge)
            #             break
            #         elif edge.grida is face.gridb and edge.gridb is face.grida:
            #             edges.append(edge)
            #             break
            for i, edge in enumerate(self.edges):
                if edge.panela is self:
                    facet = Face(2*i, edge.grida, edge.edge_point, self)
                elif edge.panelb is self:
                    facet = Face(2*i, edge.gridb, edge.edge_point, self)
                facet.set_dirl(self.crd.dirx)
                if facet.cord.dirz.dot(self.crd.dirz) < 0.0:
                    facet = Face(2*i, facet.gridb, facet.grida, self)
                facet.set_dirl(self.crd.dirx)
                facet.edge = edge
                self._facets.append(facet)
                if edge.panela is self:
                    facet = Face(2*i + 1, edge.edge_point, edge.gridb, self)
                elif edge.panelb is self:
                    facet = Face(2*i + 1, edge.edge_point, edge.grida, self)
                facet.set_dirl(self.crd.dirx)
                if facet.cord.dirz.dot(self.crd.dirz) < 0.0:
                    facet = Face(2*i + 1, facet.gridb, facet.grida, self)
                facet.set_dirl(self.crd.dirx)
                facet.edge = edge
                self._facets.append(facet)
        return self._facets

    @property
    def panel_gradient(self) -> 'NDArray':
        if self._panel_gradient is None:
            n = 1.0
            sum_x = 0.0
            sum_y = 0.0
            sum_xx = 0.0
            sum_xy = 0.0
            sum_yy = 0.0
            x_lst = [0.0]
            y_lst = [0.0]
            o_lst = [1.0]
            for edge in self.edges:
                if edge.panel is None:
                    pnte = self.crd.point_to_local(edge.edge_point)
                    xe = pnte.x
                    ye = pnte.y
                    n += 1.0
                    sum_x += xe
                    sum_y += ye
                    sum_xx += xe*xe
                    sum_xy += xe*ye
                    sum_yy += ye*ye
                    x_lst.append(xe)
                    y_lst.append(ye)
                    o_lst.append(1.0)
            amat = asarray([[sum_xx, sum_xy, sum_x],
                            [sum_xy, sum_yy, sum_y],
                            [sum_x, sum_y, n]])
            bmat = asarray([x_lst, y_lst, o_lst])
            cmat = solve(amat, bmat)
            self._panel_gradient = cmat
        return self._panel_gradient

    def diff_mu(self, mu: 'NDArray', mug: 'NDArray') -> Vector2D:
        qjac = Vector2D(0.0, 0.0)
        jac = 0.0
        i = 0
        for face in self.faces:
            i += 1
            qxJ = face.face_qxJ(mu, mug)
            qjac += qxJ
            jac += face.jac
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
                ra, rb, rc = gres[i-1], gres[i], pres
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
