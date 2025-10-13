from typing import TYPE_CHECKING

from numpy import (absolute, arange, asarray, full, minimum, ones, pi, sqrt,
                   zeros)
from numpy.linalg import inv, solve
from pygeom.geom2d import Vector2D
from pygeom.geom3d import IHAT, KHAT, Coordinate, Vector
from pygeom.geom3d.tools import angle_between_vectors

from .grid import Grid
from .horseshoedoublet import HorseshoeDoublet

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.flow import Flow
    from .edge import Edge
    from .panelgroup import PanelGroup
    from .panelsection import PanelSection
    from .panelsheet import PanelSheet
    from .panelsurface import PanelSurface

OOR2 = 1/sqrt(2.0)
ANGTOL = pi/4


class PanelFace:
    fid: int = None
    grda: Grid = None
    grdb: Grid = None
    pnl: 'Panel' = None
    ind: int = None
    _pnto: Vector = None
    _nrml: Vector = None
    _jac: float = None
    _area: float = None
    _cord: Coordinate = None
    _pnta: Vector2D = None
    _pntb: Vector2D = None
    _pntc: Vector2D = None
    _xba: float = None
    _xac: float = None
    _xcb: float = None
    _yab: float = None
    _ybc: float = None
    _yca: float = None
    _baryinv: float = None

    def __init__(self, fid: int, grda: Grid, grdb: Grid, pnl: 'Panel') -> None:
        self.fid = fid
        self.grda = grda
        self.grdb = grdb
        self.pnl = pnl

    @property
    def grdc(self) -> Vector:
        return self.pnl.pnto

    @property
    def pnto(self) -> Vector:
        if self._pnto is None:
            self._pnto = (self.grda + self.grdb + self.grdc)/3
        return self._pnto

    def calc_normal_and_jac(self) -> tuple[Vector, float]:
        if self._nrml is None or self._area is None:
            vecab = self.grdb - self.grda
            vecbc = self.grdc - self.grdb
            nrml, jac = vecab.cross(vecbc).to_unit(return_magnitude=True)
        return nrml, jac

    @property
    def nrml(self) -> Vector:
        if self._nrml is None:
            self._nrml, self._jac = self.calc_normal_and_jac()
        return self._nrml

    @property
    def jac(self) -> float:
        if self._jac is None:
            self._nrml, self._jac = self.calc_normal_and_jac()
        return self._jac

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self.jac/2.0
        return self._area

    def set_dirl(self, vecl: Vector) -> None:
        dirz = self.nrml
        vecy = dirz.cross(vecl)
        vecx = vecy.cross(dirz)
        self._cord = Coordinate(self.grdc, vecx, vecy)

    @property
    def cord(self) -> Coordinate:
        if self._cord is None:
            raise ValueError('PanelFace coordinate not set. Call set_dirl() first.')
        return self._cord

    @property
    def pnta(self) -> Vector2D:
        if self._pnta is None:
            veca = Vector.from_obj(self.grda)
            loca = self.cord.point_to_local(veca)
            self._pnta = Vector2D.from_obj(loca)
        return self._pnta

    @property
    def pntb(self) -> Vector2D:
        if self._pntb is None:
            vecb = Vector.from_obj(self.grdb)
            locb = self.cord.point_to_local(vecb)
            self._pntb = Vector2D.from_obj(locb)
        return self._pntb

    @property
    def pntc(self) -> Vector2D:
        if self._pntc is None:
            vecc = Vector.from_obj(self.grdc)
            locc = self.cord.point_to_local(vecc)
            self._pntc = Vector2D.from_obj(locc)
        return self._pntc

    @property
    def xba(self) -> float:
        if self._xba is None:
            self._xba = self.pntb.x - self.pnta.x
        return self._xba

    @property
    def xac(self) -> float:
        if self._xac is None:
            self._xac = self.pnta.x - self.pntc.x
        return self._xac

    @property
    def xcb(self) -> float:
        if self._xcb is None:
            self._xcb = self.pntc.x - self.pntb.x
        return self._xcb

    @property
    def yab(self) -> float:
        if self._yab is None:
            self._yab = self.pnta.y - self.pntb.y
        return self._yab

    @property
    def ybc(self) -> float:
        if self._ybc is None:
            self._ybc = self.pntb.y - self.pntc.y
        return self._ybc

    @property
    def yca(self) -> float:
        if self._yca is None:
            self._yca = self.pntc.y - self.pnta.y
        return self._yca

    def face_qxJ(self, mu: 'NDArray', mug: 'NDArray') -> Vector2D:
        muc = mu[self.pnl.ind]
        mua = mug[self.grda.ind]
        mub = mug[self.grdb.ind]
        qxJ = mua*self.ybc + mub*self.yca + muc*self.yab
        qyJ = mua*self.xcb + mub*self.xac + muc*self.xba
        return Vector2D(qxJ, qyJ)

    @property
    def baryinv(self) -> 'NDArray':
        if self._baryinv is None:
            amat = zeros((3, 3))
            amat[0, :] = 1.0
            amat[1, 0] = self.pnta.x
            amat[2, 0] = self.pnta.y
            amat[1, 1] = self.pntb.x
            amat[2, 1] = self.pntb.y
            amat[1, 2] = self.pntc.x
            amat[2, 2] = self.pntc.y
            self._baryinv = inv(amat)
        return self._baryinv

    def mint_and_absz(self, pnts: Vector) -> tuple['NDArray', 'NDArray']:
        t123, absz = self.t123_and_absz(pnts)
        mint = t123.min(axis=-1)
        return mint, absz

    def t123_and_absz(self, pnts: Vector) -> tuple['NDArray', 'NDArray']:
        pntl = self.cord.point_to_local(pnts)
        xy1 = ones((*pnts.shape, 3))
        xy1[..., 1] = pntl.x
        xy1[..., 2] = pntl.y
        t123 = xy1@self.baryinv.transpose()
        absz = absolute(pntl.z)
        return t123, absz


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
    # _vecab: Vector = None
    _vecaxb: Vector = None
    _sumaxb: Vector = None
    # _dirxab: Vector = None
    # _diryab: Vector = None
    # _dirzab: Vector = None
    # _baryinv: 'NDArray' = None
    _area: float = None
    _nrm: Vector = None
    _crd: Coordinate = None
    _faces: list[PanelFace] = None
    _edges: list['Edge'] = None
    _panel_gradient: 'NDArray' = None
    _hsvs: list[HorseshoeDoublet] = None
    _grdpnls: list[list['Panel']] = None
    _grdinds: list[list[int]] = None
    _grdfacs: list[list[float]] = None

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
                self._vecas[i] = face.grda
        return self._vecas

    @property
    def vecbs(self) -> Vector:
        if self._vecbs is None:
            self._vecbs = Vector.zeros(self.num)
            for i, face in enumerate(self.faces):
                self._vecbs[i] = face.grdb
        return self._vecbs

    @property
    def veccs(self) -> Vector:
        if self._veccs is None:
            self._veccs = Vector.zeros(self.num)
            for i, face in enumerate(self.faces):
                self._veccs[i] = face.grdc
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
    def edges(self) -> list['Edge']:
        if self._edges is None:
            self._edges = []
        return self._edges

    @property
    def faces(self) -> list[PanelFace]:
        if self._faces is None:
            self._faces = []
            for i in range(self.num):
                a = i - 1
                b = i
                grda = self.grids[a]
                grdb = self.grids[b]
                face = PanelFace(i, grda, grdb, self)
                face.set_dirl(self.crd.dirx)
                self._faces.append(face)
        return self._faces

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
