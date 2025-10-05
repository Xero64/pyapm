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
    grds: list[Grid] = None
    ind: int = None
    grp: 'PanelGroup' = None
    sht: 'PanelSheet' = None
    sct: 'PanelSection' = None
    srfc: 'PanelSurface' = None
    _grdvec: Vector = None
    _num: int = None
    _pnto: Vector = None
    _grdrel: Vector = None
    _veca: Vector = None
    _vecb: Vector = None
    _vecab: Vector = None
    _vecaxb: Vector = None
    _sumaxb: Vector = None
    _dirxab: Vector = None
    _diryab: Vector = None
    _dirzab: Vector = None
    _baryinv: 'NDArray' = None
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

    def __init__(self, pid: int, grds: list[Grid]) -> None:
        self.pid = pid
        self.grds = grds
        for grd in self.grds:
            grd.pnls.add(self)

    def set_index(self, ind: int) -> None:
        self.ind = ind

    def dndl(self, gain: float, hvec: Vector) -> Vector:
        return gain*hvec.cross(self.nrm)

    @property
    def grdvec(self) -> Vector:
        if self._grdvec is None:
            self._grdvec = Vector.from_iter(self.grds)
        return self._grdvec

    @property
    def num(self) -> int:
        if self._num is None:
            self._num = self.grdvec.size
        return self._num

    @property
    def pnto(self) -> Vector:
        if self._pnto is None:
            self._pnto = self.grdvec.sum()/self.num
        return self._pnto

    @property
    def grdrel(self) -> Vector:
        if self._grdrel is None:
            self._grdrel = self.grdvec - self.pnto
        return self._grdrel

    @property
    def veca(self) -> Vector:
        if self._veca is None:
            inda = arange(-1, self.num-1)
            self._veca = self.grdvec[inda]
        return self._veca

    @property
    def vecb(self) -> Vector:
        if self._vecb is None:
            indb = arange(0, self.num)
            self._vecb = self.grdvec[indb]
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
        if self.sht is not None:
            noload = self.sht.noload
        if self.sct is not None:
            noload = self.sct.noload
        if self.grp is not None:
            noload = self.grp.noload
        return noload

    @property
    def nohsv(self) -> bool:
        nohsv = False
        if self.sht is not None:
            nohsv = self.sht.nohsv
        if self.sct is not None:
            nohsv = self.sct.nohsv
        if self.grp is not None:
            nohsv = self.grp.nohsv
        return nohsv

    def set_horseshoes(self, diro: Vector) -> None:
        self._hsvs = []
        if not self.nohsv:
            for i in range(self.num):
                grda = self.grds[i]
                grdb = self.grds[i-1]
                if grda.te and grdb.te:
                    self._hsvs.append(HorseshoeDoublet(grda, grdb, diro, self.ind))

    def check_panel(self, pnl: 'Panel') -> tuple[bool, bool, bool]:
        if pnl.grp is not None and self.grp is not None:
            grpchk = pnl.grp == self.grp
        else:
            grpchk = False
        if pnl.srfc is not None and self.srfc is not None:
            srfchk = pnl.srfc == self.srfc
        else:
            srfchk = False
        if srfchk:
            if pnl.sht is not None and self.sht is not None:
                typchk = True
            elif pnl.sct is not None and self.sct is not None:
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
        if grda in pnl.grds and grdb in pnl.grds:
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
                grda = self.grds[a]
                grdb = self.grds[b]
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

    @property
    def hsvs(self) -> list[float | None]:
        if self._hsvs is None:
            self.set_horseshoes(IHAT)
        return self._hsvs

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
            for i, grd in enumerate(self.grds):
                self._grdpnls.append([])
                for pnl in grd.pnls:
                    grpchk, srfchk, typchk = self.check_panel(pnl)
                    if grpchk:
                        angchk = self.check_angle(pnl)
                        if angchk:
                            self._grdpnls[i].append(pnl)
                    elif srfchk and typchk:
                        if grd.te:
                            angchk = self.check_angle(pnl)
                            if angchk:
                                self._grdpnls[i].append(pnl)
                        else:
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
            for i, grd in enumerate(self.grds):
                pnldist = [(grd-pnl.pnto).return_magnitude() for pnl in self.grdpnls[i]]
                pnlinvd = [1/dist for dist in pnldist]
                suminvd = sum(pnlinvd)
                self._grdfacs.append([invd/suminvd for invd in pnlinvd])
        return self._grdfacs

    def grid_res(self, pnlres: 'NDArray') -> list[float]:
        grdres = []
        for i in range(self.num):
            grdres.append(0.0)
            for ind, fac in zip(self.grdinds[i], self.grdfacs[i]):
                grdres[i] += pnlres[ind]*fac
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

    def __str__(self) -> str:
        return f'Panel({self.pid:d})'

    def __repr__(self) -> str:
        return self.__str__()
