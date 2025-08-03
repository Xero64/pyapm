from typing import TYPE_CHECKING

from numpy import (absolute, acos, asarray, full, logical_not, minimum, ones,
                   pi, sqrt)
from numpy.linalg import solve
from pygeom.geom2d import Vector2D
from pygeom.geom3d import IHAT, KHAT, Coordinate, Vector

from .dirichletpoly import DirichletPoly
from .grid import Grid
from .horseshoedoublet import HorseshoeDoublet

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .edge import Edge
    from .panelsection import PanelSection
    from .panelsheet import PanelSheet
    from .panelsurface import PanelSurface

oor2 = 1/sqrt(2.0)
angtol = pi/4

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


class Panel(DirichletPoly):
    pid: int = None
    grds: list[Grid] = None
    gids: list[int] = None
    ind: int = None
    grp: int = None
    sht: 'PanelSheet' = None
    sct: 'PanelSection' = None
    srfc: 'PanelSurface' = None
    _crd: Coordinate = None
    _faces: list[PanelFace] = None
    _edges: list['Edge'] = None
    _panel_gradient: 'NDArray' = None
    _hsvs: list[HorseshoeDoublet] = None
    _grdlocs: list[Vector] = None
    _edgpnls: list['Panel'] = None
    _edginds: list[list[int]] = None
    _edgpnts: list[Vector] = None
    _edgdist: list[float] = None
    _edgfacs: list[float] = None
    _edgpntl: list[Vector] = None
    _edglens: list[float] = None
    _grdpnls: list[list['Panel']] = None
    _grdinds: list[list[int]] = None
    _grdfacs: list[list[float]] = None

    def __init__(self, pid: int, grds: list[Grid]):
        super().__init__(grds)
        self.pid = pid
        for grd in self.grds:
            grd.pnls.add(self)

    def set_index(self, ind: int):
        self.ind = ind

    def dndl(self, gain: float, hvec: Vector):
        return gain*hvec.cross(self.nrm)

    @property
    def noload(self):
        noload = False
        if self.sht is not None:
            noload = self.sht.noload
        if self.sct is not None:
            noload = self.sct.noload
        # if self.grp is not None:
        #     noload = self.grp.noload
        return noload

    @property
    def nohsv(self):
        nohsv = False
        if self.sht is not None:
            nohsv = self.sht.nohsv
        if self.sct is not None:
            nohsv = self.sct.nohsv
        # if self.grp is not None:
        #     nohsv = self.grp.nohsv
        return nohsv

    def set_horseshoes(self, diro: Vector):
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
        angchk = abs(ang) < angtol
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
            if magy < oor2:
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
                # print(f'{edge.panela = }, {edge.panelb = }, {edge.panel = }')
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
            # print(f'amat = \n{amat}\n')
            bmat = asarray([x_lst, y_lst, o_lst])
            # print(f'bmat = \n{bmat}\n')
            cmat = solve(amat, bmat)
            self._panel_gradient = cmat
        return self._panel_gradient

    @property
    def hsvs(self):
        if self._hsvs is None:
            self.set_horseshoes(IHAT)
        return self._hsvs

    @property
    def area(self):
        if self.noload:
            area = 0.0
        else:
            area = super().area
        return area

    @property
    def grdlocs(self):
        if self._grdlocs is None:
            self._grdlocs = []
            for grd in self.grds:
                self._grdlocs.append(self.crd.point_to_local(grd))
        return self._grdlocs

    @property
    def edgpnls(self):
        if self._edgpnls is None:
            self._edgpnls = []
            for i in range(self.num):
                grda = self.grds[i-1]
                grdb = self.grds[i]
                pnls = grda.pnls & grdb.pnls
                pnllst = list(pnls)
                self._edgpnls.append(pnllst)
                # pnls.remove(self)
                # pnllst = list(pnls)
                # pnls.clear()
                # if len(pnllst) > 0:
                #     pnlb = pnllst[0]
                #     _, srfchk, _ = self.check_panel(pnlb)
                #     if srfchk:
                #         if not grda.te and not grdb.te:
                #             angchk = self.check_angle(pnlb)
                #             if angchk:
                #                 pnls.add(pnlb)
                #         else:
                #             pnls.add(pnlb)
                #     else:
                #         angchk = self.check_angle(pnlb)
                #         if angchk:
                #             pnls.add(pnlb)
                # if grda.te and grdb.te:
                #     pnls.add(self)
                # elif len(self._edgpnls[i]) > 0:
                #     pnls.add(self)
                # self._edgpnls.append(list(pnls))
        return self._edgpnls

    @property
    def edginds(self):
        if self._edginds is None:
            self._edginds = []
            for i in range(self.num):
                self._edginds.append([])
                for pnl in self.edgpnls[i]:
                    self._edginds[i].append(pnl.ind)
        return self._edginds

    @property
    def edgpnts(self):
        if self._edgpnts is None:
            self._edgpnts = []
            for i in range(self.num):
                grda = self.grds[i-1]
                grdb = self.grds[i]
                if len(self.edgpnls[i]) == 2:
                    pnla = self.edgpnls[i][-1]
                    pnlb = self.edgpnls[i][-2]
                    dirx = (grdb-grda).to_unit()
                    dirza = pnla.nrm
                    dirzb = pnlb.nrm
                    dirya = dirza.cross(dirx)
                    diryb = dirzb.cross(dirx)
                    veca = pnla.pnto - grda
                    vecb = pnlb.pnto - grda
                    xa = veca.dot(dirx)
                    xb = vecb.dot(dirx)
                    ya = veca.dot(dirya)
                    yb = vecb.dot(diryb)
                    xc = xa - (xb-xa)/(yb-ya)*ya
                    self._edgpnts.append(grda + dirx*xc)
                else:
                    self._edgpnts.append((grda + grdb)/2)
        return self._edgpnts

    @property
    def edgdist(self):
        if self._edgdist is None:
            self._edgdist = []
            for i, pnt in enumerate(self.edgpnts):
                self._edgdist.append([])
                for pnl in self.edgpnls[i]:
                    self._edgdist[i].append((pnt-pnl.pnto).return_magnitude())
        return self._edgdist

    @property
    def edgfacs(self):
        if self._edgfacs is None:
            self._edgfacs = []
            for i in range(self.num):
                self._edgfacs.append([])
                if len(self.edgdist[i]) == 1:
                    self._edgfacs[i].append(1.0)
                elif len(self.edgdist[i]) == 2:
                    totdist = self.edgdist[i][0] + self.edgdist[i][1]
                    self._edgfacs[i].append(self.edgdist[i][1]/totdist)
                    self._edgfacs[i].append(self.edgdist[i][0]/totdist)
                else:
                    ValueError(f'Too many panels associated to edge {i:d} of panel {self.pid:d}.')
        return self._edgfacs

    def edge_mu(self, mu: 'NDArray', display: bool = False):
        edgmu = []
        if display:
            print(f'edgpnls = {self.edgpnls}')
            print(f'edginds = {self.edginds}')
        for i in range(self.num):
            if len(self.edginds[i]) == 0:
                edgmu.append(None)
            else:
                edgmu.append(0.0)
                for ind, fac in zip(self.edginds[i], self.edgfacs[i]):
                    edgmu[i] += mu[ind]*fac
        return edgmu

    @property
    def edgpntl(self):
        if self._edgpntl is None:
            self._edgpntl = [self.crd.point_to_local(pnt) for pnt in self.edgpnts]
        return self._edgpntl

    def diff_mu(self, mu: 'NDArray', mug: 'NDArray') -> Vector2D:
        qjac = Vector2D(0.0, 0.0)
        jac = 0.0
        i = 0
        # print(f'{self.pid = }')
        for face in self.faces:
            # print(f'{i = }')
            i += 1
            qxJ = face.face_qxJ(mu, mug)
            qjac += qxJ
            jac += face.jac
        q = qjac/jac
        return q

    def diff_mu_old(self, mu: 'NDArray', display: bool = False):
        pnlmu = mu[self.ind]
        if display:
            print(f'pnlmu = {pnlmu}')
        edgmu = self.edge_mu(mu)
        if display:
            print(f'edgmu = {edgmu}')
        edgx = [pnt.x for pnt in self.edgpntl]
        edgy = [pnt.y for pnt in self.edgpntl]
        Js = 0.0
        qxJs = 0.0
        qyJs = 0.0
        # print(f'{self.pid = }')
        for i in range(self.num):
            # print(f'{i = }')
            a = i-1
            b = i
            mua = edgmu[a]
            mub = edgmu[b]
            if mua is not None and mub is not None:
                muc = pnlmu
                xa = edgx[a]
                xb = edgx[b]
                xc = 0.0
                ya = edgy[a]
                yb = edgy[b]
                yc = 0.0
                J = xa*yb - xa*yc - xb*ya + xb*yc + xc*ya - xc*yb
                # print(f'{J = }')
                Js += J
                ybc = yb - yc
                yca = yc - ya
                yab = ya - yb
                xcb = xc - xb
                xac = xa - xc
                xba = xb - xa
                # print(f'{self.edginds = }')
                # print(f'{mua = }, {mub = }, {muc = }')
                # print(f'{ybc = }, {yca = }, {yab = }')
                # print(f'{xcb = }, {xac = }, {xba = }')
                qxJ = mua*ybc + mub*yca + muc*yab
                qxJs += qxJ
                qyJ = mua*xcb + mub*xac + muc*xba
                qyJs += qyJ
                # print(f'{qxJ = }, {qyJ = }')
        if Js == 0.0:
            for i in range(self.num):
                a = i-1
                b = i
                mua = edgmu[a]
                if mua is None:
                    mua = pnlmu
                mub = edgmu[b]
                if mub is None:
                    mub = pnlmu
                muc = pnlmu
                xa = edgx[a]
                xb = edgx[b]
                xc = 0.0
                ya = edgy[a]
                yb = edgy[b]
                yc = 0.0
                J = xa*yb - xa*yc - xb*ya + xb*yc + xc*ya - xc*yb
                # print(f'{J = }')
                Js += J
                ybc = yb - yc
                yca = yc - ya
                yab = ya - yb
                xcb = xc - xb
                xac = xa - xc
                xba = xb - xa
                # print(f'{mua = }, {mub = }, {muc = }')
                # print(f'{ybc = }, {yca = }, {yab = }')
                # print(f'{xcb = }, {xac = }, {xba = }')
                qxJ = mua*ybc + mub*yca + muc*yab
                qxJs += qxJ
                qyJ = mua*xcb + mub*xac + muc*xba
                qyJs += qyJ
                # print(f'{qxJ = }, {qyJ = }')
        qx = -qxJs/Js
        qy = -qyJs/Js
        return qx, qy

    @property
    def grdpnls(self):
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
    def grdinds(self):
        if self._grdinds is None:
            self._grdinds = []
            for i in range(self.num):
                self._grdinds.append([])
                for pnl in self.grdpnls[i]:
                    self._grdinds[i].append(pnl.ind)
        return self._grdinds

    @property
    def grdfacs(self):
        if self._grdfacs is None:
            self._grdfacs = []
            for i, grd in enumerate(self.grds):
                pnldist = [(grd-pnl.pnto).return_magnitude() for pnl in self.grdpnls[i]]
                pnlinvd = [1/dist for dist in pnldist]
                suminvd = sum(pnlinvd)
                self._grdfacs.append([invd/suminvd for invd in pnlinvd])
        return self._grdfacs

    def grid_res(self, pnlres: 'NDArray'):
        grdres = []
        for i in range(self.num):
            grdres.append(0.0)
            for ind, fac in zip(self.grdinds[i], self.grdfacs[i]):
                grdres[i] += pnlres[ind]*fac
        return grdres

    def within_and_absz_ttol(self, pnts: Vector, ttol: float=0.1):
        shp = pnts.shape
        # pnts = pnts.reshape((-1, 1))
        rgcs = pnts - self.pnto
        wint = full(pnts.shape, False)
        absz = full(pnts.shape, float('inf'))
        for i in range(self.num):
            dirx = self.dirxab[0, i]
            diry = self.diryab[0, i]
            dirz = self.dirzab[0, i]
            xy1 = ones((pnts.shape[0], 3))
            xy1[:, 1] = rgcs.dot(dirx)
            xy1[:, 2] = rgcs.dot(diry)
            t123 = xy1@self.baryinv[i].transpose()
            mint = t123.min(axis=1)
            chk = mint > -ttol
            wint[chk] = True
            abszi = absolute(rgcs.dot(dirz))
            abszi[logical_not(chk)] = float('inf')
            absz = minimum(absz, abszi)
        wint = wint.reshape(shp)
        absz = absz.reshape(shp)
        return wint, absz

    def point_res(self, pnlres: 'NDArray', pnt: Vector, ttol: float=0.1):
        vecg = pnt - self.pnto
        gres = self.grid_res(pnlres)
        pres = pnlres[self.ind]
        r = 0.0
        for i in range(self.num):
            dirx = self.dirxab[0, i]
            diry = self.diryab[0, i]
            dirz = self.dirzab[0, i]
            vecl = Vector(vecg.dot(dirx), vecg.dot(diry), vecg.dot(dirz))
            ainv = self.baryinv[i]
            bmat = asarray([[1.0], [vecl.x], [vecl.y]])
            tmat = ainv*bmat
            to, ta, tb = tmat[0, 0], tmat[1, 0], tmat[2, 0]
            mint = min(to, ta, tb)
            if mint > -ttol:
                ro, ra, rb = pres, gres[i-1], gres[i]
                r = ro*to + ra*ta + rb*tb
                break
        return r

    def __repr__(self):
        return f'<pyapm.Panel {self.pid:d}>'

def angle_between_vectors(veca: Vector, vecb: Vector):
    unta = veca.to_unit()
    untb = vecb.to_unit()
    adb = unta.dot(untb)
    if adb > 1.0:
        adb = 1.0
    elif adb < -1.0:
        adb = -1.0
    return acos(adb)
