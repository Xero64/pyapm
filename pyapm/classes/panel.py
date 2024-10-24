from typing import TYPE_CHECKING

from numpy import (absolute, acos, asarray, full, logical_not, minimum, ones,
                   pi, sqrt)
from pygeom.geom3d import IHAT, KHAT, Coordinate, Vector

from .dirichletpoly import DirichletPoly
from .grid import Grid
from .horseshoedoublet import HorseshoeDoublet

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .panelsection import PanelSection
    from .panelsheet import PanelSheet
    from .panelsurface import PanelSurface

oor2 = 1/sqrt(2.0)
angtol = pi/4


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
        if self.grp is not None:
            noload = self.grp.noload
        return noload

    @property
    def nohsv(self):
        nohsv = False
        if self.sht is not None:
            nohsv = self.sht.nohsv
        if self.sct is not None:
            nohsv = self.sct.nohsv
        if self.grp is not None:
            nohsv = self.grp.nohsv
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
            diry = vecy.to_unit()
            dirx = diry.cross(dirz).to_unit()
            pntc = self.pnto
            self._crd = Coordinate(pntc, dirx, diry)
        return self._crd

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

    def diff_mu(self, mu: 'NDArray', display: bool = False):
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
        for i in range(self.num):
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
                Js += J
                ybc = yb-yc
                yca = yc-ya
                yab = ya-yb
                xcb = xc-xb
                xac = xa-xc
                xba = xb-xa
                qxJ = mua*ybc + mub*yca + muc*yab
                qxJs += qxJ
                qyJ = mua*xcb + mub*xac + muc*xba
                qyJs += qyJ
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
                Js += J
                ybc = yb-yc
                yca = yc-ya
                yab = ya-yb
                xcb = xc-xb
                xac = xa-xc
                xba = xb-xa
                qxJ = mua*ybc + mub*yca + muc*yab
                qxJs += qxJ
                qyJ = mua*xcb + mub*xac + muc*xba
                qyJs += qyJ
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
