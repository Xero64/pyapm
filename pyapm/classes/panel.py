from math import sqrt, acos, pi
from typing import List
from numpy.matlib import matrix
from pygeom.geom3d import Vector, Coordinate, ihat, khat
from .grid import Grid
from .dirichletpoly import DirichletPoly
from .horseshoedoublet import HorseshoeDoublet

oor2 = 1/sqrt(2.0)
angtol = pi/4

class Panel(DirichletPoly):
    pid: int = None
    gids: List[int] = None
    ind: int = None
    noload: bool = None
    grp: object = None
    sht: object = None
    sct: object = None
    srfc: object = None
    _crd: Coordinate = None
    _hsvs: List[HorseshoeDoublet] = None
    _grdlocs: List[Vector] = None
    _edgpnls: List[object] = None
    _edginds: List[List[int]] = None
    _edgpnts: List[Vector] = None
    _edgdist: List[float] = None
    _edgfacs: List[float] = None
    _edgpntl: List[Vector] = None
    _edglens: List[float] = None
    _grdpnls: List[List[object]] = None
    _grdinds: List[List[int]] = None
    _grdfacs: List[List[float]] = None
    def __init__(self, pid: int, grds: List[Grid]):
        super().__init__(grds)
        self.pid = pid
        self.noload = False
        for grd in self.grds:
            grd.pnls.append(self)
    def set_index(self, ind: int):
        self.ind = ind
    def set_horseshoes(self, diro: Vector):
        self._hsvs = []
        for i in range(self.num):
            grda = self.grds[i]
            grdb = self.grds[i-1]
            if grda.te and grdb.te:
                self._hsvs.append(HorseshoeDoublet(grda, grdb, diro, self.ind))
    def check_panel(self, pnl):
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
    def check_angle(self, pnl):
        ang = angle_between_vectors(pnl.crd.dirz, self.crd.dirz)
        angchk = abs(ang) < angtol
        return angchk
    def check_edge(self, pnl, grda, grdb):
        edgchk = False
        if grda in pnl.grds and grdb in pnl.grds:
            edgchk = True
        return edgchk
    @property
    def crd(self) -> Coordinate:
        if self._crd is None:
            dirz = self.nrm
            vecy = dirz**ihat
            magy = vecy.return_magnitude()
            if magy < oor2:
                vecy = dirz**khat
            diry = vecy.to_unit()
            dirx = (diry**dirz).to_unit()
            pntc = self.pnto.to_point()
            self._crd = Coordinate(pntc, dirx, diry, dirz)
        return self._crd
    @property
    def hsvs(self):
        if self._hsvs is None:
            self._hsvs = []
            for i in range(self.num):
                grda = self.grds[i]
                grdb = self.grds[i-1]
                if grda.te and grdb.te:
                    self._hsvs.append(HorseshoeDoublet(grda, grdb, ihat, self.ind))
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
                self._edgpnls.append([])
                for pnl in grda.pnls:
                    if pnl is not self:
                        edgchk = self.check_edge(pnl, grda, grdb)
                        if edgchk:
                            _, srfchk, _ = self.check_panel(pnl)
                            if srfchk:
                                if not grda.te and not grdb.te:
                                    angchk = self.check_angle(pnl)
                                    if angchk:
                                        self._edgpnls[i].append(pnl)
                                else:
                                    self._edgpnls[i].append(pnl)
                            else:
                                angchk = self.check_angle(pnl)
                                if angchk:
                                    self._edgpnls[i].append(pnl)
                if grda.te and grdb.te:
                    self._edgpnls[i].append(self)
                elif len(self._edgpnls[i]) > 0:
                    self._edgpnls[i].append(self)
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
                    dirya = dirza**dirx
                    diryb = dirzb**dirx
                    veca = pnla.pnto - grda
                    vecb = pnlb.pnto - grda
                    xa = veca*dirx
                    xb = vecb*dirx
                    ya = veca*dirya
                    yb = vecb*diryb
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
    def edge_mu(self, mu: matrix):
        edgmu = []
        for i in range(self.num):
            if len(self.edginds[i]) == 0:
                edgmu.append(None)
            else:
                edgmu.append(0.0)
                for ind, fac in zip(self.edginds[i], self.edgfacs[i]):
                    edgmu[i] += mu[ind, 0]*fac
        return edgmu
    @property
    def edgpntl(self):
        if self._edgpntl is None:
            self._edgpntl = [self.crd.point_to_local(pnt) for pnt in self.edgpnts]
        return self._edgpntl
    def diff_mu(self, mu: matrix):
        pnlmu = mu[self.ind, 0]
        edgmu = self.edge_mu(mu)
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
    def grid_res(self, pnlres: matrix):
        grdres = []
        for i in range(self.num):
            grdres.append(0.0)
            for ind, fac in zip(self.grdinds[i], self.grdfacs[i]):
                grdres[i] += pnlres[ind, 0]*fac
        return grdres
    def __repr__(self):
        return f'<pyapm.Panel {self.pid:d}>'

def angle_between_vectors(veca: Vector, vecb: Vector):
    unta = veca.to_unit()
    untb = vecb.to_unit()
    adb = unta*untb
    if adb > 1.0:
        adb = 1.0
    elif adb < -1.0:
        adb = -1.0
    return acos(adb)
