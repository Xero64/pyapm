from .poly import Poly
from .grid import Grid
from .horseshoe import HorseShoe
from typing import List
from pygeom.geom3d import Vector, Coordinate, ihat, khat
from math import sqrt
from numpy.matlib import matrix

oor2 = 1/sqrt(2.0)

class Panel(Poly):
    pid: int = None
    gids: List[int] = None
    ind: int = None
    noload: bool = None
    _crd: Coordinate = None
    _hsvs: List[HorseShoe] = None
    # _grdarea: List[float] = None
    # _grdinva: List[float] = None
    _grdlocs: List[Vector] = None
    _edgpnls: List[object] = None
    _edginds: List[List[int]] = None
    _edgpnts: List[Vector] = None
    _edgdist: List[float] = None
    _edgfacs: List[float] = None
    _edgpntl: List[Vector] = None
    # _grdpnls: List[List[object]] = None
    # _grdpind: List[List[int]] = None
    # _grddist: List[List[float]] = None
    # _grdfacs: List[List[float]] = None
    def __init__(self, pid: int, gids: List[int]):
        self.pid = pid
        self.gids = gids
        self.noload = False
    def set_grids(self, grds: List[Grid]):
        super(Panel, self).__init__(grds)
        for grd in self.grds:
            grd.pnls.append(self)
    def set_index(self, ind: int):
        self.ind = ind
    def set_horseshoes(self, diro: Vector):
        self._hsvs = []
        for b in range(self.num):
            a = b+1
            if a == self.num:
                a = 0
            if self.grds[a].te and self.grds[b].te:
                self._hsvs.append(HorseShoe(self.grds[a], self.grds[b], diro, self.ind))
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
            for b in range(self.num):
                a = b+1
                if a == self.num:
                    a = 0
                if self.grds[a].te and self.grds[b].te:
                    self._hsvs.append(HorseShoe(self.grds[a], self.grds[b], ihat, self.ind))
        return self._hsvs
    @property
    def area(self):
        if self.noload:
            return 0.0
        else:
            return super(Panel, self).area
    # @property
    # def grdarea(self):
    #     if self._grdarea is None:
    #         self._grdarea = []
    #         for grd in self.grds:
    #             self._grdarea.append(0.0)
    #             for edg in self.edgs:
    #                 if edg.grda == grd:
    #                     self._grdarea[-1] += edg.area/2
    #     return self._grdarea
    # @property
    # def grdinva(self):
    #     if self._grdinva is None:
    #         self._grdinva = []
    #         for ga in self.grdarea:
    #             self._grdinva.append(1.0/ga)                
    #     return self._grdinva
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
            for a, grda in enumerate(self.grds):
                self.edgpnls.append([])
                b = a + 1
                if b == self.num:
                    b = 0
                grdb = self.grds[b]
                if grda.te and grdb.te:
                    self._edgpnls[a].append(self)
                    continue
                for pnl in grda.pnls:
                    for grd in pnl.grds:
                        if grd == grdb:
                            self._edgpnls[a].append(pnl)
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
            self._edgpnts = [(edg.grda+edg.grdb)/2 for edg in self.edgs]
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


    # @property
    # def grdpnls(self):
    #     if self._grdpnls is None:
    #         self._grdpnls = []
    #         for i in range(self.num):
    #             self._grdpnls.append([self])
    #             a = i
    #             b = a - 1
    #             if self.edgpnls[a] is not None:
    #                 self.grdpnls[i].append(self.edgpnls[a])
    #             if self.edgpnls[b] is not None:
    #                 self.grdpnls[i].append(self.edgpnls[b])
    #     return self._grdpnls
    # @property
    # def grdpind(self):
    #     if self._grdpind is None:
    #         self._grdpind = []
    #         for i in range(self.num):
    #             self._grdpind.append([pnl.ind for pnl in self.grdpnls[i]])
    #     return self._grdpind
    # @property
    # def grddist(self):
    #     if self._grddist is None:
    #         self._grddist = []
    #         for i, grd in enumerate(self.grds):
    #             self._grddist.append([(pnl.pnto-grd).return_magnitude() for pnl in self.grdpnls[i]])
    #     return self._grddist
    def __repr__(self):
        return f'<pyapm.Panel {self.pid:d}>'
