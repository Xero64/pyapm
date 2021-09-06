from math import sqrt, pi, acos
from typing import List, Tuple
from numpy.matlib import matrix#, absolute, full, minimum, logical_not, ones
from pygeom.geom3d import Vector#, khat, Coordinate
# from pygeom.matrix3d import MatrixVector
from .grid import Grid, GridNormal
from .triangle import Triangle
from .trailingdoubletpanel import TrailingDoubletPanel

# oor2 = 1/sqrt(2.0)
angtol = pi/4

class TrianglePanel(Triangle):
    pid: int = None
    ind: int = None
    grp: object = None
    sht: object = None
    sct: object = None
    srfc: object = None
    tdps: List[TrailingDoubletPanel] = None
    _indd: Tuple[int] = None
    _nrma: GridNormal = None
    _nrmb: GridNormal = None
    _nrmc: GridNormal = None
    _nrms: List[GridNormal] = None
    _inds: Tuple[int] = None
    _pnto: Vector = None
    _edgpnls: List['TrianglePanel'] = None
    # _grdlocs: List[Vector] = None
    # _edgpnls: List[object] = None
    # _edginds: List[List[int]] = None
    # _edgpnts: List[Vector] = None
    # _edgdist: List[float] = None
    # _edgfacs: List[float] = None
    # _edgpntl: List[Vector] = None
    # _edglens: List[float] = None
    # _grdpnls: List[List[object]] = None
    # _grdinds: List[List[int]] = None
    # _grdfacs: List[List[float]] = None
    def __init__(self, pid: int, grda: Grid, grdb: Grid, grdc: Grid) -> None:
        self.pid = pid
        super().__init__(grda, grdb, grdc)
        grda.pnls.append(self)
        grdb.pnls.append(self)
        grdc.pnls.append(self)
        self.tdps = []
    @property
    def indd(self) -> Tuple[int]:
        if self._indd is None:
            self._indd = (self.grda.ind, self.grdb.ind, self.grdc.ind)
        return self._indd
    def set_index(self, ind: int) -> None:
        self.ind = ind
    def set_grid_normals(self, nrma: GridNormal, nrmb: GridNormal, nrmc: GridNormal):
        self._nrma = nrma
        self._nrmb = nrmb
        self._nrmc = nrmc
    @property
    def nrma(self) -> GridNormal:
        if self._nrma is None:
            self._nrma = GridNormal(self.grda, self.dirz.x, self.dirz.y, self.dirz.z)
        return self._nrma
    @property
    def nrmb(self) -> GridNormal:
        if self._nrmb is None:
            self._nrmb = GridNormal(self.grdb, self.dirz.x, self.dirz.y, self.dirz.z)
        return self._nrmb
    @property
    def nrmc(self) -> GridNormal:
        if self._nrmc is None:
            self._nrmc = GridNormal(self.grdc, self.dirz.x, self.dirz.y, self.dirz.z)
        return self._nrmc
    @property
    def nrms(self) -> List[GridNormal]:
        if self._nrms is None:
            self._nrms = [self.nrma, self.nrmb, self.nrmc]
        return self._nrms
    @property
    def inds(self) -> Tuple[int]:
        if self._inds is None:
            self._inds = (self.nrma.ind, self.nrmb.ind, self.nrmc.ind)
        return self._inds
    @property
    def pnto(self):
        if self._pnto is None:
            self._pnto = sum(self.grds)/3
        return self._pnto
    def dndl(self, gain: float, hvec: Vector):
        dndla = gain*(hvec**self.nrma)
        dndlb = gain*(hvec**self.nrmb)
        dndlc = gain*(hvec**self.nrmc)
        return dndla, dndlb, dndlc
    # @property
    # def noload(self) -> bool:
    #     noload = False
    #     if self.sht is not None:
    #         noload = self.sht.noload
    #     if self.sct is not None:
    #         noload = self.sct.noload
    #     if self.grp is not None:
    #         noload = self.grp.noload
    #     return noload
    # @property
    # def notdbl(self) -> bool:
    #     nohsv = False
    #     if self.sht is not None:
    #         nohsv = self.sht.nohsv
    #     if self.sct is not None:
    #         nohsv = self.sct.nohsv
    #     if self.grp is not None:
    #         nohsv = self.grp.nohsv
    #     return nohsv
    def mesh_trailing_doublet_panels(self, pid: int, diro: Vector) -> int:
        if self.grda.te and self.grdb.te:
            tdp = TrailingDoubletPanel(pid, self.grdb, self.grda, diro)
            self.tdps.append(tdp)
            pid += 1
        if self.grdb.te and self.grdc.te:
            tdp = TrailingDoubletPanel(pid, self.grdc, self.grdb, diro)
            self.tdps.append(tdp)
            pid += 1
        if self.grdc.te and self.grda.te:
            tdp = TrailingDoubletPanel(pid, self.grda, self.grdc, diro)
            self.tdps.append(tdp)
            pid += 1
        return pid
    def check_panel(self, pnl: 'TrianglePanel'):
        grpchk = False
        if pnl.grp is not None and self.grp is not None:
            grpchk = pnl.grp == self.grp
        srfchk = False
        if pnl.srfc is not None and self.srfc is not None:
            srfchk = pnl.srfc == self.srfc
        typchk = False
        if srfchk:
            if pnl.sht is not None and self.sht is not None:
                typchk = True
            elif pnl.sct is not None and self.sct is not None:
                typchk = True
        return grpchk, srfchk, typchk
    def check_angle(self, pnl: 'TrianglePanel') -> bool:
        ang = angle_between_vectors(pnl.dirz, self.dirz)
        angchk = abs(ang) < angtol
        return angchk
    def check_edge(self, pnl: 'TrianglePanel', grda: Grid, grdb: Grid) -> bool:
        edgchk = False
        if grda in pnl.grds and grdb in pnl.grds:
            edgchk = True
        return edgchk
    # @property
    # def crd(self) -> Coordinate:
    #     if self._crd is None:
    #         dirz = self.nrm
    #         vecy = dirz**ihat
    #         magy = vecy.return_magnitude()
    #         if magy < oor2:
    #             vecy = dirz**khat
    #         diry = vecy.to_unit()
    #         dirx = (diry**dirz).to_unit()
    #         pntc = self.pnto.to_point()
    #         self._crd = Coordinate(pntc, dirx, diry, dirz)
    #     return self._crd
    # @property
    # def wetarea(self):
    #     if self.noload:
    #         wetarea = 0.0
    #     else:
    #         wetarea = self.area
    #     return wetarea
    # @property
    # def grdlocs(self):
    #     if self._grdlocs is None:
    #         self._grdlocs = []
    #         for grd in self.grds:
    #             self._grdlocs.append(self.crd.point_to_local(grd))
    #     return self._grdlocs
    @property
    def edgpnls(self):
        if self._edgpnls is None:
            self._edgpnls = []
            for i in range(3):
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
    # @property
    # def edginds(self):
    #     if self._edginds is None:
    #         self._edginds = []
    #         for i in range(self.num):
    #             self._edginds.append([])
    #             for pnl in self.edgpnls[i]:
    #                 self._edginds[i].append(pnl.ind)
    #     return self._edginds
    # @property
    # def edgpnts(self):
    #     if self._edgpnts is None:
    #         self._edgpnts = []
    #         for i in range(self.num):
    #             grda = self.grds[i-1]
    #             grdb = self.grds[i]
    #             if len(self.edgpnls[i]) == 2:
    #                 pnla = self.edgpnls[i][-1]
    #                 pnlb = self.edgpnls[i][-2]
    #                 dirx = (grdb-grda).to_unit()
    #                 dirza = pnla.nrm
    #                 dirzb = pnlb.nrm
    #                 dirya = dirza**dirx
    #                 diryb = dirzb**dirx
    #                 veca = pnla.pnto - grda
    #                 vecb = pnlb.pnto - grda
    #                 xa = veca*dirx
    #                 xb = vecb*dirx
    #                 ya = veca*dirya
    #                 yb = vecb*diryb
    #                 xc = xa - (xb-xa)/(yb-ya)*ya
    #                 self._edgpnts.append(grda + dirx*xc)
    #             else:
    #                 self._edgpnts.append((grda + grdb)/2)
    #     return self._edgpnts
    # @property
    # def edgdist(self):
    #     if self._edgdist is None:
    #         self._edgdist = []
    #         for i, pnt in enumerate(self.edgpnts):
    #             self._edgdist.append([])
    #             for pnl in self.edgpnls[i]:
    #                 self._edgdist[i].append((pnt-pnl.pnto).return_magnitude())
    #     return self._edgdist
    # @property
    # def edgfacs(self):
    #     if self._edgfacs is None:
    #         self._edgfacs = []
    #         for i in range(self.num):
    #             self._edgfacs.append([])
    #             if len(self.edgdist[i]) == 1:
    #                 self._edgfacs[i].append(1.0)
    #             elif len(self.edgdist[i]) == 2:
    #                 totdist = self.edgdist[i][0] + self.edgdist[i][1]
    #                 self._edgfacs[i].append(self.edgdist[i][1]/totdist)
    #                 self._edgfacs[i].append(self.edgdist[i][0]/totdist)
    #             else:
    #                 ValueError(f'Too many panels associated to edge {i:d} of panel {self.pid:d}.')
    #     return self._edgfacs
    # def edge_mu(self, mu: matrix):
    #     edgmu = []
    #     for i in range(self.num):
    #         if len(self.edginds[i]) == 0:
    #             edgmu.append(None)
    #         else:
    #             edgmu.append(0.0)
    #             for ind, fac in zip(self.edginds[i], self.edgfacs[i]):
    #                 edgmu[i] += mu[ind, 0]*fac
    #     return edgmu
    # @property
    # def edgpntl(self):
    #     if self._edgpntl is None:
    #         self._edgpntl = [self.crd.point_to_local(pnt) for pnt in self.edgpnts]
    #     return self._edgpntl
    def diff_mu(self, mu: matrix):
        mua = mu[self.indd[0], 0]
        mub = mu[self.indd[1], 0]
        muc = mu[self.indd[2], 0]
        J = self.jac
        ybc = self.pntb.y - self.pntc.y
        yca = self.pntc.y - self.pnta.y
        yab = self.pnta.y - self.pntb.y
        xcb = self.pntc.x - self.pntb.x
        xac = self.pnta.x - self.pntc.x
        xba = self.pntb.x - self.pnta.x
        qxJ = mua*ybc + mub*yca + muc*yab
        qyJ = mua*xcb + mub*xac + muc*xba
        qx = -qxJ/J
        qy = -qyJ/J
        return qx, qy
    # @property
    # def grdpnls(self):
    #     if self._grdpnls is None:
    #         self._grdpnls = []
    #         for i, grd in enumerate(self.grds):
    #             self._grdpnls.append([])
    #             for pnl in grd.pnls:
    #                 grpchk, srfchk, typchk = self.check_panel(pnl)
    #                 if grpchk:
    #                     angchk = self.check_angle(pnl)
    #                     if angchk:
    #                         self._grdpnls[i].append(pnl)
    #                 elif srfchk and typchk:
    #                     if grd.te:
    #                         angchk = self.check_angle(pnl)
    #                         if angchk:
    #                             self._grdpnls[i].append(pnl)
    #                     else:
    #                         self._grdpnls[i].append(pnl)
    #     return self._grdpnls
    # @property
    # def grdinds(self):
    #     if self._grdinds is None:
    #         self._grdinds = []
    #         for i in range(self.num):
    #             self._grdinds.append([])
    #             for pnl in self.grdpnls[i]:
    #                 self._grdinds[i].append(pnl.ind)
    #     return self._grdinds
    # @property
    # def grdfacs(self):
    #     if self._grdfacs is None:
    #         self._grdfacs = []
    #         for i, grd in enumerate(self.grds):
    #             pnldist = [(grd-pnl.pnto).return_magnitude() for pnl in self.grdpnls[i]]
    #             pnlinvd = [1/dist for dist in pnldist]
    #             suminvd = sum(pnlinvd)
    #             self._grdfacs.append([invd/suminvd for invd in pnlinvd])
    #     return self._grdfacs
    # def grid_res(self, pnlres: matrix):
    #     grdres = []
    #     for i in range(self.num):
    #         grdres.append(0.0)
    #         for ind, fac in zip(self.grdinds[i], self.grdfacs[i]):
    #             grdres[i] += pnlres[ind, 0]*fac
    #     return grdres
    # def within_and_absz_ttol(self, pnts: MatrixVector, ttol: float=0.1):
    #     shp = pnts.shape
    #     pnts = pnts.reshape((-1, 1))
    #     rgcs = pnts-self.pnto
    #     wint = full(pnts.shape, False)
    #     absz = full(pnts.shape, float('inf'))
    #     for i in range(self.num):
    #         dirx = self.dirxab[0, i]
    #         diry = self.diryab[0, i]
    #         dirz = self.dirzab[0, i]
    #         xy1 = ones((pnts.shape[0], 3), dtype=float)
    #         xy1[:, 1] = rgcs*dirx
    #         xy1[:, 2] = rgcs*diry
    #         t123 = xy1*self.baryinv[i].transpose()
    #         mint = t123.min(axis=1)
    #         chk = mint > -ttol
    #         wint[chk] = True
    #         abszi = absolute(rgcs*dirz)
    #         abszi[logical_not(chk)] = float('inf')
    #         absz = minimum(absz, abszi)
    #     wint = wint.reshape(shp)
    #     absz = absz.reshape(shp)
    #     return wint, absz
    # def point_res(self, pnlres: matrix, pnt: Vector, ttol: float=0.1):
    #     vecg = pnt - self.pnto
    #     gres = self.grid_res(pnlres)
    #     pres = pnlres[self.ind, 0]
    #     r = 0.0
    #     for i in range(self.num):
    #         dirx = self.dirxab[0, i]
    #         diry = self.diryab[0, i]
    #         dirz = self.dirzab[0, i]
    #         vecl = Vector(vecg*dirx, vecg*diry, vecg*dirz)
    #         ainv = self.baryinv[i]
    #         bmat = matrix([[1.0], [vecl.x], [vecl.y]])
    #         tmat = ainv*bmat
    #         to, ta, tb = tmat[0, 0], tmat[1, 0], tmat[2, 0]
    #         mint = min(to, ta, tb)
    #         if mint > -ttol:
    #             ro, ra, rb = pres, gres[i-1], gres[i]
    #             r = ro*to + ra*ta + rb*tb
    #             break
    #     return r
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
