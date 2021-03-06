from typing import List, Dict
from py2md.classes import MDTable
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector, MatrixVector
from .panelsurface import PanelSurface
from .panelresult import PanelResult
from .surfaceload import SurfaceLoad
from ..tools.rigidbody import RigidBody

class SurfaceStructure(object):
    srfc: PanelSurface = None
    loads: Dict[str, SurfaceLoad] = None
    rref: Vector = None
    axis: str = None
    pntinds: Dict[int, int] = None
    rpnts: List[Vector] = None
    ks: List[Vector] = None
    gs: List[Vector] = None
    _pnts: MatrixVector = None
    _ypos: List[float] = None
    _zpos: List[float] = None
    _rbdy: RigidBody = None
    def __init__(self, srfc: PanelSurface):
        self.srfc = srfc
        self.update()
    def update(self):
        self.rref = self.srfc.point
        self.axis = 'y'
        self.pntinds = {}
        self.rpnts = []
        self.ks = []
        self.gs = []
    def set_axis(self, axis):
        self.axis = axis
    @property
    def strps(self):
        return self.srfc.strps
    @property
    def pnts(self):
        if self._pnts is None:
            numstrp = len(self.srfc.strps)
            self._pnts = zero_matrix_vector((2*numstrp, 1), dtype=float)
            for i, strp in enumerate(self.srfc.strps):
                inda = 2*i
                indb = inda+1
                self._pnts[inda, 0] = strp.prfa.point
                self._pnts[indb, 0] = strp.prfb.point
        return self._pnts
    @property
    def ypos(self):
        if self._ypos is None:
            self._ypos = self.pnts.y.transpose().tolist()[0]
        return self._ypos
    @property
    def zpos(self):
        if self._zpos is None:
            self._zpos = self.pnts.z.transpose().tolist()[0]
        return self._zpos
    @property
    def points_table(self):
        table = MDTable()
        table.add_column('#', 'd')
        table.add_column('x', '.5f')
        table.add_column('y', '.5f')
        table.add_column('z', '.5f')
        for i in range(self.pnts.shape[0]):
            pnt = self.pnts[i, 0]
            table.add_row([i, pnt.x, pnt.y, pnt.z])
        return table
    def add_load(self, pres: PanelResult, sf: float=1.0):
        if self.loads is None:
            self.loads = {}
        load = SurfaceLoad(pres, self, sf=sf)
        self.loads[pres.name] = load
        return load
    def add_section_constraint(self, sind: int, ksx: float=0.0, ksy: float=0.0, ksz: float=0.0,
                               gsx: float=0.0, gsy: float=0.0, gsz: float=0.0):
        self._rbdy = None
        for i, strp in enumerate(self.srfc.strps):
            inda = 2*i
            if strp.prfa is self.srfc.scts[sind]:
                self.pntinds[inda] = len(self.rpnts)
                self.rpnts.append(strp.prfa.point)
                self.ks.append(Vector(ksx, ksy, ksz))
                self.gs.append(Vector(gsx, gsy, gsz))
    @property
    def rbdy(self):
        if self._rbdy is None:
            self._rbdy = RigidBody(self.rref, self.rpnts, self.ks, self.gs)
        return self._rbdy
    # def section_constraints(self, sida: int, sidb: int):
    #     self.pnta = self.srfc.sects[sida].return_point(0.25)
    #     self.pntb = self.srfc.sects[sidb].return_point(0.25)
    #     x = (self.pnta.x+self.pntb.x)/2
    #     y = (self.pnta.y+self.pntb.y)/2
    #     z = (self.pnta.z+self.pntb.z)/2
    #     self.rref = Point(x, y, z)
