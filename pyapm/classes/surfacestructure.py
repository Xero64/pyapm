from typing import TYPE_CHECKING

from py2md.classes import MDTable
from pygeom.geom3d import Vector

from ..tools.rigidbody import RigidBody
from .panelresult import PanelResult
from .surfaceload import SurfaceLoad

if TYPE_CHECKING:
    from .panelstrip import PanelStrip
    from .panelsurface import PanelSurface


class SurfaceStructure():
    srfc: 'PanelSurface' = None
    loads: dict[str, SurfaceLoad] = None
    rref: Vector = None
    axis: str = None
    pntinds: dict[int, int] = None
    rpnts: list[Vector] = None
    ks: list[Vector] = None
    gs: list[Vector] = None
    _pnts: Vector = None
    _ypos: list[float] = None
    _zpos: list[float] = None
    _rbdy: RigidBody = None

    def __init__(self, srfc: 'PanelSurface') -> None:
        self.srfc = srfc
        self.update()

    def update(self) -> None:
        self.rref = self.srfc.point
        self.set_axis()
        self.pntinds = {}
        self.rpnts = []
        self.ks = []
        self.gs = []

    def set_axis(self, axis: str='y') -> None:
        self.axis = axis

    @property
    def strps(self) -> list['PanelStrip']:
        return self.srfc.strps

    @property
    def pnts(self) -> Vector:
        if self._pnts is None:
            numstrp = len(self.srfc.strps)
            self._pnts = Vector.zeros((2*numstrp, 1))
            for i, strp in enumerate(self.srfc.strps):
                inda = 2*i
                indb = inda + 1
                self._pnts[inda, 0] = strp.prfa.point
                self._pnts[indb, 0] = strp.prfb.point
        return self._pnts

    @property
    def ypos(self) -> list[float]:
        if self._ypos is None:
            self._ypos = self.pnts.y.transpose().tolist()[0]
        return self._ypos

    @property
    def zpos(self) -> list[float]:
        if self._zpos is None:
            self._zpos = self.pnts.z.transpose().tolist()[0]
        return self._zpos

    @property
    def points_table(self) -> MDTable:
        table = MDTable()
        table.add_column('#', 'd')
        table.add_column('x', '.5f')
        table.add_column('y', '.5f')
        table.add_column('z', '.5f')
        for i in range(self.pnts.shape[0]):
            pnt = self.pnts[i, 0]
            table.add_row([i, pnt.x, pnt.y, pnt.z])
        return table

    def add_load(self, pres: 'PanelResult', sf: float=1.0) -> SurfaceLoad:
        if self.loads is None:
            self.loads = {}
        load = SurfaceLoad(pres, self, sf=sf)
        self.loads[pres.name] = load
        return load

    def add_section_constraint(self, sind: int, ksx: float=0.0, ksy: float=0.0,
                               ksz: float=0.0, gsx: float=0.0, gsy: float=0.0,
                               gsz: float=0.0) -> None:
        self._rbdy = None
        for i, strp in enumerate(self.srfc.strps):
            inda = 2*i
            if strp.prfa is self.srfc.scts[sind]:
                self.pntinds[inda] = len(self.rpnts)
                self.rpnts.append(strp.prfa.point)
                self.ks.append(Vector(ksx, ksy, ksz))
                self.gs.append(Vector(gsx, gsy, gsz))

    @property
    def rbdy(self) -> RigidBody:
        if self._rbdy is None:
            self._rbdy = RigidBody(self.rref, self.rpnts, self.ks, self.gs)
        return self._rbdy
