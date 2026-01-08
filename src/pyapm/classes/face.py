from typing import TYPE_CHECKING

from numpy import absolute, asarray, ones, zeros
from numpy.linalg import inv
from pygeom.geom2d import Vector2D
from pygeom.geom3d import Coordinate, Vector

from .grid import Grid, Vertex

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .edge import PanelEdge
    from .panel import Panel


class Face():
    grida: Grid = None
    gridb: Grid = None
    panel: 'Panel' = None
    ind: int = None
    edge: 'PanelEdge' = None
    _vertexa: Vertex = None
    _vertexb: Vertex = None
    _pointo: Vector = None
    _normal: Vector = None
    _jac: float = None
    _area: float = None
    _cord: Coordinate = None
    _pointa: Vector2D = None
    _pointb: Vector2D = None
    _pointc: Vector2D = None
    _xba: float = None
    _xac: float = None
    _xcb: float = None
    _yab: float = None
    _ybc: float = None
    _yca: float = None
    _baryinv: float = None
    _velv: Vector2D = None
    _vele: Vector2D = None
    _velp: Vector2D = None
    _indv: 'NDArray' = None
    _inde: 'NDArray' = None
    _indp: 'NDArray' = None

    def __init__(self, grida: Grid, gridb: Grid, panel: 'Panel') -> None:
        self.grida = grida
        self.gridb = gridb
        self.panel = panel

    @property
    def gridc(self) -> Vector:
        return self.panel.pnto

    @property
    def vertexa(self) -> Vertex:
        if self._vertexa is None:
            if isinstance(self.grida, Grid):
                for vertex in self.grida.vertices:
                    if self.panel in vertex.panels:
                        self._vertexa = vertex
                        break
        return self._vertexa

    @property
    def vertexb(self) -> Vertex:
        if self._vertexb is None:
            if isinstance(self.gridb, Grid):
                for vertex in self.gridb.vertices:
                    if self.panel in vertex.panels:
                        self._vertexb = vertex
                        break
        return self._vertexb

    @property
    def no_load(self) -> bool:
        no_load = False
        if self.panel.no_load:
            no_load = True
        return no_load

    @property
    def pointo(self) -> Vector:
        if self._pointo is None:
            self._pointo = (self.grida + self.gridb + self.gridc)/3
        return self._pointo

    def calc_normal_and_jac(self) -> tuple[Vector, float]:
        vecab = self.gridb - self.grida
        vecbc = self.gridc - self.gridb
        nrml, jac = vecab.cross(vecbc).to_unit(return_magnitude=True)
        return nrml, jac

    @property
    def normal(self) -> Vector:
        if self._normal is None:
            self._normal, self._jac = self.calc_normal_and_jac()
        return self._normal

    @property
    def jac(self) -> float:
        if self._jac is None:
            self._normal, self._jac = self.calc_normal_and_jac()
        return self._jac

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self.jac/2.0
        return self._area

    def set_dirl(self, vecl: Vector) -> None:
        dirz = self.normal
        vecy = dirz.cross(vecl)
        vecx = vecy.cross(dirz)
        self._cord = Coordinate(self.pointo, vecx, vecy)

    @property
    def cord(self) -> Coordinate:
        if self._cord is None:
            raise ValueError('PanelFace coordinate not set. Call set_dirl() first.')
        return self._cord

    @property
    def pointa(self) -> Vector2D:
        if self._pointa is None:
            veca = Vector.from_obj(self.grida)
            loca = self.cord.point_to_local(veca)
            self._pointa = Vector2D.from_obj(loca)
        return self._pointa

    @property
    def pointb(self) -> Vector2D:
        if self._pointb is None:
            vecb = Vector.from_obj(self.gridb)
            locb = self.cord.point_to_local(vecb)
            self._pointb = Vector2D.from_obj(locb)
        return self._pointb

    @property
    def pointc(self) -> Vector2D:
        if self._pointc is None:
            vecc = Vector.from_obj(self.gridc)
            locc = self.cord.point_to_local(vecc)
            self._pointc = Vector2D.from_obj(locc)
        return self._pointc

    @property
    def xba(self) -> float:
        if self._xba is None:
            self._xba = self.pointb.x - self.pointa.x
        return self._xba

    @property
    def xac(self) -> float:
        if self._xac is None:
            self._xac = self.pointa.x - self.pointc.x
        return self._xac

    @property
    def xcb(self) -> float:
        if self._xcb is None:
            self._xcb = self.pointc.x - self.pointb.x
        return self._xcb

    @property
    def yab(self) -> float:
        if self._yab is None:
            self._yab = self.pointa.y - self.pointb.y
        return self._yab

    @property
    def ybc(self) -> float:
        if self._ybc is None:
            self._ybc = self.pointb.y - self.pointc.y
        return self._ybc

    @property
    def yca(self) -> float:
        if self._yca is None:
            self._yca = self.pointc.y - self.pointa.y
        return self._yca

    def calc_vel_and_ind(self) -> None:
        if isinstance(self.vertexa, Vertex) and isinstance(self.vertexb, Vertex):
            self._velv = Vector2D.from_iter_xy([self.ybc, self.yca],
                                               [self.xcb, self.xac])/self.jac
            self._indv = asarray([self.vertexa.ind, self.vertexb.ind], dtype=int)
            self._inde = asarray([], dtype=int)
            self._vele = Vector2D.zeros(self._inde.shape)
        elif isinstance(self.vertexa, Vertex):
            self._velv = Vector2D(self.ybc, self.xcb)/self.jac
            self._indv = asarray([self.vertexa.ind], dtype=int)
            self._inde = asarray([self.edge.ind], dtype=int)
            self._vele = Vector2D(self.yca, self.xac)/self.jac
        elif isinstance(self.vertexb, Vertex):
            self._vele = Vector2D(self.ybc, self.xcb)/self.jac
            self._inde = asarray([self.edge.ind], dtype=int)
            self._indv = asarray([self.vertexb.ind], dtype=int)
            self._velv = Vector2D(self.yca, self.xac)/self.jac
        self._velp = Vector2D(self.yab, self.xba)/self.jac
        self._indp = asarray([self.panel.ind], dtype=int)

    @property
    def velv(self) -> Vector2D:
        if self._velv is None:
            self.calc_vel_and_ind()
        return self._velv

    @property
    def vele(self) -> Vector2D:
        if self._vele is None:
            self.calc_vel_and_ind()
        return self._vele

    @property
    def velp(self) -> Vector2D:
        if self._velp is None:
            self.calc_vel_and_ind()
        return self._velp

    @property
    def indv(self) -> 'NDArray':
        if self._indv is None:
            self.calc_vel_and_ind()
        return self._indv

    @property
    def inde(self) -> 'NDArray':
        if self._inde is None:
            self.calc_vel_and_ind()
        return self._inde

    @property
    def indp(self) -> 'NDArray':
        if self._indp is None:
            self.calc_vel_and_ind()
        return self._indp

    def face_qxJ(self, mud: 'NDArray', mue: 'NDArray', muv: 'NDArray') -> Vector2D:
        if isinstance(self.vertexa, Vertex) and isinstance(self.vertexb, Vertex):
            mua = muv[self.vertexa.ind]
            mub = muv[self.vertexb.ind]
        elif isinstance(self.vertexa, Vertex):
            mua = muv[self.vertexa.ind]
            mub = mue[self.edge.ind]
        elif isinstance(self.vertexb, Vertex):
            mua = mue[self.edge.ind]
            mub = muv[self.vertexb.ind]
        else:
            raise ValueError('Face has no vertex assigned for velocity calculation.')
        muc = mud[self.panel.ind]
        qxJ = mua*self.ybc + mub*self.yca + muc*self.yab
        qyJ = mua*self.xcb + mub*self.xac + muc*self.xba
        return Vector2D(qxJ, qyJ)

    @property
    def baryinv(self) -> 'NDArray':
        if self._baryinv is None:
            amat = zeros((3, 3))
            amat[0, :] = 1.0
            amat[1, 0] = self.pointa.x
            amat[2, 0] = self.pointa.y
            amat[1, 1] = self.pointb.x
            amat[2, 1] = self.pointb.y
            amat[1, 2] = self.pointc.x
            amat[2, 2] = self.pointc.y
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

    def __repr__(self) -> str:
        return f'Face(grida={self.grida}, gridb={self.gridb}, gridc={self.gridc}, panel={self.panel})'
