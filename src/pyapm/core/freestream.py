from typing import TYPE_CHECKING

from numpy import arcsin, arctan2, cos, radians, sin
from pygeom.geom3d import Transform, Vector

from .flow import Flow

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FreeStream():
    vfs: Vector = None
    ofs: Vector = None
    _speed: float = None
    _alpha: float = None
    _beta: float = None
    _atfm: Transform = None
    _wtfm: Transform = None
    _pqr: Vector = None

    def __init__(self, vfs: Vector, ofs = Vector(0.0, 0.0, 0.0)) -> None:
        self.vfs = vfs
        self.ofs = ofs

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def calculate_phi(self, pnts: Vector) -> 'NDArray':
        return pnts.dot(self.vfs)

    def calculate_vel(self, pnts: Vector) -> Vector:
        vel = Vector.zeros(pnts.shape)
        vel[:, ] = self.vfs - pnts.cross(self.ofs)
        return vel

    def calculate_flow(self, pnts: Vector) -> 'Flow':
        phi = self.calculate_phi(pnts)
        vel = self.calculate_vel(pnts)
        return Flow(phi, vel)

    def calculate_dynp(self, rho: float) -> float:
        return 0.5 * rho * self.speed**2

    @property
    def speed(self) -> float:
        if self._speed is None:
            self._speed = self.vfs.return_magnitude()
        return self._speed

    @property
    def alpha(self) -> float:
        if self._alpha is None:
            self._alpha = arctan2(self.vfs.z, self.vfs.x)
        return self._alpha

    @property
    def beta(self) -> float:
        if self._beta is None:
            self._beta = arcsin(self.vfs.y / self.speed)
        return self._beta

    @property
    def atfm(self) -> Transform:
        if self._atfm is None:
            cosal, sinal = cos(self.alpha), sin(self.alpha)
            cosbt, sinbt = cos(self.beta), sin(self.beta)
            dirx = Vector(cosbt*cosal, -sinbt, cosbt*sinal)
            diry = Vector(sinbt*cosal, cosbt, sinbt*sinal)
            self._atfm = Transform(dirx, diry)
        return self._atfm

    @property
    def wtfm(self) -> Transform:
        if self._wtfm is None:
            self._wtfm = Transform(-self.atfm.dirx, self.atfm.diry)
        return self._wtfm

    @property
    def pqr(self) -> Vector:
        if self._pqr is None:
            self._pqr = self.wtfm.vector_to_local(self.ofs)
        return self._pqr

    @classmethod
    def from_vab_deg(cls, speed: float, alpha: float, beta: float) -> 'FreeStream':
        alrad = radians(alpha)
        btrad = radians(beta)
        return cls.from_vab_rad(speed, alrad, btrad)

    @classmethod
    def from_vab_rad(cls, speed: float, alpha: float, beta: float) -> 'FreeStream':
        cosal = cos(alpha)
        sinal = sin(alpha)
        cosbt = cos(beta)
        sinbt = sin(beta)
        vfs = Vector(speed * cosal * cosbt, speed * sinbt, speed * sinal * cosbt)
        return FreeStream(vfs)
