from pygeom.geom3d import Vector


class ConstantGrid(Vector):
    gid: int | None = None
    ind: int | None = None

    def __init__(self, gid: int, x: float, y: float, z: float) -> None:
        super().__init__(x, y, z)
        self.gid = gid
        self.pnls = set()

    def __hash__(self):
        return hash(tuple([self.gid, self.x, self.y, self.z]))

    def __str__(self) -> str:
        return f'Grid {self.gid:d}, {super().__str__():}'

    def __repr__(self) -> str:
        return f'<Grid {self.gid:d}>'

    def __format__(self, frm: str) -> str:
        return f'Grid {self.gid:d}, {super().__format__(frm):}'
