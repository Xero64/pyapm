from .grid import Grid

class WakePanel:
    ind: int = None
    grda: Grid = None
    grdb: Grid = None
    grdc: Grid = None

    def __init__(self, ind: int, grda: Grid, grdb: Grid, grdc: Grid) -> None:
        self.ind = ind
        self.grda = grda
        self.grdb = grdb
        self.grdc = grdc

class TrailingPanel:
    ind: int = None
    grda: Grid = None
    grdb: Grid = None

    def __init__(self, ind: int, grda: Grid, grdb: Grid) -> None:
        self.ind = ind
        self.grda = grda
        self.grdb = grdb

class TrailingVortex:
    grds: list[Grid] = None

    def __init__(self, grds: list[Grid]) -> None:
        self.grds = grds

class BoundVortex:
    grda: Grid = None
    grdb: Grid = None

    def __init__(self, grda: Grid, grdb: Grid) -> None:
        self.grda = grda
        self.grdb = grdb

class WakeStrip:
    bvtx: BoundVortex = None
    tvtxa: TrailingVortex = None
    tvtxb: TrailingVortex = None

class PanelWake:
    bvtxs: dict[int, BoundVortex] = None
    tvtxs: dict[int, TrailingVortex] = None

    def __init__(self, bvtxs: dict[int, BoundVortex], tvtxs: dict[int, TrailingVortex]) -> None:
        self.bvtxs = bvtxs
        self.tvtxs = tvtxs
