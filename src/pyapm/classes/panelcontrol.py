from typing import TYPE_CHECKING, Any

from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from ..classes.panel import Panel


class PanelControl():
    name: str = None
    posgain: float = None
    neggain: float = None
    xhinge: float = None
    uhvec: Vector = None
    reverse: bool = None
    pnls: list['Panel'] = None

    def __init__(self, name: str, posgain: float, neggain: float,
                 xhinge: float) -> None:
        self.name = name
        self.posgain = posgain
        self.neggain = neggain
        self.xhinge = xhinge
        self.pnls = []

    def set_hinge_vector(self, hvec: Vector) -> None:
        if hvec.return_magnitude() != 0.0:
            self.uhvec = hvec.to_unit()
        else:
            self.uhvec = hvec

    def duplicate(self, mirror: bool = True) -> 'PanelControl':
        if mirror and self.reverse:
            posgain, neggain = -self.neggain, -self.posgain
        else:
            posgain, neggain = self.posgain, self.neggain
        if mirror:
            uhvec = Vector(self.uhvec.x, -self.uhvec.y, self.uhvec.z)
        else:
            uhvec = Vector(self.uhvec.x, self.uhvec.y, self.uhvec.z)
        ctrl = PanelControl(self.name, posgain, neggain, self.xhinge)
        ctrl.reverse = False
        ctrl.set_hinge_vector(uhvec)
        return ctrl

    def add_panel(self, pnl: 'Panel') -> None:
        self.pnls.append(pnl)

    @classmethod
    def from_dict(cls, name: str, control_dict: dict[str, Any]) -> 'PanelControl':
        xhinge = control_dict['xhinge']
        posgain = control_dict.get('posgain', 1.0)
        neggain = control_dict.get('neggain', 1.0)
        ctrl = cls(name, posgain, neggain, xhinge)
        hvec_dict = control_dict.get('hvec', {'x': 0.0, 'y': 0.0, 'z': 0.0})
        hvec = Vector.from_dict(hvec_dict)
        ctrl.set_hinge_vector(hvec)
        ctrl.reverse = control_dict.get('reverse', False)
        return ctrl

    def __repr__(self):
        return f'PanelControl({self.name})'

    def __str__(self):
        return f'PanelControl({self.name})'
