from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel import Panel
    from .panelcontrol import PanelControl


class PanelGroup:
    name: str = None
    _pnls: list['Panel'] = None
    _ctrls: dict[str, 'PanelControl'] = None
    exclude: bool = False
    noload: bool = False
    nohsv: bool = False

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def pnls(self) -> list['Panel']:
        if self._pnls is None:
            self._pnls = []
        return self._pnls

    @property
    def ctrls(self) -> dict[str, 'PanelControl']:
        if self._ctrls is None:
            self._ctrls = {}
        return self._ctrls

    def add_panel(self, panel: 'Panel') -> None:
        self.pnls.append(panel)

    def add_control(self, control: 'PanelControl') -> None:
        self.ctrls[control.name] = control
        control.pnls = self.pnls

    def __repr__(self) -> str:
        return f'PanelGroup(name={self.name})'

    def __str__(self) -> str:
        return f'PanelGroup: {self.name} with {len(self.pnls)} panels and {len(self.ctrls)} controls'
