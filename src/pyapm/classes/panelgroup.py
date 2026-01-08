from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel import Panel
    from .panelcontrol import PanelControl


class PanelGroup:
    name: str = None
    _panels: list['Panel'] = None
    _controls: dict[str, 'PanelControl'] = None
    exclude: bool = False
    no_load: bool = False
    nohsv: bool = False

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def panels(self) -> list['Panel']:
        if self._panels is None:
            self._panels = []
        return self._panels

    @property
    def controls(self) -> dict[str, 'PanelControl']:
        if self._controls is None:
            self._controls = {}
        return self._controls

    def add_panel(self, panel: 'Panel') -> None:
        self.panels.append(panel)

    def add_control(self, control: 'PanelControl') -> None:
        self.controls[control.name] = control
        control.panels = self.panels

    def __repr__(self) -> str:
        return f'PanelGroup(name={self.name})'

    def __str__(self) -> str:
        return self.__repr__()
