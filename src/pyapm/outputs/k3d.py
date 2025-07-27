from typing import TYPE_CHECKING, Any

from k3d import mesh, vectors
from k3d.plot import Plot
from k3d.colormaps import matplotlib_color_maps
from numpy import concatenate, float32, uint32, zeros

if TYPE_CHECKING:
    from k3d.factory import Mesh, Vectors
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector

    from ..classes.panelresult import PanelResult
    from ..classes.panelsystem import PanelSystem

SET_SET = {'_system', '_result'}


class PanelPlot:
    _system: 'PanelSystem'
    _result: 'PanelResult | None'
    _pinds: 'NDArray'
    _verts: 'NDArray'
    _faces: 'NDArray'
    _pntos: 'NDArray'
    _grids: 'NDArray'
    _vinds: 'NDArray'

    __slots__ = tuple(__annotations__)

    def __init__(self, system: 'PanelSystem | None' = None,
                 result: 'PanelResult | None' = None) -> None:
        self._system = system
        self._result = result
        self.reset()

    def reset(self, exclude: set[str] | None = None) -> None:
        if exclude is None:
            exclude = set()
        exclude.update(SET_SET)
        for attr in self.__slots__:
            if attr not in exclude and attr.startswith('_'):
                setattr(self, attr, None)

    def plot(self) -> 'Plot':
        return Plot()

    @property
    def system(self) -> 'PanelSystem':
        if self._system is None:
            self._system = self._result.sys
        return self._system

    @system.setter
    def system(self, system: 'PanelSystem | None') -> None:
        self._system = system

    @property
    def result(self) -> 'PanelResult | None':
        return self._result

    @result.setter
    def result(self, result: 'PanelResult | None') -> None:
        self._result = result

    def create_plot(self, **kwargs: dict[str, Any]) -> Plot:
        return Plot(**kwargs)

    def calculate_panel(self) -> None:
        num_faces = 0
        num_verts = 0
        for panel in self.system.pnls.values():
            num_faces += panel.num
            num_verts += panel.num*3
        self._pinds = zeros(num_verts, dtype=uint32)
        self._vinds = zeros(num_verts, dtype=uint32)
        self._verts = zeros((num_verts, 3), dtype=float32)
        self._faces = zeros((num_faces, 3), dtype=uint32)
        self._pntos = zeros((self.system.numpnl, 3), dtype=float32)

        k = 0
        l = 0
        for i, panel in enumerate(self.system.pnls.values()):
            self._pntos[i, :] = panel.pnto.to_xyz()
            for j in range(panel.num):
                self._pinds[k] = panel.ind
                self._vinds[k] = panel.grds[j - 1].ind + self.system.numpnl
                self._verts[k, :] = panel.grds[j - 1].to_xyz()
                self._faces[l, 0] = k
                k += 1
                self._pinds[k] = panel.ind
                self._vinds[k] = panel.grds[j].ind + self.system.numpnl
                self._verts[k, :] = panel.grds[j].to_xyz()
                self._faces[l, 1] = k
                k += 1
                self._pinds[k] = panel.ind
                self._vinds[k] = panel.ind
                self._verts[k, :] = panel.pnto.to_xyz()
                self._faces[l, 2] = k
                k += 1
                l += 1

    @property
    def pinds(self) -> 'NDArray':
        if self._pinds is None:
            self.calculate_panel()
        return self._pinds

    @property
    def vinds(self) -> 'NDArray':
        if self._vinds is None:
            self.calculate_panel()
        return self._vinds

    @property
    def verts(self) -> 'NDArray':
        if self._verts is None:
            self.calculate_panel()
        return self._verts

    @property
    def faces(self) -> 'NDArray':
        if self._faces is None:
            self.calculate_panel()
        return self._faces

    @property
    def pntos(self) -> 'NDArray':
        if self._pntos is None:
            self.calculate_panel()
        return self._pntos

    @property
    def grids(self) -> 'NDArray':
        if self._grids is None:
            self._grids = zeros((self.system.numgrd, 3), dtype=float32)
            for grid in self.system.grds.values():
                self._grids[grid.ind, :] = grid.to_xyz()
        return self._grids

    def panel_mesh(self, **kwargs: dict[str, Any]) -> 'Mesh':
        kwargs['color'] = kwargs.get('color', 0xffd500)
        kwargs['wireframe'] = kwargs.get('wireframe', False)
        kwargs['flat_shading'] = kwargs.get('flat_shading', False)
        defcmap = matplotlib_color_maps.Turbo
        kwargs['color_map'] = kwargs.get('color_map', defcmap)
        return mesh(self.verts, self.faces, **kwargs)

    def panel_mesh_plot(self, values: 'NDArray', **kwargs: dict[str, Any]) -> 'Mesh':
        kwargs['color'] = kwargs.get('color', 0xffd500)
        kwargs['wireframe'] = kwargs.get('wireframe', False)
        kwargs['flat_shading'] = kwargs.get('flat_shading', False)
        defcmap = matplotlib_color_maps.Turbo
        kwargs['color_map'] = kwargs.get('color_map', defcmap)
        attribute = values[self.pinds].astype(float32)
        return mesh(self.verts, self.faces, attribute=attribute, **kwargs)

    def panel_vectors(self, vecs: 'Vector', **kwargs: dict[str, Any]) -> 'Vectors':
        scale = kwargs.pop('scale', 1.0)
        values = vecs.stack_xyz().astype(float32)*scale
        return vectors(self.pntos, values, **kwargs)

    def panel_mu_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot mu.')
        return self.panel_mesh_plot(self.result.mu, **kwargs)

    def panel_sigma_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot sigma.')
        return self.panel_mesh_plot(self.result.sig, **kwargs)

    def grid_mesh_plot(self, values: 'NDArray', **kwargs: dict[str, Any]) -> 'Mesh':
        kwargs['color'] = kwargs.get('color', 0xffd500)
        kwargs['wireframe'] = kwargs.get('wireframe', False)
        kwargs['flat_shading'] = kwargs.get('flat_shading', False)
        defcmap = matplotlib_color_maps.Turbo
        kwargs['color_map'] = kwargs.get('color_map', defcmap)
        attribute = values[self.vinds].astype(float32)
        return mesh(self.verts, self.faces, attribute=attribute, **kwargs)

    def grid_mu_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot mu.')
        values = concatenate((self.result.mu, self.result.mug))
        return self.grid_mesh_plot(values, **kwargs)

    def grid_sigma_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot sigma.')
        values = concatenate((self.result.sig, self.result.sigg))
        return self.grid_mesh_plot(values, **kwargs)

    def panel_vectors_plot(self, vecs: 'Vector', **kwargs: dict[str, Any]) -> 'Vectors':
        return self.panel_vectors(vecs, **kwargs)

    def panel_force_plot(self, **kwargs: dict[str, Any]) -> 'Vectors':
        if self.result is None:
            raise ValueError('Result must be set to plot forces.')
        return self.panel_vectors_plot(self.result.nfres.nffrc, **kwargs)

    def panel_pressure_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot pressures.')
        return self.panel_mesh_plot(self.result.nfres.nfprs, **kwargs)

    def panel_cp_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot pressures.')
        kwargs['color_range'] = kwargs.get('color_range', [])
        return self.panel_mesh_plot(self.result.nfres.nfcp, **kwargs)
