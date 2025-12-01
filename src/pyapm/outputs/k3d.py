from typing import TYPE_CHECKING, Any

from k3d import mesh, vectors
from k3d.colormaps import matplotlib_color_maps
from k3d.helpers import map_colors
from k3d.plot import Plot
from numpy import concatenate, float32, uint32, zeros
from pyapm.classes.grid import Vertex

if TYPE_CHECKING:
    from k3d.objects import Mesh, Vectors
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
    _fpnts: 'NDArray'
    _finds: 'NDArray'

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

    def calculate_faces(self) -> None:
        # num_faces = 0
        # num_verts = 0
        # for panel in self.system.dpanels.values():
        #     num_faces += panel.num
        #     num_verts += panel.num*3
        num_faces = self.system.num_dfacets
        num_verts = self.system.num_dfacets*3
        self._finds = zeros(num_verts, dtype=uint32)
        self._pinds = zeros(num_verts, dtype=uint32)
        self._vinds = zeros(num_verts, dtype=uint32)
        self._verts = zeros((num_verts, 3), dtype=float32)
        self._faces = zeros((num_faces, 3), dtype=uint32)
        self._fpnts = zeros((num_faces, 3), dtype=float32)
        self._pntos = zeros((self.system.num_dpanels, 3), dtype=float32)

        edge_beg = self.system.num_dpanels
        vert_beg = self.system.num_dpanels + self.system.num_edges

        for i, dfacet in enumerate(self.system.dfacets):
            self._finds[i*3:i*3 + 3] = i
            self._pinds[i*3:i*3 + 3] = dfacet.panel.ind
            if isinstance(dfacet.vertexa, Vertex):
                self._vinds[i*3] = vert_beg + dfacet.vertexa.ind
                self._vinds[i*3 + 1] = edge_beg + dfacet.edge.ind
                self._verts[i*3, :] = dfacet.vertexa.to_xyz()
                self._verts[i*3 + 1, :] = dfacet.edge.edge_point.to_xyz()
            elif isinstance(dfacet.vertexb, Vertex):
                self._vinds[i*3] = edge_beg + dfacet.edge.ind
                self._vinds[i*3 + 1] = vert_beg + dfacet.vertexb.ind
                self._verts[i*3, :] = dfacet.edge.edge_point.to_xyz()
                self._verts[i*3 + 1, :] = dfacet.vertexb.to_xyz()
            self._vinds[i*3 + 2] = dfacet.panel.ind
            self._verts[i*3 + 2, :] = dfacet.panel.pnto.to_xyz()
            self._faces[i, 0] = i*3
            self._faces[i, 1] = i*3 + 1
            self._faces[i, 2] = i*3 + 2
            self._fpnts[i, :] = dfacet.cord.pnt.to_xyz()

        # k = 0
        # l = 0
        # for i, panel in enumerate(self.system.dpanels.values()):
        #     self._pntos[i, :] = panel.pnto.to_xyz()
        #     for j in range(panel.num):
        #         self._pinds[k] = panel.ind
        #         self._vinds[k] = panel.grids[j - 1].ind + self.system.num_dpanels
        #         self._verts[k, :] = panel.grids[j - 1].to_xyz()
        #         self._faces[l, 0] = k
        #         k += 1
        #         self._pinds[k] = panel.ind
        #         self._vinds[k] = panel.grids[j].ind + self.system.num_dpanels
        #         self._verts[k, :] = panel.grids[j].to_xyz()
        #         self._faces[l, 1] = k
        #         k += 1
        #         self._pinds[k] = panel.ind
        #         self._vinds[k] = panel.ind
        #         self._verts[k, :] = panel.pnto.to_xyz()
        #         self._faces[l, 2] = k
        #         k += 1
        #         l += 1

    @property
    def pinds(self) -> 'NDArray':
        if self._pinds is None:
            self.calculate_faces()
        return self._pinds

    @property
    def vinds(self) -> 'NDArray':
        if self._vinds is None:
            self.calculate_faces()
        return self._vinds

    @property
    def verts(self) -> 'NDArray':
        if self._verts is None:
            self.calculate_faces()
        return self._verts

    @property
    def faces(self) -> 'NDArray':
        if self._faces is None:
            self.calculate_faces()
        return self._faces

    @property
    def pntos(self) -> 'NDArray':
        if self._pntos is None:
            self.calculate_faces()
        return self._pntos

    @property
    def grids(self) -> 'NDArray':
        if self._grids is None:
            self._grids = zeros((self.system.num_grids, 3), dtype=float32)
            for grid in self.system.grids.values():
                self._grids[grid.ind, :] = grid.to_xyz()
        return self._grids

    @property
    def finds(self) -> 'NDArray':
        if self._finds is None:
            self.calculate_faces()
        return self._finds

    @property
    def fpnts(self) -> 'NDArray':
        if self._fpnts is None:
            self.calculate_faces()
        return self._fpnts

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
            raise ValueError('Result must be set to plot mud.')
        return self.panel_mesh_plot(self.result.mud, **kwargs)

    def panel_sigma_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot sigma.')
        return self.panel_mesh_plot(self.result.sigd, **kwargs)

    def vertex_mesh_plot(self, values: 'NDArray', **kwargs: dict[str, Any]) -> 'Mesh':
        kwargs['color'] = kwargs.get('color', 0xffd500)
        kwargs['wireframe'] = kwargs.get('wireframe', False)
        kwargs['flat_shading'] = kwargs.get('flat_shading', False)
        defcmap = matplotlib_color_maps.Turbo
        kwargs['color_map'] = kwargs.get('color_map', defcmap)
        attribute = values[self.vinds].astype(float32)
        return mesh(self.verts, self.faces, attribute=attribute, **kwargs)

    def vertex_mu_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot mu.')
        values = concatenate((self.result.mud, self.result.mue, self.result.muv))
        return self.vertex_mesh_plot(values, **kwargs)

    def vertex_sigma_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot sigma.')
        values = concatenate((self.result.sigd, self.result.sige, self.result.sigv))
        return self.vertex_mesh_plot(values, **kwargs)

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

    def face_mesh_plot(self, values: 'NDArray', **kwargs: dict[str, Any]) -> 'Mesh':
        kwargs['color'] = kwargs.get('color', 0xffd500)
        kwargs['wireframe'] = kwargs.get('wireframe', False)
        kwargs['flat_shading'] = kwargs.get('flat_shading', False)
        defcmap = matplotlib_color_maps.Turbo
        kwargs['color_map'] = kwargs.get('color_map', defcmap)
        attribute = values[self.finds].astype(float32)
        return mesh(self.verts, self.faces, attribute=attribute, **kwargs)

    def face_vx_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot face vx.')
        return self.face_mesh_plot(self.result.fres.fvel.x, **kwargs)

    def face_vy_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot face vy.')
        return self.face_mesh_plot(self.result.fres.fvel.y, **kwargs)

    def face_cp_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        if self.result is None:
            raise ValueError('Result must be set to plot face cp.')
        kwargs['color_range'] = kwargs.get('color_range', [])
        return self.face_mesh_plot(self.result.fres.fcp, **kwargs)

    def face_vectors(self, vecs: 'Vector', **kwargs: dict[str, Any]) -> 'Vectors':
        scale = kwargs.pop('scale', 1.0)
        mags = vecs.return_magnitude()
        defcmap = matplotlib_color_maps.Turbo
        colors = map_colors(mags, color_map=defcmap).reshape((-1, 1)).repeat(2, axis=1).flatten().astype(uint32).tolist()
        values = vecs.stack_xyz().astype(float32)*scale
        kwargs['colors'] = kwargs.setdefault('colors', colors)
        return vectors(self.fpnts, values, **kwargs)

    def face_vectors_plot(self, vecs: 'Vector', **kwargs: dict[str, Any]) -> 'Vectors':
        return self.face_vectors(vecs, **kwargs)

    def face_force_plot(self, **kwargs: dict[str, Any]) -> 'Vectors':
        if self.result is None:
            raise ValueError('Result must be set to plot face forces.')
        return self.face_vectors_plot(self.result.fres.ffrc, **kwargs)
