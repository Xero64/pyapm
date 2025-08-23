from typing import TYPE_CHECKING, Any

from k3d import Plot, mesh, vectors
from numpy import float32, uint32, zeros

if TYPE_CHECKING:
    from k3d.objects import Mesh, Vectors
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector

    from .constantresult import ConstantResult
    from .constantsystem import ConstantSystem

SET_SET = {'_system', '_result'}


class ConstantPlot:
    _system: 'ConstantSystem'
    _result: 'ConstantResult | None'
    _dpanel_pinds: 'NDArray'
    _dpanel_verts: 'NDArray'
    _dpanel_faces: 'NDArray'
    _dpanel_pntos: 'NDArray'
    _npanel_pinds: 'NDArray'
    _npanel_verts: 'NDArray'
    _npanel_faces: 'NDArray'
    _npanel_pntos: 'NDArray'
    _grids: 'NDArray'

    __slots__ = tuple(__annotations__)

    def __init__(self, system: 'ConstantSystem | None' = None,
                 result: 'ConstantResult | None' = None) -> None:
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

    @property
    def system(self) -> 'ConstantSystem':
        if self._system is None:
            self._system = self._result.system
        return self._system

    @system.setter
    def system(self, system: 'ConstantSystem | None') -> None:
        self._system = system

    @property
    def result(self) -> 'ConstantResult | None':
        return self._result

    @result.setter
    def result(self, result: 'ConstantResult | None') -> None:
        self._result = result

    def create_plot(self, **kwargs: dict[str, Any]) -> Plot:
        return Plot(**kwargs)

    def calculate_dpanel(self) -> None:
        num_faces = 0
        num_verts = 0
        for dpanel in self.system.dpanels:
            num_faces += dpanel.num
            num_verts += dpanel.num*3
        self._dpanel_pinds = zeros(num_verts, dtype=uint32)
        self._dpanel_verts = zeros((num_verts, 3), dtype=float32)
        self._dpanel_faces = zeros((num_faces, 3), dtype=uint32)
        self._dpanel_pntos = zeros((self.system.num_dpanels, 3), dtype=float32)

        k = 0
        l = 0
        for i, dpanel in enumerate(self.system.dpanels):
            self._dpanel_pntos[i, :] = dpanel.point.to_xyz()
            for tria in dpanel.trias:
                for j in range(3):
                    self._dpanel_pinds[k] = i
                    self._dpanel_verts[k, :] = tria.grds[j].to_xyz()
                    self._dpanel_faces[l, j] = k
                    k += 1
                l += 1

    @property
    def dpanel_pinds(self) -> 'NDArray':
        if self._dpanel_pinds is None:
            self.calculate_dpanel()
        return self._dpanel_pinds

    @property
    def dpanel_verts(self) -> 'NDArray':
        if self._dpanel_verts is None:
            self.calculate_dpanel()
        return self._dpanel_verts

    @property
    def dpanel_faces(self) -> 'NDArray':
        if self._dpanel_faces is None:
            self.calculate_dpanel()
        return self._dpanel_faces

    @property
    def dpanel_pntos(self) -> 'NDArray':
        if self._dpanel_pntos is None:
            self.calculate_dpanel()
        return self._dpanel_pntos

    def calculate_npanel(self) -> None:
        num_faces = 0
        num_verts = 0
        for npanel in self.system.npanels:
            num_faces += npanel.num
            num_verts += npanel.num*3
        self._npanel_pinds = zeros(num_verts, dtype=uint32)
        self._npanel_verts = zeros((num_verts, 3), dtype=float32)
        self._npanel_faces = zeros((num_faces, 3), dtype=uint32)
        self._npanel_pntos = zeros((self.system.num_npanels, 3), dtype=float32)

        k = 0
        l = 0
        for i, npanel in enumerate(self.system.npanels):
            self._npanel_pntos[i, :] = npanel.point.to_xyz()
            for tria in npanel.trias:
                for j in range(3):
                    self._npanel_pinds[k] = i
                    self._npanel_verts[k, :] = tria.grds[j].to_xyz()
                    self._npanel_faces[l, j] = k
                    k += 1
                l += 1

    @property
    def npanel_pinds(self) -> 'NDArray':
        if self._npanel_pinds is None:
            self.calculate_npanel()
        return self._npanel_pinds

    @property
    def npanel_verts(self) -> 'NDArray':
        if self._npanel_verts is None:
            self.calculate_npanel()
        return self._npanel_verts

    @property
    def npanel_faces(self) -> 'NDArray':
        if self._npanel_faces is None:
            self.calculate_npanel()
        return self._npanel_faces

    @property
    def npanel_pntos(self) -> 'NDArray':
        if self._npanel_pntos is None:
            self.calculate_npanel()
        return self._npanel_pntos

    @property
    def grids(self) -> 'NDArray':
        if self._grids is None:
            self._grids = zeros((self.system.num_grids, 3), dtype=float32)
            for grid in self.system.grids:
                self._grids[grid.ind, :] = grid.to_xyz()
        return self._grids

    def dpanel_mesh(self, **kwargs: dict[str, Any]) -> 'Mesh':
        return mesh(self.dpanel_verts, self.dpanel_faces, **kwargs)

    def npanel_mesh(self, **kwargs: dict[str, Any]) -> 'Mesh':
        return mesh(self.npanel_verts, self.npanel_faces, **kwargs)

    def dpanel_mesh_plot(self, values: 'NDArray', **kwargs: dict[str, Any]) -> 'Mesh':
        attribute = values[self.dpanel_pinds].astype(float32)
        return mesh(self.dpanel_verts, self.dpanel_faces, attribute=attribute, **kwargs)

    def npanel_mesh_plot(self, values: 'NDArray', **kwargs: dict[str, Any]) -> 'Mesh':
        attribute = values[self.npanel_pinds].astype(float32)
        return mesh(self.npanel_verts, self.npanel_faces, attribute=attribute, **kwargs)

    def dpanel_vectors(self, vecs: 'Vector', **kwargs: dict[str, Any]) -> 'Vectors':
        values = vecs.stack_xyz().astype(float32)
        return vectors(self.dpanel_pntos, values, **kwargs)

    def npanel_vectors(self, vecs: 'Vector', **kwargs: dict[str, Any]) -> 'Vectors':
        values = vecs.stack_xyz().astype(float32)
        return vectors(self.npanel_pntos, values, **kwargs)

    def dpanel_mu_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        return self.dpanel_mesh_plot(self.result.mud, **kwargs)

    def dpanel_sigma_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        return self.dpanel_mesh_plot(self.result.sig, **kwargs)

    def npanel_mu_plot(self, **kwargs: dict[str, Any]) -> 'Mesh':
        return self.npanel_mesh_plot(self.result.mun, **kwargs)

    def dpanel_normal_vectors(self, **kwargs: dict[str, Any]) -> 'Vectors':
        scale = kwargs.pop('scale', 1.0)
        if self.result is None:
            dnrml = self.system.dnormal
        else:
            dnrml = self.result.dnrml
        return self.dpanel_vectors(dnrml*scale, **kwargs)

    def dpanel_normal_approx_vectors(self, **kwargs: dict[str, Any]) -> 'Vectors':
        scale = kwargs.pop('scale', 1.0)
        return self.dpanel_vectors(self.result.dnormal_approx*scale, **kwargs)

    def npanel_normal_vectors(self, **kwargs: dict[str, Any]) -> 'Vectors':
        scale = kwargs.pop('scale', 1.0)
        if self.result is None:
            nnrml = self.system.nnormal
        else:
            nnrml = self.result.nnrml
        return self.npanel_vectors(nnrml*scale, **kwargs)

    def npanel_normal_approx_vectors(self, **kwargs: dict[str, Any]) -> 'Vectors':
        scale = kwargs.pop('scale', 1.0)
        return self.npanel_vectors(self.result.nnormal_approx*scale, **kwargs)

    def grid_vectors(self, vecs: 'Vector', **kwargs: dict[str, Any]) -> 'Vectors':
        scale = kwargs.pop('scale', 1.0)
        vecs = vecs*scale
        values = vecs.stack_xyz().astype(float32)
        return vectors(self.grids, values, **kwargs)

    def grid_velocity_vectors(self, **kwargs: dict[str, Any]) -> 'Vectors':
        vecs = self.result.result.ngvel
        return self.grid_vectors(vecs, **kwargs)

    def grid_force_vectors(self, **kwargs: dict[str, Any]) -> 'Vectors':
        vecs = self.result.result.ngfrc
        return self.grid_vectors(vecs, **kwargs)

    def grid_moment_vectors(self, **kwargs: dict[str, Any]) -> 'Vectors':
        vecs = self.result.result.ngmom
        return self.grid_vectors(vecs, **kwargs)

    def grid_pressure_vectors(self, **kwargs: dict[str, Any]) -> 'Vectors':
        vecs = self.result.result.ngfrc/self.system.gridarea
        return self.grid_vectors(vecs, **kwargs)
