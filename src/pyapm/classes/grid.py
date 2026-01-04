from typing import TYPE_CHECKING

from numpy import asarray, reciprocal, zeros
from pygeom.geom2d import Vector2D
from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .panel import Panel


class Vertex(Vector):
    ind: int = None
    panels: set['Panel'] = None

    def __init__(self, x: float, y: float, z: float) -> None:
        super().__init__(x, y, z)
        self.panels = set()

    def __repr__(self) -> str:
        return f'Vertex({self.x}, {self.y}, {self.z})'

    def __str__(self) -> str:
        return self.__repr__()

    def __format__(self, frm: str) -> str:
        return f'Vertex({self.x:{frm}}, {self.y:{frm}}, {self.z:{frm}})'


class Grid(Vector):
    gid: int = None
    ind: int = None
    panels: set['Panel'] = None
    # _ordered_panels: list['Panel'] = None
    _vertices: list[Vertex] = None

    def __init__(self, gid: int, x: float, y: float, z: float) -> None:
        self.gid = gid
        super().__init__(x, y, z)
        self.panels = set()

    @property
    def vertices(self) -> list[Vertex]:
        if self._vertices is None:
            panel_groups = panel_groups_from_grid(self)
            self._vertices: list[Vertex] = []
            for panel_group in panel_groups:
                vertex = Vertex(self.x, self.y, self.z)
                for panel in panel_group:
                    vertex.panels.add(panel)
                self._vertices.append(vertex)
        return self._vertices

    # @property
    # def ordered_panels(self) -> list['Panel']:
    #     if self._ordered_panels is None:
    #         panels = list(self.panels)
    #         self._ordered_panels = [panels[0]]
    #         for paneli in self._ordered_panels:
    #             gridi_ind = paneli.grids.index(self)
    #             gridi_aft = paneli.grids[gridi_ind + 1]
    #             for panelj in panels:
    #                 if panelj not in self._ordered_panels:
    #                     gridj_ind = panelj.grids.index(self)
    #                     gridj_beg = panelj.grids[gridj_ind - 1]
    #                     if gridi_aft is gridj_beg:
    #                         self._ordered_panels.append(panelj)
    #                         break
    #             if len(self._ordered_panels) == len(self.panels):
    #                 break
    #     return self._ordered_panels

    def __repr__(self) -> str:
        return f'Grid({self.gid}, {self.x}, {self.y}, {self.z})'

    def __str__(self) -> str:
        return self.__repr__()

    def __format__(self, frm: str) -> str:
        return f'Grid({self.gid:d}, {self.x:{frm}}, {self.y:{frm}}, {self.z:{frm}})'


def grids_parray(grids: list[Grid]) -> 'NDArray':
    num_grids = len(grids)
    max_pind = None
    for grid in grids:
        for panel in grid.panels:
            if max_pind is None or panel.ind > max_pind:
                max_pind = panel.ind
    parray = zeros((num_grids, max_pind + 1), dtype=float)
    for grid in grids:
        distances = []
        indices = []
        for panel in grid.panels:
            distance = (grid - panel.pnto).return_magnitude()
            distances.append(distance)
            indices.append(panel.ind)
        distances = asarray(distances)
        rec_distances = zeros(distances.shape)
        reciprocal(distances, where=distances!=0.0, out=rec_distances)
        check = distances == 0.0
        if check.any():
            rec_distances[check] = 1.0
            rec_distances[~check] = 0.0
        rec_distances_sum = rec_distances.sum()
        weights = rec_distances / rec_distances_sum
        parray[grid.ind, indices] = weights
    return parray

def panel_groups_from_grid(grid: Grid) -> list[list['Panel']]:

    panels = list(grid.panels)
    panel_groups: list[list[object]] = []

    while len(panels) > 0:

        panel_groups.append([])
        paneli = panels.pop(0)
        panel_groups[-1].append(paneli)

        gridi_bef, gridi_aft = paneli.grids_before_and_after_grid(grid)
        edgei_bef, edgei_aft = paneli.edges_before_and_after_grid(grid)

        # Backward search for adjacent panels
        while len(panels) > 0 and edgei_bef.is_internal():

            for panelj in panels:

                gridj_bef, gridj_aft = panelj.grids_before_and_after_grid(grid)
                edgej_bef, _ = panelj.edges_before_and_after_grid(grid)

                if gridi_bef is gridj_aft:
                    panel_groups[-1].insert(0, panelj)
                    panels.remove(panelj)
                    gridi_bef = gridj_bef
                    edgei_bef = edgej_bef

                    # Check if before edge is internal
                    if edgei_bef.not_internal():
                        break

                if len(panels) == 0:
                    break

        # Forward search for adjacent panels
        while len(panels) > 0 and edgei_aft.is_internal():

            for panelj in panels:

                gridj_bef, gridj_aft = panelj.grids_before_and_after_grid(grid)
                _, edgej_aft = panelj.edges_before_and_after_grid(grid)

                if gridi_aft is gridj_bef:
                    panel_groups[-1].append(panelj)
                    panels.remove(panelj)
                    gridi_aft = gridj_aft
                    edgei_aft = edgej_aft

                    # Check if after edge is internal
                    if edgei_aft.not_internal():
                        break

                if len(panels) == 0:
                    break

        if edgei_bef.not_internal() and edgei_aft.not_internal():
            continue

    return panel_groups

# def vertices_parray(vertices: list[Vertex],
#                     num_dpanel: int) -> 'NDArray':
#     num_verts = len(vertices)
#     parray = zeros((num_verts, num_dpanel), dtype=float)
#     for vertex in vertices:
#         distances = []
#         indices = []
#         for panel in vertex.panels:
#             distance = (vertex - panel.pnto).return_magnitude()
#             distances.append(distance)
#             indices.append(panel.ind)
#         distances = asarray(distances)
#         rec_distances = zeros(distances.shape)
#         reciprocal(distances, where=distances!=0.0, out=rec_distances)
#         check = distances == 0.0
#         if check.any():
#             rec_distances[check] = 1.0
#             rec_distances[~check] = 0.0
#         rec_distances_sum = rec_distances.sum()
#         weights = rec_distances / rec_distances_sum
#         parray[vertex.ind, indices] = weights
#     return parray

# def vertices_parray(vertices: list[Vertex],
#                     num_dpanel: int) -> 'NDArray':
#     num_verts = len(vertices)
#     parray = zeros((num_verts, num_dpanel), dtype=float)
#     for vertex in vertices:
#         numpanel = len(vertex.panels)
#         for panel in vertex.panels:
#             indv = panel.vertices.index(vertex)
#             vecg = vertex - panel.pnto
#             vecl = panel.crd.vector_to_local(vecg)
#             dirl = Vector2D.from_obj(vecl)
#             indp = panel.vert_indps[indv]
#             velp = panel.vert_velps[indv]
#             Dmue = velp.dot(dirl)
#             Dmue = Dmue / Dmue.sum()
#             # parray[vertex.ind, panel.ind] += 1.0/numpanel
#             parray[vertex.ind, indp] += Dmue/numpanel
#     return parray

def vertices_parray(vertices: list[Vertex],
                    num_dpanel: int) -> 'NDArray':
    num_verts = len(vertices)
    parray = zeros((num_verts, num_dpanel), dtype=float)
    for vertex in vertices:
        numpanel = len(vertex.panels)
        for panel in vertex.panels:
            indv = panel.vertices.index(vertex)
            indps = panel.vert_indps[indv]
            facps = panel.vert_facps[indv]
            parray[vertex.ind, indps] += facps / numpanel
    return parray
