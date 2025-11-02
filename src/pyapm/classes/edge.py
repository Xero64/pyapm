from typing import TYPE_CHECKING

from numpy import argwhere, sort, unique, zeros
from pygeom.geom2d import Vector2D
from pygeom.geom3d import Coordinate

from ..classes.face import Face

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector

    from ..classes.grid import Grid
    from ..classes.panel import Panel
    from ..classes.panelsystem import PanelSystem

class PanelEdge():
    grida: 'Grid' = None
    gridb: 'Grid' = None
    panel: 'Panel' = None
    mesh_edge: 'MeshEdge' = None
    _face: Face = None

    def __init__(self, grida: 'Grid', gridb: 'Grid', panel: 'Panel') -> None:
        self.grida = grida
        self.gridb = gridb
        self.panel = panel

    @property
    def face(self) -> Face:
        if self._face is None:
            self._face = Face(self.grida, self.gridb, self.panel)
            self._face.set_dirl(self.panel.crd.dirx)
        return self._face

    @property
    def edge_point(self) -> 'Vector':
        if self.mesh_edge is not None:
            return self.mesh_edge.edge_point
        else:
            return None

    @property
    def ind(self) -> int:
        if self.mesh_edge is not None:
            return self.mesh_edge.ind
        else:
            return None

    def __repr__(self) -> str:
        return f'PanelEdge({self.grida}, {self.gridb}, {self.panel})'

    def __str__(self) -> str:
        return self.__repr__()


class MeshEdge():
    ind: int = None
    _edge_point: 'Vector' = None

    def __init__(self) -> None:
        pass

    @property
    def edge_point(self) -> 'Vector':
        return self._edge_point


class BoundaryEdge(MeshEdge):
    panel_edge: PanelEdge = None
    _mid_point: 'Vector' = None
    _edge_point: 'Vector' = None

    def __init__(self, edge: PanelEdge) -> None:
        self.panel_edge = edge
        self.panel_edge.mesh_edge = self

    @property
    def panel(self) -> 'Panel':
        return self.panel_edge.panel

    @property
    def face(self) -> 'Face':
        return self.panel_edge.face

    @property
    def mid_point(self) -> 'Vector':
        if self._mid_point is None:
            self._mid_point = 0.5 * (self.panel_edge.grida + self.panel_edge.gridb)
        return self._mid_point

    @property
    def edge_point(self) -> 'Vector':
        return self.mid_point

    def __repr__(self) -> str:
        return f'BoundaryEdge({self.panel_edge})'

    def __str__(self) -> str:
        return self.__repr__()


class InternalEdge(MeshEdge):
    panel_edgea: PanelEdge = None
    panel_edgeb: PanelEdge = None
    _mid_point: 'Vector' = None
    _direcy: 'Vector' = None
    _coorda: 'Coordinate' = None
    _coordb: 'Coordinate' = None
    _pointa: 'Vector2D' = None
    _pointb: 'Vector2D' = None
    _edge_point: 'Vector' = None
    _panela_len: float = None
    _panelb_len: float = None
    _panel_tot: float = None
    _panela_fac: float = None
    _panelb_fac: float = None

    def __init__(self, panel_edgea: PanelEdge, panel_edgeb: PanelEdge) -> None:
        self.panel_edgea = panel_edgea
        self.panel_edgea.mesh_edge = self
        self.panel_edgeb = panel_edgeb
        self.panel_edgeb.mesh_edge = self

    @property
    def panela(self) -> 'Panel':
        return self.panel_edgea.panel

    @property
    def panelb(self) -> 'Panel':
        return self.panel_edgeb.panel

    @property
    def facea(self) -> 'Face':
        return self.panel_edgea.face

    @property
    def faceb(self) -> 'Face':
        return self.panel_edgeb.face

    @property
    def mid_point(self) -> 'Vector':
        if self._mid_point is None:
            self._mid_point = 0.5 * (self.panel_edgea.grida + self.panel_edgea.gridb)
        return self._mid_point

    @property
    def direcy(self) -> 'Vector':
        if self._direcy is None:
            vecy = self.panel_edgea.gridb - self.panel_edgea.grida
            self._direcy = vecy.to_unit()
        return self._direcy

    @property
    def coorda(self) -> 'Coordinate':
        if self._coorda is None:
            if self.panela is None:
                self._coorda = None
            else:
                direcz_a = self.panela.crd.dirz
                direcx_a = self.direcy.cross(direcz_a)
                self._coorda = Coordinate(self.mid_point, direcx_a, self.direcy)
        return self._coorda

    @property
    def coordb(self) -> 'Coordinate':
        if self._coordb is None:
            if self.panelb is None:
                self._coordb = None
            else:
                direcz_b = self.panelb.crd.dirz
                direcx_b = self.direcy.cross(direcz_b)
                self._coordb = Coordinate(self.mid_point, direcx_b, self.direcy)
        return self._coordb

    @property
    def pointa(self) -> 'Vector2D':
        if self._pointa is None:
            vecag = self.coorda.point_to_local(self.panela.pnto)
            self._pointa = Vector2D.from_obj(vecag)
        return self._pointa

    @property
    def pointb(self) -> 'Vector2D':
        if self._pointb is None:
            vecbg = self.coordb.point_to_local(self.panelb.pnto)
            self._pointb = Vector2D.from_obj(vecbg)
        return self._pointb

    @property
    def edge_point(self) -> 'Vector':
        if self._edge_point is None:
            m = (self.pointb.y - self.pointa.y) / (self.pointb.x - self.pointa.x)
            c = self.pointa.y - m * self.pointa.x
            self._edge_point = self.mid_point + self.direcy*c
        return self._edge_point

    @property
    def panela_len(self) -> float:
        if self._panela_len is None:
            if self.panela is None:
                self._panela_len = 0.0
            else:
                panela_vec = self.edge_point - self.panela.pnto
                self._panela_len = panela_vec.return_magnitude()
        return self._panela_len

    @property
    def panelb_len(self) -> float:
        if self._panelb_len is None:
            if self.panelb is None:
                self._panelb_len = 0.0
            else:
                panelb_vec = self.edge_point - self.panelb.pnto
                self._panelb_len = panelb_vec.return_magnitude()
        return self._panelb_len

    @property
    def panel_tot(self) -> float:
        if self._panel_tot is None:
            self._panel_tot = self.panela_len + self.panelb_len
        return self._panel_tot

    @property
    def panela_fac(self) -> float:
        if self._panela_fac is None:
            self._panela_fac = self.panela_len / self.panel_tot
        return self._panela_fac

    @property
    def panelb_fac(self) -> float:
        if self._panelb_fac is None:
            self._panelb_fac = self.panelb_len / self.panel_tot
        return self._panelb_fac

    def __repr__(self) -> str:
        return f'InternalEdge({self.panel_edgea}, {self.panel_edgeb})'

    def __str__(self) -> str:
        return self.__repr__()


def edges_from_system(system: 'PanelSystem') -> list[MeshEdge]:
    """Create a list of unique edges from a PanelSystem.
    Args:
        system (PanelSystem): The panel system from which to extract edges.
    Returns:
        list[Edge]: A list of unique edges in the panel system.
    """
    all_edges: list[PanelEdge] = []
    for dpanel in system.dpanels.values():
        for panel_edge in dpanel.panel_edges:
            all_edges.append(panel_edge)
    # for wpanel in system.wpanels.values():
    #     for edge in wpanel.edges:
    #         all_edges.append(edge)
    num_edges = len(all_edges)
    edge_gids = zeros((num_edges, 2), dtype=int)
    for i, edge in enumerate(all_edges):
        edge_gids[i, 0] = edge.grida.gid
        edge_gids[i, 1] = edge.gridb.gid

    sorted_edges = sort(edge_gids, axis=1)
    unique_edges, edge_inverse = unique(sorted_edges, axis=0, return_inverse=True)
    # print(f'{num_edges = }')
    # print(f'{unique_edges = }')
    # print(f'{unique_edges.shape[0] = }')
    # print(f'{edge_inverse = }')
    # print(f'{edge_inverse.size = }')
    edges = []
    for i in range(unique_edges.shape[0]):
        edge_inds = argwhere(edge_inverse == i).flatten()
        if edge_inds.size == 1:
            edge = BoundaryEdge(all_edges[edge_inds[0]])
            edges.append(edge)
        elif edge_inds.size > 1:
            edge = InternalEdge(all_edges[edge_inds[0]], all_edges[edge_inds[1]])
            edges.append(edge)
    return edges

# def edges_array(edges: list[InternalEdge]) -> 'NDArray':
#     num_edges = len(edges)
#     # conedges = [edge for edge in edges if edge.panel is None]
#     # num_edges = len(conedges)
#     max_vind = None
#     max_pind = None
#     for i, edge in enumerate(edges):
#         if max_vind is None or edge.grida.ind > max_vind:
#             max_vind = edge.grida.ind
#         if max_vind is None or edge.gridb.ind > max_vind:
#             max_vind = edge.gridb.ind
#         if max_pind is None or (edge.panela is not None and edge.panela.ind > max_pind):
#             max_pind = edge.panela.ind
#         if max_pind is None or (edge.panelb is not None and edge.panelb.ind > max_pind):
#             max_pind = edge.panelb.ind
#     varray = zeros((num_edges, max_vind + 1), dtype=float)
#     parray = zeros((num_edges, max_pind + 1), dtype=float)
#     for i, edge in enumerate(edges):
#         if edge.panel is None:
#             varray[i, edge.grida.ind] = edge.gridb_fac
#             varray[i, edge.gridb.ind] = edge.grida_fac
#             parray[i, edge.panela.ind] = edge.panelb_fac
#             parray[i, edge.panelb.ind] = edge.panela_fac
#         else:
#             varray[i, edge.grida.ind] = 0.5
#             varray[i, edge.gridb.ind] = 0.5
#             parray[i, edge.panel.ind] = 1.0
#             # grid_inds = edge.panel.edge_indg
#             panel_inds = edge.panel.edge_indp
#             # edge_velg = edge.panel.edge_velg
#             edge_velp = edge.panel.edge_velp
#             # edge_mug = edge_velg.dot(edge.vecec)
#             edge_mud = edge_velp.dot(edge.vecec)
#             parray[i, panel_inds] -= edge_mud
#             # varray[i, grid_inds] += edge_mug
#     amat = varray.transpose() @ varray
#     bmat = varray.transpose() @ parray
#     earray = solve(amat, bmat)
#     return earray

def edges_parray(mesh_edges: list[MeshEdge]) -> 'NDArray':
    num_edges = len(mesh_edges)
    max_pind = None
    for mesh_edge in mesh_edges:
        if isinstance(mesh_edge, BoundaryEdge):
            if max_pind is None or mesh_edge.panel_edge.panel.ind > max_pind:
                max_pind = mesh_edge.panel_edge.panel.ind
        elif isinstance(mesh_edge, InternalEdge):
            if mesh_edge.panela is not None:
                if max_pind is None or mesh_edge.panela.ind > max_pind:
                    max_pind = mesh_edge.panela.ind
            if mesh_edge.panelb is not None:
                if max_pind is None or mesh_edge.panelb.ind > max_pind:
                    max_pind = mesh_edge.panelb.ind
    parray = zeros((num_edges, max_pind + 1), dtype=float)
    for mesh_edge in mesh_edges:
        if isinstance(mesh_edge, BoundaryEdge):
            parray[mesh_edge.ind, mesh_edge.panel.ind] = 1.0
            vecg = mesh_edge.panel_edge.edge_point - mesh_edge.panel.pnto
            vecl = mesh_edge.face.cord.vector_to_local(vecg)
            dirl = Vector2D.from_obj(vecl)
            dmue = mesh_edge.panel.edge_velp.dot(dirl)
            parray[mesh_edge.ind, mesh_edge.panel.edge_indp] -= dmue
        elif isinstance(mesh_edge, InternalEdge):
            parray[mesh_edge.ind, mesh_edge.panela.ind] = mesh_edge.panelb_fac
            parray[mesh_edge.ind, mesh_edge.panelb.ind] = mesh_edge.panela_fac
    return parray
