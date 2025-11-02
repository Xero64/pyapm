from typing import TYPE_CHECKING

from numpy import sort, unique, zeros
from pygeom.geom2d import Vector2D
from pygeom.geom3d import Coordinate

from ..classes.face import Face

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector

    from ..classes.grid import Grid
    from ..classes.panel import Panel
    from ..classes.panelsystem import PanelSystem

class Edge():
    grida: 'Grid' = None
    gridb: 'Grid' = None
    panel: 'Panel' = None
    _face: Face = None

    def __init__(self, grida: 'Grid', gridb: 'Grid', panel: 'Panel') -> None:
        self.grida = grida
        self.gridb = gridb
        self.panel = panel

    @property
    def face(self) -> Face:
        if self._face is None:
            self._face = Face(-1, self.grida, self.gridb, self.panel)
        return self._face


class InternalEdge():
    grida: 'Grid' = None
    gridb: 'Grid' = None
    ind: int = None
    _panela: 'Panel' = None
    _panelb: 'Panel' = None
    _facea: 'Face' = None
    _faceb: 'Face' = None
    _panel: 'Panel' = None
    _direcy: 'Vector' = None
    _coorda: 'Coordinate' = None
    _coordb: 'Coordinate' = None
    _pointa: 'Vector2D' = None
    _pointb: 'Vector2D' = None
    _edge_point: 'Vector' = None
    _panela_len: float = None
    _panelb_len: float = None
    _grida_len: float = None
    _gridb_len: float = None
    _panel_tot: float = None
    _grid_tot: float = None
    _panela_fac: float = None
    _panelb_fac: float = None
    _grida_fac: float = None
    _gridb_fac: float = None
    _vecec: 'Vector2D' = None

    def __init__(self, grida: 'Grid', gridb: 'Grid') -> None:
        self.grida = grida
        self.gridb = gridb
        self.panela
        self.panelb
        self.panel
        self.facea
        self.faceb
        self.face

    @property
    def panela(self) -> 'Panel':
        if self._panela is None:
            panels_a: list['Panel'] = []
            for panel in self.grida.panels:
                for i in range(-1, panel.num-1):
                    if panel.grids[i] is self.grida and panel.grids[i + 1] is self.gridb:
                        panels_a.append(panel)
            if len(panels_a) == 1:
                self._panela = panels_a[0]
                self._panela.edges.append(self)
            elif len(panels_a) > 1:
                raise ValueError(f'Multiple panels found for edge {self.grida} to {self.gridb}')
            else:
                self._panela = None
        return self._panela

    @property
    def panelb(self) -> 'Panel':
        if self._panelb is None:
            panels_b: list['Panel'] = []
            for panel in self.gridb.panels:
                for i in range(-1, panel.num-1):
                    if panel.grids[i] is self.gridb and panel.grids[i + 1] is self.grida:
                        panels_b.append(panel)
            if len(panels_b) == 1:
                self._panelb = panels_b[0]
                self._panelb.edges.append(self)
            elif len(panels_b) > 1:
                raise ValueError(f'Multiple panels found for edge {self.gridb} to {self.grida}')
            else:
                self._panelb = None
        return self._panelb

    @property
    def facea(self) -> 'Face':
        if self._facea is None and self.panela is not None:
            for face in self.panela.faces:
                if face.grida is self.grida and face.gridb is self.gridb:
                    self._facea = face
                    break
                elif face.grida is self.gridb and face.gridb is self.grida:
                    self._facea = face
                    break
        return self._facea

    @property
    def faceb(self) -> 'Face':
        if self._faceb is None and self.panelb is not None:
            for face in self.panelb.faces:
                if face.grida is self.gridb and face.gridb is self.grida:
                    self._faceb = face
                    break
                elif face.grida is self.grida and face.gridb is self.gridb:
                    self._faceb = face
                    break
        return self._faceb

    @property
    def panel(self) -> 'Panel':
        if self._panel is None:
            if self.panela is not None and self.panelb is None:
                self._panel = self.panela
            elif self.panelb is not None and self.panela is None:
                self._panel = self.panelb
            else:
                self._panel = None
        return self._panel

    @property
    def face(self) -> 'Face':
        if self.panel is not None:
            if self.panel is self.panela:
                return self.facea
            elif self.panel is self.panelb:
                return self.faceb
        return None

    @property
    def direcy(self) -> 'Vector':
        if self._direcy is None:
            vecy = self.gridb - self.grida
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
                self._coorda = Coordinate(self.grida, direcx_a, self.direcy)
        return self._coorda

    @property
    def coordb(self) -> 'Coordinate':
        if self._coordb is None:
            if self.panelb is None:
                self._coordb = None
            else:
                direcz_b = self.panelb.crd.dirz
                direcx_b = self.direcy.cross(direcz_b)
                self._coordb = Coordinate(self.grida, direcx_b, self.direcy)
        return self._coordb

    @property
    def pointa(self) -> 'Vector2D':
        if self._pointa is None:
            if self.coorda is None:
                self._pointa = None
            else:
                vecag = self.coorda.point_to_local(self.panela.pnto)
                self._pointa = Vector2D.from_obj(vecag)
        return self._pointa

    @property
    def pointb(self) -> 'Vector2D':
        if self._pointb is None:
            if self.coordb is None:
                self._pointb = None
            else:
                vecbg = self.coordb.point_to_local(self.panelb.pnto)
                self._pointb = Vector2D.from_obj(vecbg)
        return self._pointb

    @property
    def edge_point(self) -> 'Vector':
        if self._edge_point is None:
            if self.pointa is None or self.pointb is None:
                self._edge_point = 0.5 * (self.grida + self.gridb)
            else:
                m = (self.pointb.y - self.pointa.y) / (self.pointb.x - self.pointa.x)
                c = self.pointa.y - m * self.pointa.x
                self._edge_point = self.grida + self.direcy*c
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
    def grida_len(self) -> float:
        if self._grida_len is None:
            self._grida_len = self.grida.return_magnitude()
        return self._grida_len

    @property
    def gridb_len(self) -> float:
        if self._gridb_len is None:
            self._gridb_len = self.gridb.return_magnitude()
        return self._gridb_len

    @property
    def panel_tot(self) -> float:
        if self._panel_tot is None:
            self._panel_tot = self.panela_len + self.panelb_len
        return self._panel_tot

    @property
    def grid_tot(self) -> float:
        if self._grid_tot is None:
            self._grid_tot = self.grida_len + self.gridb_len
        return self._grid_tot

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

    @property
    def grida_fac(self) -> float:
        if self._grida_fac is None:
            self._grida_fac = self.grida_len / self.grid_tot
        return self._grida_fac

    @property
    def gridb_fac(self) -> float:
        if self._gridb_fac is None:
            self._gridb_fac = self.gridb_len / self.grid_tot
        return self._gridb_fac

    @property
    def vecec(self) -> 'Vector2D':
        if self._vecec is None and self.panel is not None:
            for face in self.panel.faces:
                if (face.grida is self.grida and face.gridb is self.gridb) or \
                   (face.grida is self.gridb and face.gridb is self.grida):
                    break
            face_pnte = (face.pointa + face.pointb) * 0.5
            face_pntc = face.pointc
            self._vecec = face_pnte - face_pntc
        return self._vecec

    def __repr__(self) -> str:
        return f'Edge({self.grida}, {self.gridb})'

    def __str__(self) -> str:
        return self.__repr__()


def edges_from_system(system: 'PanelSystem') -> list[InternalEdge]:
    """Create a list of unique edges from a PanelSystem.
    Args:
        system (PanelSystem): The panel system from which to extract edges.
    Returns:
        list[Edge]: A list of unique edges in the panel system.
    """
    total_edges = 0
    for panel in system.dpanels.values():
        total_edges += panel.num
    all_edges = zeros((total_edges, 2), dtype=int)
    k = 0
    for panel in system.dpanels.values():
        for i in range(-1, panel.num - 1):
            grida = panel.grids[i]
            gridb = panel.grids[i + 1]
            all_edges[k, 0] = grida.gid
            all_edges[k, 1] = gridb.gid
            k += 1
    sorted_edges = sort(all_edges, axis=1)
    unique_edges = unique(sorted_edges, axis=0)
    edges = []
    for edge in unique_edges:
        grida = system.grids[edge[0]]
        gridb = system.grids[edge[1]]
        edges.append(InternalEdge(grida, gridb))
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

def edges_parray(edges: list[InternalEdge]) -> 'NDArray':
    num_edges = len(edges)
    max_pind = None
    for edge in edges:
        if max_pind is None or (edge.panela is not None and edge.panela.ind > max_pind):
            max_pind = edge.panela.ind
        if max_pind is None or (edge.panelb is not None and edge.panelb.ind > max_pind):
            max_pind = edge.panelb.ind
    parray = zeros((num_edges, max_pind + 1), dtype=float)
    for edge in edges:
        if edge.panel is None:
            parray[edge.ind, edge.panela.ind] = edge.panelb_fac
            parray[edge.ind, edge.panelb.ind] = edge.panela_fac
        else:
            parray[edge.ind, edge.panel.ind] = 1.0
            vecg = edge.edge_point - edge.panel.pnto
            vecl = edge.face.cord.vector_to_local(vecg)
            dirl = Vector2D.from_obj(vecl)
            dmue = edge.panel.edge_velp.dot(dirl)
            parray[edge.ind, edge.panel.edge_indp] -= dmue
    return parray
