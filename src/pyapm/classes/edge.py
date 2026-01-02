from typing import TYPE_CHECKING

from numpy import argwhere, sort, unique, zeros
from pygeom.geom2d import Vector2D
from pygeom.geom3d import Coordinate

from .face import Face
from .grid import Vertex

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector

    from .grid import Grid
    from .panel import Panel
    from .panelsystem import PanelSystem
    from .wakepanel import WakePanel


class PanelEdge():
    grida: 'Grid' = None
    gridb: 'Grid' = None
    panel: 'Panel' = None
    mesh_edge: 'MeshEdge' = None
    _face: Face = None
    _length: float = None

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

    @property
    def length(self) -> float:
        if self._length is None:
            edge_vec = self.gridb - self.grida
            self._length = edge_vec.return_magnitude()
        return self._length

    def is_internal(self) -> bool:
        if self.mesh_edge is not None:
            return isinstance(self.mesh_edge, InternalEdge)
        else:
            raise ValueError('Mesh edge not assigned to panel edge.')

    def not_internal(self) -> bool:
        return not self.is_internal()

    def __repr__(self) -> str:
        return f'PanelEdge({self.grida}, {self.gridb}, {self.panel})'

    def __str__(self) -> str:
        return self.__repr__()


# class TrailingPanelEdge():
#     grida: 'Grid' = None
#     gridb: 'Grid' = None
#     panel: 'Panel' = None
#     mesh_edge: 'MeshEdge' = None
#     _face: Face = None

#     def __init__(self, grida: 'Grid', gridb: 'Grid', panel: 'Panel') -> None:
#         self.grida = grida
#         self.gridb = gridb
#         self.panel = panel

#     @property
#     def face(self) -> Face:
#         if self._face is None:
#             self._face = Face(self.grida, self.gridb, self.panel)
#             self._face.set_dirl(self.panel.crd.dirx)
#         return self._face

#     @property
#     def edge_point(self) -> 'Vector':
#         if self.mesh_edge is not None:
#             return self.mesh_edge.edge_point
#         else:
#             return None

#     @property
#     def ind(self) -> int:
#         if self.mesh_edge is not None:
#             return self.mesh_edge.ind
#         else:
#             return None

#     def is_internal(self) -> bool:
#         if self.mesh_edge is not None:
#             return isinstance(self.mesh_edge, InternalEdge)
#         else:
#             raise ValueError('Mesh edge not assigned to panel edge.')

#     def not_internal(self) -> bool:
#         return not self.is_internal()

#     def __repr__(self) -> str:
#         return f'PanelEdge({self.grida}, {self.gridb}, {self.panel})'

#     def __str__(self) -> str:
#         return self.__repr__()


class BoundEdge():
    grida: 'Grid' = None
    gridb: 'Grid' = None
    wake_panel: 'WakePanel' = None
    mesh_edge: 'WakeBoundEdge' = None

    def __init__(self, grida: 'Grid', gridb: 'Grid', wake_panel: 'WakePanel') -> None:
        self.grida = grida
        self.gridb = gridb
        self.wake_panel = wake_panel

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
        return f'BoundEdge({self.grida}, {self.gridb}, {self.wake_panel})'

    def __str__(self) -> str:
        return self.__repr__()


class VortexEdge():
    grida: 'Grid' = None
    gridb: 'Grid' = None
    wake_panel: 'Panel' = None
    mesh_edge: 'MeshEdge' = None

    def __init__(self, grida: 'Grid', gridb: 'Grid',
                 wake_panel: 'WakePanel') -> None:
        self.grida = grida
        self.gridb = gridb
        self.wake_panel = wake_panel

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
        return f'VortexEdge({self.grida}, {self.gridb}, {self.wake_panel})'

    def __str__(self) -> str:
        return self.__repr__()


class MeshEdge():
    ind: int = None
    edge_type: str = None
    _edge_point: 'Vector' = None

    def __init__(self) -> None:
        pass

    @property
    def edge_point(self) -> 'Vector':
        return self._edge_point


class BoundaryEdge(MeshEdge):
    panel_edge: PanelEdge = None
    _mid_point: 'Vector' = None
    _length: float = None
    _Dmue: float = None

    def __init__(self, panel_edge: PanelEdge) -> None:
        self.panel_edge = panel_edge
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

    @property
    def length(self) -> float:
        if self._length is None:
            edge_vec = self.panel_edge.gridb - self.panel_edge.grida
            self._length = edge_vec.return_magnitude()
        return self._length

    def return_Dmue(self) -> float:
        vecg = self.panel_edge.edge_point - self.panel.pnto
        vecl = self.face.cord.vector_to_local(vecg)
        dirl = Vector2D.from_obj(vecl)
        Dmue = self.panel.edge_velp.dot(dirl)
        return Dmue

    def __repr__(self) -> str:
        return f'BoundaryEdge({self.panel_edge})'

    def __str__(self) -> str:
        return self.__repr__()


class BluntEdge(MeshEdge):
    panel_edge: PanelEdge = None
    adjacent_edge: PanelEdge = None
    _mid_point: 'Vector' = None

    def __init__(self, panel_edge: PanelEdge,
                 adjacent_edge: PanelEdge) -> None:
        self.panel_edge = panel_edge
        self.adjacent_edge = adjacent_edge
        self.panel_edge.mesh_edge = self
        self.adjacent_edge.mesh_edge = self

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

    def return_Dmue(self) -> float:
        vecg = self.panel_edge.edge_point - self.panel.pnto
        vecl = self.face.cord.vector_to_local(vecg)
        dirl = Vector2D.from_obj(vecl)
        Dmue = self.panel.edge_velp.dot(dirl)
        return Dmue

    def __repr__(self) -> str:
        return f'BluntEdge({self.panel_edge}, )'

    def __str__(self) -> str:
        return self.__repr__()



class InternalEdge(MeshEdge):
    panel_edgea: PanelEdge = None
    panel_edgeb: PanelEdge = None
    _edge_type: str = None
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

    def __init__(self, panel_edgea: PanelEdge,
                 panel_edgeb: PanelEdge) -> None:
        self.panel_edgea = panel_edgea
        self.panel_edgea.mesh_edge = self
        self.panel_edgeb = panel_edgeb
        self.panel_edgeb.mesh_edge = self

    @property
    def edge_type(self) -> str:
        if self._edge_type is None:
            tepanela = hasattr(self.panela, '_adj_inds')
            tepanelb = hasattr(self.panelb, '_adj_inds')
            if tepanela and tepanelb:
                self._edge_type = 'trailing edge'
            elif tepanela or tepanelb:
                self._edge_type = 'blunt edge'
            else:
                self._edge_type = 'internal edge'
        return self._edge_type

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


class WakeBoundEdge(MeshEdge):
    panel_edge: PanelEdge = None
    adjacent_edge: PanelEdge = None
    bound_edge: BoundEdge = None
    edge_type: str = 'wake_bound'
    _mid_point: 'Vector' = None
    _edge_point: Vertex = None

    def __init__(self, panel_edge: 'PanelEdge',
                 adjacent_edge: 'PanelEdge',
                 bound_edge: 'BoundEdge') -> None:
        self.panel_edge = panel_edge
        self.panel_edge.mesh_edge = self
        self.adjacent_edge = adjacent_edge
        self.bound_edge = bound_edge
        self.bound_edge.mesh_edge = self

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

    def return_Dmue(self) -> float:
        vecg = self.panel_edge.edge_point - self.panel.pnto
        vecl = self.face.cord.vector_to_local(vecg)
        dirl = Vector2D.from_obj(vecl)
        Dmue = self.panel.edge_velp.dot(dirl)
        return Dmue

    def __repr__(self) -> str:
        return f'WakeBoundEdge({self.panel_edge}, {self.adjacent_edge}, {self.bound_edge})'

    def __str__(self) -> str:
        return self.__repr__()


class WakeVortexEdge(MeshEdge):
    panel_edge: PanelEdge = None
    adjacent_edge: PanelEdge = None
    vortex_edge: VortexEdge = None
    edge_type: str = 'wake_vortex'
    _mid_point: 'Vector' = None
    _edge_point: Vertex = None
    _panela_len: float = None
    _panelb_len: float = None
    _panel_tot: float = None
    _panela_fac: float = None
    _panelb_fac: float = None
    _wake_fac: int = None

    def __init__(self, panel_edge: 'PanelEdge',
                 adjacent_edge: 'PanelEdge',
                 vortex_edge: 'VortexEdge') -> None:
        self.panel_edge = panel_edge
        self.panel_edge.mesh_edge = self
        self.adjacent_edge = adjacent_edge
        self.vortex_edge = vortex_edge
        self.vortex_edge.mesh_edge = self

    @property
    def panel(self) -> 'Panel':
        return self.panel_edge.panel

    @property
    def face(self) -> 'Face':
        return self.panel_edge.face

    @property
    def panela(self) -> 'Panel':
        return self.panel_edge.panel

    @property
    def panelb(self) -> 'Panel':
        return self.adjacent_edge.panel

    @property
    def panelw(self) -> 'WakePanel':
        return self.vortex_edge.wake_panel

    @property
    def mid_point(self) -> 'Vector':
        if self._mid_point is None:
            self._mid_point = 0.5 * (self.panel_edge.grida + self.panel_edge.gridb)
        return self._mid_point

    @property
    def edge_point(self) -> 'Vector':
        return self.mid_point

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

    @property
    def wake_fac(self) -> float:
        if self._wake_fac is None:
            if self.panel_edge.grida is self.vortex_edge.grida and self.panel_edge.gridb is self.vortex_edge.gridb:
                self._wake_fac = -self.panela_fac
            elif self.panel_edge.grida is self.vortex_edge.gridb and self.panel_edge.gridb is self.vortex_edge.grida:
                self._wake_fac = self.panela_fac
        return self._wake_fac

    def return_Dmue(self) -> float:
        vecg = self.panel_edge.edge_point - self.panel.pnto
        vecl = self.face.cord.vector_to_local(vecg)
        dirl = Vector2D.from_obj(vecl)
        Dmue = self.panel.edge_velp.dot(dirl)
        return Dmue

    def __repr__(self) -> str:
        return f'WakeVortexEdge({self.panel_edge}, {self.adjacent_edge}, {self.vortex_edge})'

    def __str__(self) -> str:
        return self.__repr__()


def edges_from_system(system: 'PanelSystem') -> list[MeshEdge]:
    """Create a list of unique edges from a PanelSystem.
    Args:
        system (PanelSystem): The panel system from which to extract edges.
    Returns:
        list[Edge]: A list of unique edges in the panel system.
    """

    all_edges: list[PanelEdge | BoundEdge | VortexEdge] = []
    for dpanel in system.dpanels.values():
        all_edges.extend(dpanel.panel_edges)
    for wpanel in system.wpanels.values():
        all_edges.extend(wpanel.panel_edges)
    num_edges = len(all_edges)
    edge_gids = zeros((num_edges, 2), dtype=int)
    for i, edge in enumerate(all_edges):
        edge_gids[i, 0] = edge.grida.gid
        edge_gids[i, 1] = edge.gridb.gid

    sorted_edges = sort(edge_gids, axis=1)
    unique_edges, edge_inverse = unique(sorted_edges, axis=0,
                                        return_inverse=True)

    edges = []
    for i in range(unique_edges.shape[0]):
        edge_inds = argwhere(edge_inverse == i).flatten()
        ind_edges = [all_edges[ind] for ind in edge_inds]
        if len(edge_inds) == 1:
            panel_edge = ind_edges[0]
            if isinstance(panel_edge, PanelEdge):
                edge = BoundaryEdge(panel_edge)
                edges.append(edge)
        elif len(edge_inds) == 2:
            panel_edgea = ind_edges[0]
            panel_edgeb = ind_edges[1]
            if isinstance(panel_edgea, PanelEdge) and isinstance(panel_edgeb, PanelEdge):
                tpanela = hasattr(panel_edgea.panel, '_adj_inds')
                tpanelb = hasattr(panel_edgeb.panel, '_adj_inds')
                if tpanela and tpanelb:
                    edge = InternalEdge(panel_edgea, panel_edgeb)
                    edges.append(edge)
                elif tpanela and not tpanelb:
                    edge = BluntEdge(panel_edgeb, panel_edgea)
                    edges.append(edge)
                elif not tpanela and tpanelb:
                    edge = BluntEdge(panel_edgea, panel_edgeb)
                    edges.append(edge)
                else:
                    edge = InternalEdge(panel_edgea, panel_edgeb)
                    edges.append(edge)
        elif len(edge_inds) == 3:
            bound_edge = None
            vortex_edge = None
            panel_edges = []
            for ind_edge in ind_edges:
                if isinstance(ind_edge, BoundEdge):
                    bound_edge = ind_edge
                elif isinstance(ind_edge, VortexEdge):
                    vortex_edge = ind_edge
                else:
                    panel_edges.append(ind_edge)
            if len(panel_edges) != 2:
                raise ValueError('Error identifying edges for wake edge.')
            if bound_edge is None and vortex_edge is None:
                raise ValueError('Error identifying edges for wake edge.')
            if bound_edge is not None:
                panel_edgea = panel_edges[0]
                panel_edgeb = panel_edges[1]
                edge = WakeBoundEdge(panel_edgea, panel_edgeb, bound_edge)
                edges.append(edge)
                edge = WakeBoundEdge(panel_edgeb, panel_edgea, bound_edge)
                edges.append(edge)
            elif vortex_edge is not None:
                panel_edgea = panel_edges[0]
                panel_edgeb = panel_edges[1]
                edge = WakeVortexEdge(panel_edgea, panel_edgeb, vortex_edge)
                edges.append(edge)
                edge = WakeVortexEdge(panel_edgeb, panel_edgea, vortex_edge)
                edges.append(edge)
        else:
            raise ValueError('Error identifying edges for panel system.')

    return edges

def edges_parrays(mesh_edges: list[MeshEdge], num_dpanels: int,
                  num_wpanels: int) -> tuple['NDArray', 'NDArray']:
    num_edges = len(mesh_edges)
    parrayd = zeros((num_edges, num_dpanels), dtype=float)
    parrayw = zeros((num_edges, num_wpanels), dtype=float)
    for mesh_edge in mesh_edges:
        if isinstance(mesh_edge, (BoundaryEdge, BluntEdge, WakeBoundEdge)):
            parrayd[mesh_edge.ind, mesh_edge.panel.ind] = 1.0
            Dmue = mesh_edge.return_Dmue()
            parrayd[mesh_edge.ind, mesh_edge.panel.edge_indp] -= Dmue
        elif isinstance(mesh_edge, InternalEdge):
            parrayd[mesh_edge.ind, mesh_edge.panela.ind] = mesh_edge.panelb_fac
            parrayd[mesh_edge.ind, mesh_edge.panelb.ind] = mesh_edge.panela_fac
        elif isinstance(mesh_edge, WakeVortexEdge):
            parrayd[mesh_edge.ind, mesh_edge.panela.ind] = mesh_edge.panelb_fac
            parrayd[mesh_edge.ind, mesh_edge.panelb.ind] = mesh_edge.panela_fac
            parrayw[mesh_edge.ind, mesh_edge.panelw.ind] = mesh_edge.wake_fac
    return parrayd, parrayw
