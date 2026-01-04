#%%
# Import Dependencies
from IPython.display import display_markdown
from k3d import text2d
from matplotlib.pyplot import figure
from numpy.linalg import norm
from pyapm.classes import PanelSystem
from pyapm.classes.panel import Panel
from pyapm.outputs.k3d import PanelPlot
from pyapm.outputs.msh import panelresult_to_msh
from pyvlm.classes import LatticeSystem
from pyapm.methods.cpm.inputs.pyvlm import constant_system_from_lattice_system

#%%
# Create Panel System
jsonfilepath = '../files/Tiny_Wing_Blunt.json'
psys = PanelSystem.from_json(jsonfilepath)
display_markdown(psys)

lsys = LatticeSystem.from_json(jsonfilepath)
display_markdown(lsys)

csys = constant_system_from_lattice_system(lsys)
display_markdown(csys)

# psys.assemble_panels_phi()
# psys.solve_system()

#%%
# Solve Panel Result
pres = psys.results['Alpha 5 deg, Speed 50 m/s']
display_markdown(pres)

cres = csys.results['Alpha 5 deg, Speed 50 m/s']
display_markdown(cres)

display_markdown(pres.surface_loads)

display_markdown(cres.surface_loads)

#%%
# Output MSH File
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%%
# Print Outs
print(f'sigd = \n{pres.sigd}')
print(f'mud = \n{pres.mud}')
print(f'mue = \n{pres.mue}')
print(f'mug = \n{pres.mug}')
print(f'muw = \n{pres.muw}')
print(f'phi = \n{pres.phi}')
print(f'{norm(pres.phi)}')

#%%
# Plots
num_strips = len(psys.surfaces[0].strips)

dpanels: list['Panel'] = []
xplst = []
muplst = []
for dpanel in psys.surfaces[0].strips[num_strips // 2].dpanels:
    xplst.append(float(dpanel.pnto.x))
    muplst.append(float(pres.mud[dpanel.ind]))
    dpanels.append(dpanel)

xelst = []
muelst = []
for edge in psys.edges:
    if hasattr(edge, 'panel'):
        if edge.panel in dpanels:
            if abs(edge.edge_point.y - edge.panel.pnto.y) < 1e-8:
                xelst.append(float(edge.edge_point.x))
                muelst.append(float(pres.mue[edge.ind]))
    elif hasattr(edge, 'panela') and hasattr(edge, 'panelb'):
        if edge.panela in dpanels or edge.panelb in dpanels:
            if abs(edge.edge_point.y - edge.panela.pnto.y) < 1e-8:
                xelst.append(float(edge.edge_point.x))
                muelst.append(float(pres.mue[edge.ind]))

xglst = []
muglst = []
for grid in psys.grids.values():
    found = False
    for panel in grid.panels:
        if panel in dpanels:
            found = True
    if found:
        xglst.append(float(grid.x))
        muglst.append(float(pres.mug[grid.ind]))

xvlst = []
muvlst = []
for vertex in psys.vertices:
    found = False
    for panel in vertex.panels:
        if panel in dpanels:
            found = True
    if found:
        xvlst.append(float(vertex.x))
        muvlst.append(float(pres.muv[vertex.ind]))

# xflst = []
# vxflst = []
# for dpanel in dpanels:
#     for facet in dpanel.facets:
#         xflst.append(float(facet.cord.pnt.x))
#         vxflst.append(float(pres.fres.fvel.x[facet.ind]))

print(f'xplst = {xplst}')
print(f'muplst = {muplst}')
print(f'xelst = {xelst}')
print(f'muelst = {muelst}')
print(f'xglst = {xglst}')
print(f'muglst = {muglst}')
print(f'xvlst = {xvlst}')
print(f'muvlst = {muvlst}')

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(xplst, muplst, marker='o')
ax.scatter(xelst, muelst, marker='x', color='red')
# ax.scatter(xglst, muglst, marker='+', color='green')
ax.scatter(xvlst, muvlst, marker='.', color='orange')
_ = ax.set_title('Mu Distribution Along Chord')

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(xplst, muplst, marker='o')
ax.scatter(xelst, muelst, marker='x', color='red')
# ax.scatter(xglst, muglst, marker='+', color='green')
ax.scatter(xvlst, muvlst, marker='.', color='orange')
ax.set_xlim(0.7, 0.8)
ax.set_ylim(4.0, 6.0)
_ = ax.set_title('Mu Distribution Along Chord')

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(xplst, muplst, marker='o')
ax.scatter(xelst, muelst, marker='x', color='red')
# ax.scatter(xglst, muglst, marker='+', color='green')
ax.scatter(xvlst, muvlst, marker='.', color='orange')
ax.set_xlim(0.7, 0.8)
ax.set_ylim(-9.0, -7.0)
_ = ax.set_title('Mu Distribution Along Chord')

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(xflst, vxflst)
# _ = ax.set_title('Facet Vx Distribution Along Chord')

#%%
# Display Result
pnlpl = PanelPlot(psys, pres)

# mshplot = pnlpl.plot()
# mshplot += pnlpl.panel_mesh()
# mshplot += text2d("Panel Mesh", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
# mshplot.display()

# sigplot = pnlpl.plot()
# sigplot += pnlpl.panel_sigma_plot()
# sigplot += text2d("Panel Sigma Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
# sigplot.display()

# sigvplot = pnlpl.plot()
# sigvplot += pnlpl.vertex_sigma_plot()
# sigvplot += text2d("Vertex Sigma Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
# sigvplot.display()

mudplot = pnlpl.plot()
mudplot += pnlpl.panel_mu_plot()
mudplot += text2d("Panel Mud Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
mudplot.display()

mugplot = pnlpl.plot()
mugplot += pnlpl.vertex_mu_plot()
mugplot += text2d("Grid Mu Plot", position=(0.5, 0.5), is_html=True, label_box=False, color=0x000000)
mugplot.display()

fvxplot = pnlpl.plot()
fvxplot += pnlpl.face_vx_plot(color_range=(0.0, 80.0))
fvxplot += text2d("Face Vx Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
fvxplot.display()

fvyplot = pnlpl.plot()
fvyplot += pnlpl.face_vy_plot(color_range=(-10.0, 10.0))
fvyplot += text2d("Face Vy Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
fvyplot.display()

fcpplot = pnlpl.plot()
fcpplot += pnlpl.face_cp_plot(color_range=(-2.0, 1.0))
fcpplot += text2d("Face Cp Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
fcpplot.display()

ffrcplot = pnlpl.plot()
ffrcplot += pnlpl.panel_mesh()
ffrcplot += pnlpl.face_force_plot(scale=0.5, head_size=0.05, line_width=0.001)
ffrcplot += text2d("Face Force Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
ffrcplot.display()

# fcpplot = pnlpl.plot()
# fcpplot += pnlpl.panel_mesh()
# fcpplot += pnlpl.face_cpvec_plot(scale=0.25, head_size=0.05, line_width=0.001)
# fcpplot += text2d("Face Cp Vector Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
# fcpplot.display()

# #%%
# # Find Vertex
# for dpanel in dpanels:
#     if dpanel.pnto.x > 0.2 and dpanel.pnto.x < 0.3:
#         if dpanel.pnto.z > 0.0:
#             vertex = dpanel.vertices[0]
#             break

# #%%
# # Print Out Edges
# from numpy import concatenate, unique, bincount
# from pygeom.geom2d import Vector2D

# # vertex = psys.vertices[11]
# print(f'{vertex = }')
# print(f'{vertex.panels = }')

# muv = pres.muv[vertex.ind]
# print(f'{muv = }')

# panels = list(vertex.panels)

# muv_check_sum = 0.0

# pinds = [panel.ind for panel in panels]

# mups = pres.mud[pinds]
# print(f'{mups = }')

# mups_avg = sum(mups) / len(mups)
# print(f'{mups_avg = }')

# for panel in panels:
#     print('\n----------------------------------------\n')
#     print(f'{panel = }')

#     indv = panel.vertices.index(vertex)
#     inda = indv
#     indb = indv + 1
#     if indv + 1 == panel.num:
#         indb = 0

#     mesh_edgea = panel.mesh_edges[inda]
#     mesh_edgeb = panel.mesh_edges[indb]
#     indpa = mesh_edgea.indps
#     indpb = mesh_edgeb.indps
#     facpa = mesh_edgea.facps
#     facpb = mesh_edgeb.facps
#     veca = panel.crd.point_to_local(mesh_edgea.edge_point)
#     vecb = panel.crd.point_to_local(mesh_edgeb.edge_point)
#     vecv = panel.crd.point_to_local(vertex)
#     dira = Vector2D.from_obj(veca)
#     dirb = Vector2D.from_obj(vecb)
#     dirv = Vector2D.from_obj(vecv)
#     denom = dira.cross(dirb)
#     muafac = facpa * dirv.cross(dirb) / denom
#     mubfac = facpb * dira.cross(dirv) / denom
#     mupfac = denom + dirv.cross(dira) + dirb.cross(dirv)
#     mupfac = mupfac / denom
#     print(f'{indv = }')
#     print(f'{indpa = }')
#     print(f'{indpb = }')
#     print(f'{mupfac = }')
#     print(f'{muafac = }')
#     print(f'{mubfac = }')
#     conindps = concatenate(([panel.ind], indpa, indpb))
#     confacps = concatenate(([mupfac], muafac, mubfac))
#     uniqindps, invindps = unique(conindps, return_inverse=True)
#     uniqfacps = bincount(invindps, weights=confacps)

#     print(f'{conindps = }')
#     print(f'{confacps = }')
#     print(f'{uniqindps = }')
#     print(f'{uniqfacps = }')

#     mups = pres.mud[uniqindps]

#     muv_check = uniqfacps @ mups
#     print(f'{muv_check = }')

#     muv_check_sum += muv_check

#     mue1 = pres.mue[mesh_edgea.ind]
#     mue2 = pres.mue[mesh_edgeb.ind]
#     mup = pres.mud[panel.ind]
#     print(f'{mue1 = }')
#     print(f'{mue2 = }')
#     print(f'{mup = }')

#     mue1_chk = pres.mud[mesh_edgea.indps] @ facpa
#     mue2_chk = pres.mud[mesh_edgeb.indps] @ facpb

#     print(f'{mue1_chk = }')
#     print(f'{mue2_chk = }')

# print('\n========================================\n')

# print(f'{muv_check_sum = }')
# print(f'{muv_check_sum / len(panels) = }')
# print(f'{muv = }')
# print(f'{mups_avg = }')
