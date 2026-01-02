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
# # Print Out Edges
# from numpy import concatenate, unique, bincount
# from pygeom.geom2d import Vector2D

# vertex = psys.vertices[11]
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
# # panel = panels[0]
# # print(f'{panel = }')

#     indv = panel.vertices.index(vertex)
#     inda = indv
#     indb = indv + 1
#     if indv + 1 == panel.num:
#         indb = 0

#     # print(f'{indv = }')
#     # print(f'{inda = }')
#     # print(f'{indb = }')
#     # print(f'panel.edge_indps[inda] = \n{panel.edge_indps[inda]}\n')
#     # print(f'panel.edge_indps[indb] = \n{panel.edge_indps[indb]}\n')
#     # print(f'panel.edge_velps[inda] = \n{panel.edge_velps[inda]}\n')
#     # print(f'panel.edge_velps[indb] = \n{panel.edge_velps[indb]}\n')

#     coninds = concatenate((panel.edge_indps[inda], panel.edge_indps[indb]))
#     convelx = concatenate((panel.edge_velps[inda].x, panel.edge_velps[indb].x))
#     convely = concatenate((panel.edge_velps[inda].y, panel.edge_velps[indb].y))

#     # print(f'{coninds = }')
#     # print(f'{convelx = }')
#     # print(f'{convely = }')

#     unqinds, invinds = unique(coninds, return_inverse=True)
#     print(f'{unqinds = }')
#     # print(f'{invinds = }')

#     sumvelx = bincount(invinds, weights=convelx)
#     sumvely = bincount(invinds, weights=convely)

#     velv = Vector2D(sumvelx, sumvely)
#     print(f'velv = \n{velv}\n')

#     mup = pres.mud[panel.ind]
#     print(f'{mup = }')

#     # muv = pres.muv[vertex.ind]
#     # print(f'{muv = }')

#     mups = pres.mud[unqinds]
#     print(f'{mups = }')

#     vecg = vertex - panel.pnto
#     print(f'vecg = {vecg}')

#     vecl = panel.crd.vector_to_local(vecg)
#     print(f'vecl = {vecl}')

#     dirl = Vector2D.from_obj(vecl)
#     print(f'dirl = {dirl}')

#     facs = dirl.dot(velv)
#     print(f'{facs = }')

#     facs = facs / facs.sum()
#     print(f'{facs = }')

#     muv_check = facs @ mups
#     print(f'{muv_check = }')

#     muv_check_sum += muv_check

# print('\n========================================\n')

# print(f'{muv_check_sum = }')
# print(f'{muv_check_sum / len(panels) = }')
# print(f'{muv = }')
# print(f'{mups_avg = }')

# # %%
