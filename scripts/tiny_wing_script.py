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

#%%
# Create Panel System
jsonfilepath = '../files/Tiny_Wing.json'
psys = PanelSystem.from_json(jsonfilepath)
display_markdown(psys)

#%%
# Solve Panel Result
# rho = 1.225
# speed = 50.0
# alpha = 5.0

# pres = PanelResult('Test Case', psys)
# pres.set_density(rho=rho)
# pres.set_state(alpha=alpha, speed=speed)

pres = psys.results['Alpha 5 deg, Speed 50 m/s']

#%%
# Assemble and Solve
psys.assemble_panels_phi()
# psys.assemble_horseshoes_phi()
psys.solve_system()

#%%
# Output MSH File
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%%
# Display Result
display_markdown(pres)
display_markdown(pres.surface_loads)

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
for dpanel in psys.surfaces[0].strips[num_strips//2].dpanels:
    xplst.append(float(dpanel.pnto.x))
    muplst.append(float(pres.mud[dpanel.ind]))
    dpanels.append(dpanel)

xelst = []
muelst = []
for edge in psys.edges:
    if hasattr(edge, 'panel'):
        if edge.panel in dpanels:
            xelst.append(float(edge.edge_point.x))
            muelst.append(float(pres.mue[edge.ind]))
    elif hasattr(edge, 'panela') and hasattr(edge, 'panelb'):
        if edge.panela in dpanels or edge.panelb in dpanels:
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

xflst = []
vxflst = []
for dpanel in dpanels:
    for facet in dpanel.facets:
        xflst.append(float(facet.cord.pnt.x))
        vxflst.append(float(pres.fres.fvel.x[facet.ind]))

print(f'xplst = {xplst}')
print(f'muplst = {muplst}')
print(f'xelst = {xelst}')
print(f'muelst = {muelst}')
print(f'xglst = {xglst}')
print(f'muglst = {muglst}')

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(xplst, muplst, marker='o')
ax.scatter(xelst, muelst, marker='x', color='red')
ax.scatter(xglst, muglst, marker='+', color='green')
_ = ax.set_title('Mu Distribution Along Chord')

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(xflst, vxflst)
_ = ax.set_title('Facet Vx Distribution Along Chord')

#%%
# Display Result
pnlpl = PanelPlot(psys, pres)

mshplot = pnlpl.plot()
mshplot += pnlpl.panel_mesh()
mshplot += text2d("Panel Mesh", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
mshplot.display()

sigplot = pnlpl.plot()
sigplot += pnlpl.panel_sigma_plot()
sigplot += text2d("Panel Sigma Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
sigplot.display()

siggplot = pnlpl.plot()
siggplot += pnlpl.grid_sigma_plot()
siggplot += text2d("Grid Sigma Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
siggplot.display()

mudplot = pnlpl.plot()
mudplot += pnlpl.panel_mud_plot()
mudplot += text2d("Panel Mud Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
mudplot.display()

mugplot = pnlpl.plot()
mugplot += pnlpl.grid_mu_plot()
mugplot += text2d("Grid Mu Plot", position=(0.5, 0.5), is_html=True, label_box=False, color=0x000000)
mugplot.display()

fvxplot = pnlpl.plot()
fvxplot += pnlpl.face_vx_plot(color_range=(-80.0, 80.0))
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
ffrcplot += pnlpl.face_force_plot(scale=0.05, head_size=0.05, line_width=0.001)
ffrcplot += text2d("Face Force Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
ffrcplot.display()

# #%%
# # Print Output
# for dpanel in psys.dpanels.values():
#     for facet in dpanel.facets:
#         if facet.cord.dirz.dot(dpanel.crd.dirz) < 0.0:
#             print(f'{dpanel = }')
#             for grid in dpanel.grids:
#                 print(f'  {grid = }')
#             for panel_edge in dpanel.panel_edges:
#                 print(f'  {panel_edge = }')
#                 print(f'  {panel_edge.edge_point = }')
#             for facet in dpanel.facets:
#                 print(f'  {facet = }')
#                 print(f'  {facet.cord.dirz = }')
#             print(f'{dpanel.crd.dirz = }')

#%%
