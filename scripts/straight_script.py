#%%
# Load Dependencies
from IPython.display import display_markdown
from pyvlm import latticesystem_from_json

from pyapm import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh
from pyapm.outputs.k3d import PanelPlot

#%%
# Create Panel System
jsonfilepath = '../files/Test_straight.json'
psys = panelsystem_from_json(jsonfilepath)
display_markdown(psys)

#%%
# Create Lattice System
jsonfilepath = '../files/Test_straight.json'
lsys = latticesystem_from_json(jsonfilepath)
display_markdown(lsys)

#%%
# Panel Result
pres = psys.results['Test Alpha']
display_markdown(pres)

#%%
# Lattice Result Result
lres = lsys.results['Test Alpha']
display_markdown(lres)

#%%
# Plot Lift Distribution
axl = None
axl = pres.plot_strip_lift_force_distribution(ax=axl, label='pyapm Strip')
axl = lres.plot_strip_lift_force_distribution(ax=axl, label='pyvlm Strip')
axl = pres.plot_trefftz_lift_force_distribution(ax=axl, label='pyapm Trefftz')
axl = lres.plot_trefftz_lift_force_distribution(ax=axl, label='pyvlm Trefftz')

#%%
# Plot Strip Drag Distribution
axd = None
axd = pres.plot_strip_drag_force_distribution(ax=axd, label='pyapm Strip')
axd = lres.plot_strip_drag_force_distribution(ax=axd, label='pyvlm Strip')
axd = pres.plot_trefftz_drag_force_distribution(ax=axd, label='pyapm Trefftz')
axd = lres.plot_trefftz_drag_force_distribution(ax=axd, label='pyvlm Trefftz')

#%%
# Plot Wash Distribution
axw = None
axw = pres.plot_trefftz_wash_distribution(ax=axw, label='pyapm Trefftz')
axw = lres.plot_trefftz_wash_distribution(ax=axw, label='pyvlm Trefftz')

#%%
# MSH File Output
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%%
# Display Result
pnlpl = PanelPlot(psys, pres)

mshplot = pnlpl.plot()
mshplot += pnlpl.panel_mesh()
mshplot.display()

sigplot = pnlpl.plot()
sigplot += pnlpl.panel_sigma_plot()
sigplot.display()

siggplot = pnlpl.plot()
siggplot += pnlpl.grid_sigma_plot()
siggplot.display()

muplot = pnlpl.plot()
muplot += pnlpl.panel_mu_plot()
muplot.display()

mugplot = pnlpl.plot()
mugplot += pnlpl.grid_mu_plot()
mugplot.display()

# #%%
# # Loop Through Grids
# for pid in sorted(psys.pnls):
#     pnl = psys.pnls[pid]
#     print(f'Panel ID: {pid}')
#     # pnl = psys.pnls[1007]
#     # print(f'{pnl.crd.pnt = }')
#     # for face in pnl.faces:
#     #     print(f'{face.cord.pnt = }')
#     qx_old, qy_old = pnl.diff_mu_old(pres.mu)
#     q = pnl.diff_mu(pres.mu, pres.mug)
#     # print(f'{qx_old + q.x = :.6f}, {qy_old + q.y = :.6f}')
#     print(f'{qx_old = :.6f}, {qy_old = :.6f}')
#     print(f'{q.x = :.6f}, {q.y = :.6f}')
