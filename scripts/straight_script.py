#%%
# Load Dependencies
from IPython.display import display_markdown
from pyvlm import latticesystem_from_json

from pyapm import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh
from pyapm.outputs.k3d import PanelPlot, Plot

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

mshplot = Plot()
mshplot += pnlpl.panel_mesh()
mshplot.display()

sigplot = Plot()
sigplot += pnlpl.panel_sigma_plot()
sigplot.display()

muplot = Plot()
muplot += pnlpl.panel_mu_plot()
muplot.display()
