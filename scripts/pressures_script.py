#%%
# Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import PanelSystem
from pyapm.outputs.k3d import PanelPlot
from pyapm.tools.points import fetch_pids_ttol, point_results
from pygeom.geom3d import Vector

#%%
# Create Panel System
jsonfilepath = '../files/Test_Simple_Wing_2.json'
psys = PanelSystem.from_json(jsonfilepath)
display_markdown(psys)

#%%
# Display Panel Result
pres = psys.results['Test Case']
display_markdown(pres)

#%%
# Determine Point Pressures
pnts = Vector.zeros((7, 2))
pnts[0, 0] = Vector(0.25, 0.05, 0.05)
pnts[1, 0] = Vector(0.25, 0.05, -0.05)
pnts[2, 0] = Vector(0.25, 4.5, 0.05)
pnts[3, 0] = Vector(0.25, -0.05, 0.05)
pnts[4, 0] = Vector(0.25, -0.05, -0.05)
pnts[5, 0] = Vector(0.25, -4.5, 0.05)
pnts[6, 0] = Vector(0.25, -6.5, 0.05)
pnts[0, 1] = Vector(0.75, 0.05, 0.05)
pnts[1, 1] = Vector(0.75, 0.05, -0.05)
pnts[2, 1] = Vector(0.75, 4.5, 0.05)
pnts[3, 1] = Vector(0.75, -6.5, 0.05)
pnts[4, 1] = Vector(0.75, -0.05, 0.05)
pnts[5, 1] = Vector(0.75, -0.05, -0.05)
pnts[6, 1] = Vector(0.75, -4.5, 0.05)

pids, chkz = fetch_pids_ttol(pnts, psys, ztol=0.1)
print(f'pids = \n{pids}\n')
print(f'chkz = \n{chkz}\n')

prs = point_results(pnts, psys, pids, chkz, pres.nfres.nfprs)
print(f'prs = \n{prs}\n')

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
