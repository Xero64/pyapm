#%%
# Import Dependencies
from IPython.display import display_markdown
from k3d import text2d
from pyapm.classes import PanelSystem
from pyapm.outputs.k3d import PanelPlot

#%%
# Create Panel System
jsonfilepath = '../files/Tiny_Wing_Blunt.json'
psys = PanelSystem.from_json(jsonfilepath)
display_markdown(psys)

#%%
# Solve Panel Result
pres = psys.results['Alpha 5 deg, Speed 50 m/s']
display_markdown(pres)

print('Surface Loads:')

display_markdown(pres.surface_loads)

#%%
# Display Result
pnlpl = PanelPlot(psys, pres)

mudplot = pnlpl.plot()
mudplot += pnlpl.panel_mu_plot()
mudplot += text2d("Panel Mud Plot", position=(0.5, 0.95), is_html=True, label_box=False, color=0x000000)
mudplot.display()
