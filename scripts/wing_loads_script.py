#%%
# Import Dependencies
from IPython.display import display_markdown

from pyapm import panelsystem_from_json
from pyapm.classes.surfacestructure import SurfaceStructure

#%%
# Panel System
jsonfilepath = '../files/Test_straight_naca_2412.json'
psys = panelsystem_from_json(jsonfilepath)

pres = psys.results['Test Alpha']
display_markdown(pres)
display_markdown(pres.surface_loads)

#%%
# Plots
axl = None
axl = pres.plot_strip_lift_force_distribution(ax=axl)
axl = pres.plot_trefftz_lift_force_distribution(ax=axl)

axd = None
axd = pres.plot_strip_drag_force_distribution(ax=axd)
axd = pres.plot_trefftz_drag_force_distribution(ax=axd)

#%%
# Create Wing Structure
wstrc = SurfaceStructure(psys.srfcs[0])
wstrc.add_section_constraint(1, ksx=1.0, ksy=1.0, ksz=1.0, gsy=1.0)
wstrc.add_section_constraint(3, ksx=1.0, ksy=1.0, ksz=1.0, gsy=1.0)

#%%
# Add Wing Structure Loads
pld = wstrc.add_load(pres)
display_markdown(pld)

#%%
# Plot Loads
axf = pld.plot_forces()
axm = pld.plot_moments()
