#%% Load Dependencies
from IPython.display import display_markdown
from pyapm import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh
from pyvlm import latticesystem_from_json

#%% Create Panel System
jsonfilepath = '../files/Test_straight.json'
psys = panelsystem_from_json(jsonfilepath)
display_markdown(psys)

#%% Create Lattice System
jsonfilepath = '../files/Test_straight.json'
lsys = latticesystem_from_json(jsonfilepath)
display_markdown(lsys)

#%% Panel Result
pres = psys.results['Test Alpha']
display_markdown(pres)

#%% Lattice Result Result
lres = lsys.results['Test Alpha']
display_markdown(lres)

#%% Plot Strip Lift Distribution
axl = None
axl = pres.plot_strip_lift_force_distribution(ax=axl)
axl = lres.plot_strip_lift_force_distribution(ax=axl)

#%% Plot Trefftz Lift Distribution
axl = None
axl = pres.plot_trefftz_lift_force_distribution(ax=axl)
axl = lres.plot_trefftz_lift_force_distribution(ax=axl)

#%% Plot Strip Drag Distribution
axd = None
axd = pres.plot_strip_drag_force_distribution(ax=axd)
axd = lres.plot_strip_drag_force_distribution(ax=axd)

#%% Plot Trefftz Drag Distribution
axd = None
axd = pres.plot_trefftz_drag_force_distribution(ax=axd)
axd = lres.plot_trefftz_drag_force_distribution(ax=axd)

#%% Plot Wash Distribution
axw = None
axw = pres.plot_trefftz_wash_distribution(ax=axw)
axw = lres.plot_trefftz_wash_distribution(ax=axw)

#%% MSH File Output
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)
