#%% Import Dependencies
from IPython.display import display_markdown
from pyapm import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh
from pyapm.classes import PanelTrim

#%% Import Geometry
jsonfilepath = '../files/Aircraft.json'
psys = panelsystem_from_json(jsonfilepath)

#%% Display System
display_markdown(psys)

#%% Display Results
for case in psys.results:
    pres = psys.results[case]
    display_markdown(pres)

#%% Mesh File Output
ptrm = psys.results['Positive 1g Cruise']
panelresult_to_msh(ptrm, '../results/Aircraft.msh')

#%% Plot Lift Distribution
axl = ptrm.plot_trefftz_lift_force_distribution()
axl = ptrm.plot_strip_lift_force_distribution(ax=axl)

#%% Plot Y Force Distribution
axy = ptrm.plot_trefftz_side_force_distribution()
axy = ptrm.plot_strip_side_force_distribution(ax=axy)

#%% Plot Drag Distribution
axd = ptrm.plot_trefftz_drag_force_distribution()
axd = ptrm.plot_strip_drag_force_distribution(ax=axd)

#%% Trim CL to 0.8
CLt = 0.8
CYt = 0.0

ptrm2 = PanelTrim(f'CL = {CLt}, CY = {CYt}', psys)
ptrm2.set_targets(CLt = CLt, CYt = CYt)
ptrm2.set_trim_loads(trmmom=False)
ptrm2.trim()

display_markdown(ptrm2)
