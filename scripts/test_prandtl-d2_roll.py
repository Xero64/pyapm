#%% Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import PanelResult, panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh

#%% Create Panel Mesh
jsonfilepath = '../files/Prandtl-D2.json'
psys = panelsystem_from_json(jsonfilepath)
psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%% Solve Panel Result
alpha = 0.0
speed = 12.9
rho = 1.145

pres = PanelResult('Design Point', psys)
pres.set_density(rho=rho)
pres.set_state(alpha=alpha, speed=speed)

display_markdown(pres)
display_markdown(pres.surface_loads)
display_markdown(pres.stability_derivatives)

mshfilepath = '..\\outputs\\' + pres.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%% Solve Panel Result
alpha = 0.0
speed = 12.9
pbo2V = 0.01
rho = 1.145

pres = PanelResult('Roll Case', psys)
pres.set_density(rho=rho)
pres.set_state(alpha=alpha, speed=speed, pbo2V=pbo2V)

display_markdown(pres)
display_markdown(pres.surface_loads)
display_markdown(pres.stability_derivatives)

mshfilepath = '../outputs/' + pres.name + '.msh'
panelresult_to_msh(pres, mshfilepath)
