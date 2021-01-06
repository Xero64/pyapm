#%% Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import PanelResult, panelsystem_from_json
from pyapm.output.msh import panelresult_to_msh

#%% Create Panel Mesh
jsonfilepath = r'../files/Prandtl-D2.json'
psys = panelsystem_from_json(jsonfilepath)
display_markdown(psys)

#%% Assembly and Solution
psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%% Solve Panel Result
alpha = 0.0
speed = 13.0
rho = 1.145

pres = PanelResult('Design Point', psys)
pres.set_density(rho=rho)
pres.set_state(alpha=alpha, speed=speed)

display_markdown(pres)
display_markdown(pres.surface_loads)

mshfilepath = '..\\outputs\\' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)
