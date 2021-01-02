#%% Import Dependencies
from pyapm.classes import PanelResult, panelsystem_from_json
from pyapm.output.msh import result_to_msh

#%% Create Panel Mesh
jsonfilepath = r'../files/Test_Wing_1.json'
psys = panelsystem_from_json(jsonfilepath)

psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%% Solve Panel Result
alpha = 10.0

pres = PanelResult('Test Case', psys)
pres.set_state(alpha = alpha)

#%% Output MSH File
mshfilepath = '../outputs/' + psys.name + '.msh'
result_to_msh(pres, mshfilepath)
