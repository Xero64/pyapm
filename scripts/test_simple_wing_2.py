#%% Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import panelsystem_from_json, PanelResult
from pyapm.output.msh import panelresult_to_msh

#%% Create Panel System
jsonfilepath = r'../files/Test_Simple_Wing_2.json'
psys = panelsystem_from_json(jsonfilepath)

psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%% Solve Panel Result
rho = 1.225
speed = 1.0
alpha = 5.0

pres = PanelResult(f'Test Case', psys)
pres.set_density(rho=rho)
pres.set_state(alpha=alpha, speed=speed)

#%% Output MSH File
mshfilepath = '..\\outputs\\' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%% Display Result
display_markdown(pres)
display_markdown(pres.surface_loads)

#%% Print Outs
print(f'sig = \n{pres.sig}')
print(f'mu = \n{pres.mu}')

#%% Loop Through Grids
for pid in sorted(psys.pnls):
    pnl = psys.pnls[pid]
    qx, qy = pnl.diff_mu(pres.mu)
    qfs = pnl.crd.vector_to_local(pres.vfs)
    print(f'qx = {qx+qfs.x}, qy = {qy+qfs.y}')

#%% Solve Panel Result
rho = 1.225
speed = 1.0
alpha = 5.0
pbo2V = 0.3

pres = PanelResult(f'Test Case', psys)
pres.set_density(rho=rho)
pres.set_state(alpha=alpha, speed=speed, pbo2V=pbo2V)

#%% Output MSH File
mshfilepath = '..\\outputs\\Test Simple Wing 2 Roll.msh'
panelresult_to_msh(pres, mshfilepath)
