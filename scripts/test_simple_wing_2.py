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
alpha = 0.0

pres = PanelResult(f'Test Case', psys)
pres.set_density(rho=rho)
pres.set_state(alpha=alpha, speed=speed)

#%% Output MSH File
mshfilepath = '..\\outputs\\' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%% Display Result
display_markdown(pres)

#%% Print Total Loads
print(f'Total Force = {pres.nfres.nffrctot:.2f} N')
print(f'Total Moment = {pres.nfres.nfmomtot:.2f} N.m')

#%% Print Outs
print(f'sig = \n{pres.sig}')
print(f'mu = \n{pres.mu}')

#%% Loop Through Grids
for pid in sorted(psys.pnls):
    pnl = psys.pnls[pid]
    # edg_mu = pnl.edge_mu(pres.mu)
    # print(f'{pid:d}: {edg_mu}')
    qx, qy = pnl.diff_mu(pres.mu)
    qfs = pnl.crd.vector_to_local(pres.vfs)
    print(f'qx = {qx+qfs.x}')
    # print(f'qy = {qy+qfs.y}')
    # print(f'qfs.x = {qfs.x}')
    # print(f'qfs.y = {qfs.y}')
