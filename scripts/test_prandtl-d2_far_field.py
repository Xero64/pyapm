#%% Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import PanelResult, panelsystem_from_json
from pyapm.output.msh import panelresult_to_msh

#%% Create Panel Mesh
jsonfilepath = r'../files/Prandtl-D2.json'
psys = panelsystem_from_json(jsonfilepath)
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

#%% Panel Drag
yst = []
drg = []
i = 0
drgval = 0.0
for pnl in psys.pnls.values():
    drgval += pres.nfres.nffrc[pnl.ind, 0].x
    i += 1
    if i == 32:
        i = 0
        grdy = [grd.y for grd in pnl.grds]
        miny = min(grdy)
        maxy = max(grdy)
        rngy = maxy-miny
        drg.append(drgval/rngy) 
        yst.append(pnl.pnto.y)
        drgval = 0.0
    
for yi, di in zip(yst, drg):
    print(f'y = {yi:.6f}, drg = {di:.6f}')

#%% Plot
from matplotlib.pyplot import figure

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.plot(yst, drg)
ax.set_xlim(0.0, 1.875)
ax.set_ylabel('Induced Drag Distribution [N/m]')
_ = ax.grid(True)
