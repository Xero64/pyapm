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

#%% Plot Panel Drag
from matplotlib.pyplot import figure

ypos = []
drag = []
yfrc = []
lift = []
srfc = psys.srfcs[0]
for strp in srfc.strps:
    pnl = strp.pnls[0]
    grdy = [grd.y for grd in pnl.grds]
    miny = min(grdy)
    maxy = max(grdy)
    rngy = maxy-miny
    dragval = 0.0
    yfrcval = 0.0
    liftval = 0.0
    for pnl in strp.pnls:
        dragval += pres.nfres.nffrc[pnl.ind, 0].x
        yfrcval += pres.nfres.nffrc[pnl.ind, 0].y
        liftval += pres.nfres.nffrc[pnl.ind, 0].z
    drag.append(dragval/rngy)
    yfrc.append(yfrcval/rngy)
    lift.append(liftval/rngy)
    ypos.append(pnl.pnto.y)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.plot(ypos, drag)
ax.set_xlim(0.0, 1.875)
ax.set_ylabel('Induced Drag Distribution [N/m]')
_ = ax.grid(True)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.plot(ypos, yfrc)
ax.set_xlim(0.0, 1.875)
ax.set_ylabel('Y Force Distribution [N/m]')
_ = ax.grid(True)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.plot(ypos, lift)
ax.set_xlim(0.0, 1.875)
ax.set_ylabel('Lift Distribution [N/m]')
_ = ax.grid(True)

#%% Output to MSH File
mshfilepath = '..\\outputs\\' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)
