#%% Import Dependencies
from IPython.display import display_markdown
from pygeom.matrix3d.matrixvector import zero_matrix_vector
from pyapm.classes.panelsystem import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh
from numpy import seterr, diag, fill_diagonal

_ = seterr(divide='ignore')

#%% Create Panel System
jsonfilepath = '../files/Test_Simple_Wing_2.json'
psys = panelsystem_from_json(jsonfilepath)

display_markdown(psys)

numfreeedgs = 0
numconnedgs = 0
for pnl in psys.pnls.values():
    # print(f'{pnl.pid}: {pnl.edgpnls}')
    for edgpnl in pnl.edgpnls:
        if len(edgpnl) == 0:
            numfreeedgs += 1
        else:
            numconnedgs += 1

numconnedgs = numconnedgs/2

print(f'numfreeedgs = {numfreeedgs}')
print(f'numconnedgs = {numconnedgs}')

#%% Solve Panel Result
pres = psys.results['Test Case']

#%% Set Trailing Doublet Direction
psys.mesh_trailing_doublet_panels(pres.vfs.to_unit())

#%% Assemble and Solve
psys.assemble_panels()
psys.assemble_trailing_panels()
psys.apm(0.0)
psys.bps(0.0)
# fill_diagonal(psys._apm[0.0], 0.5)
psys.solve_system()

#%% Print Outs
# print(f'sig = \n{pres.sig}')
# print(f'mu = \n{pres.mu}')

for pnl in psys.pnls.values():
    print(pnl.pid)
    print(pnl.pnto)
    print(pnl.dirz)
    print()

print(psys.apm(0.0).sum(axis=1))
print(psys.aps(0.0).sum(axis=1))

#%% Display Result
display_markdown(pres)
display_markdown(pres.surface_loads)

#%% Output MSH File
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

# #%% Loop Through Grids
# for pid in sorted(psys.pnls):
#     pnl = psys.pnls[pid]
#     qx, qy = pnl.diff_mu(pres.mu)
#     qfs = pnl.crd.vector_to_local(pres.vfs)
#     print(f'qx = {qx+qfs.x}, qy = {qy+qfs.y}')

# #%% Distribution Plots
# axd = pres.plot_strip_drag_force_distribution()
# _ = axd.set_ylabel('Drag Force (N/m)')
# _ = axd.set_xlabel('Span-Wise Coordinate - y (m)')
# _ = pres.plot_trefftz_drag_force_distribution(ax=axd)
# axs = pres.plot_strip_side_force_distribution()
# _ = axs.set_ylabel('Side Force (N/m)')
# _ = axs.set_xlabel('Span-Wise Coordinate - y (m)')
# _ = pres.plot_trefftz_side_force_distribution(ax=axs)
# axl = pres.plot_strip_lift_force_distribution()
# _ = axl.set_ylabel('Lift Force (N/m)')
# _ = axl.set_xlabel('Span-Wise Coordinate - y (m)')
# _ = pres.plot_trefftz_lift_force_distribution(ax=axl)
# axw = pres.plot_trefftz_wash_distribution()
# _ = axw.set_ylabel('Wash (m/s)')
# _ = axw.set_xlabel('Span-Wise Coordinate - y (m)')

# #%% Solve Panel Result
# rho = 1.225
# speed = 1.0
# alpha = 5.0
# pbo2V = 0.3

# pres = PanelResult(f'Test Case', psys)
# pres.set_density(rho=rho)
# pres.set_state(alpha=alpha, speed=speed, pbo2V=pbo2V)

# #%% Output MSH File
# mshfilepath = '../results/Test Simple Wing 2 Roll.msh'
# panelresult_to_msh(pres, mshfilepath)
