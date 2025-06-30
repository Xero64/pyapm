#%%
# Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import PanelSystem
from pyapm.outputs.msh import panelresult_to_msh

#%%
# Create Panel System
jsonfilepath = '../files/Test_Simple_Wing_2.json'
psys = PanelSystem.from_json(jsonfilepath)

#%%
# Solve Panel Result
# rho = 1.225
# speed = 50.0
# alpha = 5.0

# pres = PanelResult('Test Case', psys)
# pres.set_density(rho=rho)
# pres.set_state(alpha=alpha, speed=speed)

pres = psys.results['Test Case']

#%%
# Set Horseshoe Direction
psys.set_horseshoes(pres.vfs.to_unit())

#%%
# Assemble and Solve
psys.assemble_panels_phi()
psys.assemble_horseshoes_phi()
psys.solve_system()

#%%
# Output MSH File
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%%
# Display Result
display_markdown(pres)
display_markdown(pres.surface_loads)

#%%
# Print Outs
print(f'sig = \n{pres.sig}')
print(f'mu = \n{pres.mu}')

#%%
# Loop Through Grids
for pid in sorted(psys.pnls):
    pnl = psys.pnls[pid]
    q = pnl.diff_mu(pres.mu, pres.mug)
    qfs = pnl.crd.vector_to_local(pres.vfs)
    print(f'qx = {q.x+qfs.x}, qy = {q.y+qfs.y}')

#%%
# Distribution Plots
axd = pres.plot_strip_drag_force_distribution()
_ = axd.set_ylabel('Drag Force (N/m)')
_ = axd.set_xlabel('Span-Wise Coordinate - y (m)')
_ = pres.plot_trefftz_drag_force_distribution(ax=axd)
axs = pres.plot_strip_side_force_distribution()
_ = axs.set_ylabel('Side Force (N/m)')
_ = axs.set_xlabel('Span-Wise Coordinate - y (m)')
_ = pres.plot_trefftz_side_force_distribution(ax=axs)
axl = pres.plot_strip_lift_force_distribution()
_ = axl.set_ylabel('Lift Force (N/m)')
_ = axl.set_xlabel('Span-Wise Coordinate - y (m)')
_ = pres.plot_trefftz_lift_force_distribution(ax=axl)
axw = pres.plot_trefftz_wash_distribution()
_ = axw.set_ylabel('Wash (m/s)')
_ = axw.set_xlabel('Span-Wise Coordinate - y (m)')

# #%%
# Solve Panel Result
# rho = 1.225
# speed = 1.0
# alpha = 5.0
# pbo2v = 0.3

# pres = PanelResult(f'Test Case', psys)
# pres.set_density(rho=rho)
# pres.set_state(alpha=alpha, speed=speed, pbo2v=pbo2v)

# #%%
# Output MSH File
# mshfilepath = '../results/Test Simple Wing 2 Roll.msh'
# panelresult_to_msh(pres, mshfilepath)

#%%
# Display Controls
for srfc in psys.srfcs:
    for sht in srfc.shts:
        for control in sht.ctrls:
            ctrl = sht.ctrls[control]
            ctup = psys.ctrls[control]
            print(f'control = {control}')
            print(f'ctrl.uhvec = {ctrl.uhvec}')
            print(f'ctrl.posgain = {ctrl.posgain}')
            print(f'ctrl.neggain = {ctrl.neggain}')
            print(f'ctrl.pnls = {ctrl.pnls}')

#%%
# Display Result
display_markdown(pres)

pres2 = pres.to_result(pres.name + ' + flap 15 deg')
pres2.set_controls(flap=5.0)
display_markdown(pres2)

#%%
# Output MSH File
mshfilepath = '../results/' + psys.name + ' Flap.msh'
panelresult_to_msh(pres2, mshfilepath)
