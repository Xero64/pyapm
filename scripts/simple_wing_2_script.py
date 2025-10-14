#%%
# Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import PanelSystem
from pyapm.outputs.k3d import PanelPlot
from pyapm.outputs.msh import panelresult_to_msh

#%%
# Create Panel System
jsonfilepath = '../files/Test_Simple_Wing_2.json'
psys = PanelSystem.from_json(jsonfilepath)
display_markdown(psys)

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
# Assemble and Solve
psys.assemble_panels_phi()
# psys.assemble_horseshoes_phi()
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
print(f'mud = \n{pres.mud}')
print(f'muw = \n{pres.muw}')
print(f'mug = \n{pres.mug}')

#%%
# Distribution Plots
axd = pres.plot_strip_drag_force_distribution()
_ = axd.set_ylabel('Drag Force (N/m)')
_ = axd.set_xlabel('Span-Wise Coordinate - y (m)')
# _ = pres.plot_trefftz_drag_force_distribution(ax=axd)
axs = pres.plot_strip_side_force_distribution()
_ = axs.set_ylabel('Side Force (N/m)')
_ = axs.set_xlabel('Span-Wise Coordinate - y (m)')
# _ = pres.plot_trefftz_side_force_distribution(ax=axs)
axl = pres.plot_strip_lift_force_distribution()
_ = axl.set_ylabel('Lift Force (N/m)')
_ = axl.set_xlabel('Span-Wise Coordinate - y (m)')
# _ = pres.plot_trefftz_lift_force_distribution(ax=axl)
# axw = pres.plot_trefftz_wash_distribution()
# _ = axw.set_ylabel('Wash (m/s)')
# _ = axw.set_xlabel('Span-Wise Coordinate - y (m)')

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
# Display Result
pnlpl = PanelPlot(psys, pres)

mshplot = pnlpl.plot()
mshplot += pnlpl.panel_mesh()
mshplot.display()

sigplot = pnlpl.plot()
sigplot += pnlpl.panel_sigma_plot()
sigplot.display()

siggplot = pnlpl.plot()
siggplot += pnlpl.grid_sigma_plot()
siggplot.display()

mudplot = pnlpl.plot()
mudplot += pnlpl.panel_mud_plot()
mudplot.display()

mugplot = pnlpl.plot()
mugplot += pnlpl.grid_mu_plot()
mugplot.display()
