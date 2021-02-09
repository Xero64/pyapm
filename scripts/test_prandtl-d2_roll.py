#%% Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh

#%% Create Panel Mesh
jsonfilepath = '../files/Prandtl-D2.json'
psys = panelsystem_from_json(jsonfilepath)
psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%% Solve Panel Result
pres1 = psys.results['Design Point']
display_markdown(pres1)
display_markdown(pres1.surface_loads)
display_markdown(pres1.stability_derivatives)

mshfilepath = '../results/' + pres1.name + '.msh'
panelresult_to_msh(pres1, mshfilepath)

#%% Solve Panel Result
pres2 = psys.results['Roll Case']
display_markdown(pres2)
display_markdown(pres2.surface_loads)
display_markdown(pres2.stability_derivatives)

mshfilepath = '../results/' + pres2.name + '.msh'
panelresult_to_msh(pres2, mshfilepath)

#%% Distribution Plots
axd = pres1.plot_strip_drag_force_distribution(axis='y')
_ = axd.set_ylabel('Drag Force (N/m)')
_ = axd.set_xlabel('Span-Wise Coordinate - b (m)')
_ = pres2.plot_strip_drag_force_distribution(ax=axd, axis='y')
axs = pres1.plot_strip_side_force_distribution(axis='y')
_ = axs.set_ylabel('Side Force (N/m)')
_ = axs.set_xlabel('Span-Wise Coordinate - b (m)')
_ = pres2.plot_strip_side_force_distribution(ax=axs, axis='y')
axl = pres1.plot_strip_lift_force_distribution(axis='y')
_ = axl.set_ylabel('Lift Force (N/m)')
_ = axl.set_xlabel('Span-Wise Coordinate - b (m)')
_ = pres2.plot_strip_lift_force_distribution(ax=axl, axis='y')
