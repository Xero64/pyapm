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
speed = 12.9
rho = 1.145

pres = PanelResult('Design Point', psys)
pres.set_density(rho=rho)
pres.set_state(speed=speed)

display_markdown(pres)
display_markdown(pres.surface_loads)

mshfilepath = '..\\outputs\\' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%% Distribution Plots
axt = psys.plot_strip_twist_distribution()
_ = axt.set_ylabel('Strip Twist [deg]')
_ = axt.set_xlabel('Span-Wise Coordinate - y [m]')
axt = psys.plot_strip_chord_distribution()
_ = axt.set_ylabel('Strip Chord [m]')
_ = axt.set_xlabel('Span-Wise Coordinate - y [m]')
axw = psys.plot_strip_width_distribution()
_ = axw.set_ylabel('Strip Width [m]')
_ = axw.set_xlabel('Span-Wise Coordinate - y [m]')
axd = pres.plot_strip_drag_force_distribution()
_ = axd.set_ylabel('Drag Force [N/mm]')
_ = axd.set_xlabel('Span-Wise Coordinate - y [m]')
axs = pres.plot_strip_side_force_distribution()
_ = axs.set_ylabel('Side Force [N/mm]')
_ = axs.set_xlabel('Span-Wise Coordinate - y [m]')
axl = pres.plot_strip_lift_force_distribution()
_ = axl.set_ylabel('Lift Force [N/mm]')
_ = axl.set_xlabel('Span-Wise Coordinate - y [m]')
