#%% Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh
from pyvlm.tools import Bell

#%% Create Panel Mesh
jsonfilepath = '../files/Prandtl-D2.json'
psys = panelsystem_from_json(jsonfilepath)
display_markdown(psys)

#%% Create Bell Distribution
bell = Bell(3.75, psys.srfcs[0].prfy)
bell.set_ym(psys.srfcs[0].strpy)

#%% System Plots
axt1 = psys.plot_twist_distribution()
_ = axt1.set_ylabel('Twist [deg]')
_ = axt1.set_xlabel('Span-Wise Coordinate - b [m]')
axt2 = psys.plot_tilt_distribution()
_ = axt2.set_ylabel('Tilt [deg]')
_ = axt2.set_xlabel('Span-Wise Coordinate - b [m]')
axt = psys.plot_chord_distribution()
_ = axt.set_ylabel('Chord [m]')
_ = axt.set_xlabel('Span-Wise Coordinate - b [m]')
# axw = psys.plot_strip_width_distribution()
# _ = axw.set_ylabel('Strip Width [m]')
# _ = axw.set_xlabel('Span-Wise Coordinate - b [m]')

#%% Assembly and Solution
psys.assemble_horseshoes_wash()
# psys.assemble_panels_wash()
psys.assemble_panels()
psys.assemble_horseshoes()
psys.solve_system()

#%% Solve Panel Result
pres = psys.results['Design Point']
display_markdown(pres)
display_markdown(pres.surface_loads)

bell.set_density(pres.rho)
bell.set_speed(pres.speed)
bell.set_lift(pres.nfres.nffrctot.z)

mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)

#%% Distribution Plots
axd = pres.plot_strip_drag_force_distribution(axis='y')
_ = axd.set_ylabel('Drag Force [N/m]')
_ = axd.set_xlabel('Span-Wise Coordinate - b [m]')
_ = pres.plot_trefftz_drag_force_distribution(ax=axd, axis='y')
_ = bell.plot_drag_force_distribution(ax=axd)
axs = pres.plot_strip_side_force_distribution(axis='y')
_ = axs.set_ylabel('Side Force [N/m]')
_ = axs.set_xlabel('Span-Wise Coordinate - b [m]')
_ = pres.plot_trefftz_side_force_distribution(ax=axs, axis='y')
axl = pres.plot_strip_lift_force_distribution(axis='y')
_ = axl.set_ylabel('Lift Force [N/m]')
_ = axl.set_xlabel('Span-Wise Coordinate - b [m]')
_ = pres.plot_trefftz_lift_force_distribution(ax=axl, axis='y')
_ = bell.plot_lift_force_distribution(ax=axl)
# axc = pres.plot_trefftz_circulation_distribution()
# _ = axc.set_ylabel('Circulation [m^2/s]')
# _ = axc.set_xlabel('Span-Wise Coordinate - b [m]')
axw = pres.plot_trefftz_down_wash_distribution(axis='y')
_ = axw.set_ylabel('Wash [m/s]')
_ = axw.set_xlabel('Span-Wise Coordinate - b [m]')
_ = bell.plot_trefftz_wash_distribution(ax=axw)
