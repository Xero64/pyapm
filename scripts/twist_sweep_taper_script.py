#%%
# Load Dependencies
from IPython.display import display_markdown
from pyapm import PanelSystem
from pyapm.outputs.msh import panelresult_to_msh

#%%
# Create Lattice System
jsonfilepath = '../files/Test_twist_sweep_taper.json'
psys = PanelSystem.from_json(jsonfilepath)
display_markdown(psys)

#%%
# System Plots
axt1 = psys.plot_twist_distribution()
_ = axt1.set_ylabel('Twist (deg)')
_ = axt1.set_xlabel('Span-Wise Coordinate - b (m)')
axt2 = psys.plot_tilt_distribution()
_ = axt2.set_ylabel('Tilt (deg)')
_ = axt2.set_xlabel('Span-Wise Coordinate - b (m)')
axt = psys.plot_chord_distribution()
_ = axt.set_ylabel('Chord (m)')
_ = axt.set_xlabel('Span-Wise Coordinate - b (m)')

#%%
# Solve Panel Result
pres1 = psys.results['Test Alpha']

# psys.assemble_horseshoes_wash()
psys.assemble_panels_phi(mach=pres1.mach)
psys.assemble_horseshoes_phi(mach=pres1.mach)
psys.solve_system(mach=pres1.mach)

display_markdown(pres1)
display_markdown(pres1.surface_loads)

mshfilepath = '../results/'+ psys.name + ' - ' + pres1.name + '.msh'
panelresult_to_msh(pres1, mshfilepath)

#%%
# Solve Panel Result
pres2 = psys.results['Test Alpha Mach']

# psys.assemble_horseshoes_wash()
psys.assemble_panels_phi(mach=pres2.mach)
psys.assemble_horseshoes_phi(mach=pres2.mach)
psys.solve_system(mach=pres2.mach)

display_markdown(pres2)
display_markdown(pres2.surface_loads)

mshfilepath = '../results/'+ psys.name + ' - ' + pres2.name + '.msh'
panelresult_to_msh(pres2, mshfilepath)

#%%
# Distribution Plots
axd = pres1.plot_strip_drag_force_distribution()
axd = pres2.plot_strip_drag_force_distribution(ax=axd)
_ = axd.set_ylabel('Drag Force (N/m)')
_ = axd.set_xlabel('Span-Wise Coordinate - b (m)')
# _ = pres.plot_trefftz_drag_force_distribution(ax=axd)
axs = pres1.plot_strip_side_force_distribution()
axs = pres2.plot_strip_side_force_distribution(ax=axs)
_ = axs.set_ylabel('Side Force (N/m)')
_ = axs.set_xlabel('Span-Wise Coordinate - b (m)')
# _ = pres.plot_trefftz_side_force_distribution(ax=axs)
axl = pres1.plot_strip_lift_force_distribution()
axl = pres2.plot_strip_lift_force_distribution(ax=axl)
_ = axl.set_ylabel('Lift Force (N/m)')
_ = axl.set_xlabel('Span-Wise Coordinate - b (m)')
# _ = pres.plot_trefftz_lift_force_distribution(ax=axl)
# axc = pres.plot_trefftz_circulation_distribution()
# _ = axc.set_ylabel('Circulation (m^2/s)')
# _ = axc.set_xlabel('Span-Wise Coordinate - b (m)')
# axw = pres.plot_trefftz_down_wash_distribution()
# _ = axw.set_ylabel('Wash (m/s)')
# _ = axw.set_xlabel('Span-Wise Coordinate - b (m)')