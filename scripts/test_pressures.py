#%% Import Dependencies
from IPython.display import display_markdown
from pyapm.classes import panelsystem_from_json
from pyapm.outputs.msh import panelresult_to_msh
from pyapm.tools.points import fetch_pids, point_results
from pygeom.geom3d import Vector
from pygeom.matrix3d import zero_matrix_vector
from numpy.matlib import zeros

#%% Create Panel System
jsonfilepath = '../files/Test_Simple_Wing_2.json'
psys = panelsystem_from_json(jsonfilepath)
display_markdown(psys)

#%% Display Panel Result
pres = psys.results['Test Case']
display_markdown(pres)

#%% Determine Point Pressures
pnts = zero_matrix_vector((7, 2), dtype=float)
pnts[0, 0] = Vector(0.25, 0.05, 0.05)
pnts[1, 0] = Vector(0.25, 0.05, -0.05)
pnts[2, 0] = Vector(0.25, 4.5, 0.05)
pnts[3, 0] = Vector(0.25, -0.05, 0.05)
pnts[4, 0] = Vector(0.25, -0.05, -0.05)
pnts[5, 0] = Vector(0.25, -4.5, 0.05)
pnts[6, 0] = Vector(0.25, -6.5, 0.05)
pnts[0, 1] = Vector(0.75, 0.05, 0.05)
pnts[1, 1] = Vector(0.75, 0.05, -0.05)
pnts[2, 1] = Vector(0.75, 4.5, 0.05)
pnts[3, 1] = Vector(0.75, -6.5, 0.05)
pnts[4, 1] = Vector(0.75, -0.05, 0.05)
pnts[5, 1] = Vector(0.75, -0.05, -0.05)
pnts[6, 1] = Vector(0.75, -4.5, 0.05)

pids, chkz = fetch_pids(pnts, psys, ztol=0.1)
print(f'pids = \n{pids}\n')
print(f'chkz = \n{chkz}\n')

prs = point_results(pnts, psys, pids, chkz, pres.nfres.nfprs)
print(f'prs = \n{prs}\n')

#%% Output MSH File
mshfilepath = '../results/' + psys.name + '.msh'
panelresult_to_msh(pres, mshfilepath)
