from json import load
from sys import argv

from .classes import panelsystem_from_geom, panelsystem_from_mesh
from .outputs import outputs_from_json, panelresults_to_msh, panelsystem_to_md


def main(jsonfilepath: str='', mdfilepath: str=''):

    if jsonfilepath == '':
        if len(argv) == 1:
            print('Specify a .json input file to run and create a .md output file.')
            return
        jsonfilepath = argv[1]

    with open(jsonfilepath, 'rt') as jsonfile:
        sysdct = load(jsonfile)

    sysdct['source'] = jsonfilepath

    filetype = None
    if 'type' in sysdct:
        filetype = sysdct['type']
    elif 'panels' in sysdct and 'grids' in sysdct:
        filetype = 'mesh'
    elif 'surfaces' in sysdct:
        filetype = 'geom'

    if filetype == 'geom':
        sys = panelsystem_from_geom(sysdct)
    elif filetype == 'mesh':
        sys = panelsystem_from_mesh(sysdct)
    else:
        raise ValueError('Incorrect file type.')

    if mdfilepath == '':
        if len(argv) == 3:
            mdfilepath = argv[2]
        else:
            mdfilepath = jsonfilepath.replace('.json', '.md')

    outputs = outputs_from_json(sysdct)

    panelsystem_to_md(sys, mdfilepath, outputs)

    panelresults_to_msh(sys, outputs)
