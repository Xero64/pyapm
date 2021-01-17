from .msh import panelresult_to_msh, panelresults_to_msh, panelsystem_to_msh
from .md import panelsystem_to_md

def outputs_from_json(sysdct: dict):
    outputs = {}
    for casedct in sysdct['cases']:
        name = casedct['name']
        outputs[name] = []
        if 'outputs' in casedct:
            outputs[name] = outputs[name] + casedct['outputs']
    return outputs
