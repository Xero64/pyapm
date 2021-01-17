from ..classes import PanelSystem

def panelsystem_to_md(psys: PanelSystem, mdfilepath: str, outputs: dict):
    with open(mdfilepath, 'wt') as mdfile:
        mdfile.write(psys.__str__())
        for case in psys.results:
            pres = psys.results[case]
            mdfile.write('\n')
            mdfile.write(str(pres))
            for output in outputs[case]:
                output = output.lower()
                if output == 'stability derivatives':
                    mdfile.write('\n')
                    mdfile.write(str(pres.stability_derivatives))
                elif output == 'stability derivatives body':
                    mdfile.write('\n')
                    mdfile.write(str(pres.stability_derivatives_body))
