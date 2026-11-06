"""NumPy torque-only inference, with the independent STOP preserved."""
import numpy as np

def policy(w, raw):
    """Torque-only runtime; independent STOP keeps its own normalization."""
    x = raw
    if 'export_input_mean' in w:
        x = (x-w['export_input_mean'])/w['export_input_scale']
    for layer in ('shared.0','shared.2'):
        x = np.maximum(x @ w[layer+'.weight'].T+w[layer+'.bias'], 0)
    torque = x @ w['arm.weight'].T+w['arm.bias']
    if 'export_output_scale' in w:
        torque = torque*w['export_output_scale']+w['export_output_mean']
    s = raw[:w['stopnet.0.weight'].shape[1]]
    if 'stopnet.export_input_mean' in w:
        s = (s-w['stopnet.export_input_mean'])/w['stopnet.export_input_scale']
    for layer in ('stopnet.0','stopnet.2'):
        s = np.maximum(s @ w[layer+'.weight'].T+w[layer+'.bias'], 0)
    s = s @ w['stopnet.4.weight'].T+w['stopnet.4.bias']
    return torque, float(s[0])*float(w['ttp_clip'])
