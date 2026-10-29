"""Optional notebook reader. This is ETL only; generation is entirely native C++."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def read_table(folder, name):
    folder=Path(folder)
    if (folder/f'{name}.csv').exists():
        return pd.read_csv(folder/f'{name}.csv',float_precision='round_trip')
    schema=json.loads((folder/f'{name}.schema.json').read_text())
    frame=pd.DataFrame(np.load(folder/f'{name}.npy'),columns=schema['columns'])
    frame['step']=frame['step'].astype('int64')
    for column,labels in schema['categories'].items():
        values=[{'True':True,'False':False}.get(v,v) for v in labels]
        frame[column]=frame[column].astype('int64').map(dict(enumerate(values)))
    return frame
