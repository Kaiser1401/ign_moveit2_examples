import data_utils
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np



def remove_handling_from_done(fn):
    fout = Path(fn).with_suffix('.reworked')
    data = data_utils.load_data(fn)

    iGoodAnyway = 0

    for e in data:
        assert isinstance(e, data_utils.DataEntry)
        if e.b_handling_error_likely:
            if e.b_outcome:
                iGoodAnyway +=1
            e.b_outcome = None
            e.b_prediction = None
            e.b_simulated = False

    data_utils.write_data(data, fout)

    print(f"{iGoodAnyway} entries seemed to be good anyway....")




if __name__ == "__main__":
    fn = '/home/klaus/code/pymi2_ws/sim_data/out/testdata_v2_1k.dill'
    remove_handling_from_done(fn)
