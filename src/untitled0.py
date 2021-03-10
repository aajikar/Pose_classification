# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:53:42 2021

@author: BTLab
"""

import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
import matplotlib
import multiprocessing

my_path = Path(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\Patient 0 No Thresh\Data')

flist = [p for p in my_path.iterdir() if p.is_file()]


def plot_and_save_numpy(file):
    new_dir = Path(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\Patient 0 No Thresh\Plots')
    fn  = new_dir / str(file.stem + '.png')
    array = np.load(file)
    matplotlib.image.imsave(fn, array)
    return None

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(plot_and_save_numpy)(file) for file in flist)