import os
import sys

#adding parent director to path so we can import from helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import helpers.helper as helper
import helpers.pre_processing as pre
import helpers.signal_plots as sig
from scipy.signal import hilbert
import numpy as np
import math

#load data from relative path, apply hamming window
data = helper.load_data_np('matlab_files/dianni_data/salket_m_full.mat') 
data = pre.hamming(data)
hilbert_transform = hilbert(data, axis=-1)
instantaneous_phase = np.angle(hilbert_transform) * 180/math.pi
sig.plot_subplots_phase(instantaneous_phase, 4, 21, 0)

#(3,19) high, (4,21) low