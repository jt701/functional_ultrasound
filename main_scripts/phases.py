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
import matplotlib.pyplot as plt

#load data from relative path, apply hamming window
data = helper.load_data_np('matlab_files/dianni_data/salket_m_full.mat')
data = pre.hamming(data)
hilbert_transform = hilbert(data, axis=-1)

#For degrees, 180/pi miltiplicative factor
instantaneous_phase = np.angle(hilbert_transform) 

#9 mice phase difference, uses label name or index (set label param to true), time is in minutes post ketamine
# sig.plot_subplots_phase(instantaneous_phase,'CG1l', 'Naccl', 10, 10, True) 


#two mice phase difference one region
# sig.phase_diff(instantaneous_phase, 3, 5, 9, -1200)

# phase one mouse, one region
# plt.plot(instantaneous_phase[3, 5, :]) 
# plt.show()

#focus on 5/9
#individual phases, phase difference, individual mice/groups, understand signals themselves