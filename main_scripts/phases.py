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

#preprocesing, needs to be done after splicing
# data = pre.hamming(data)
# hilbert_transform = hilbert(data, axis=-1)
# instantaneous_phase = np.angle(hilbert_transform) 

#9 mice phase difference, uses label name or index (set label param to true), time is in minutes post ketamine
# sig.plot_subplots_phase(data,'CG1l', 'Naccl', 10, 10, True) 


#one mouse two region phase difference
# sig.phase_diff(data, 3, 'Cg1l','naccl',10, 10, True)


#phase, all mice two regions
sig.plot_subplots(data, 'cg1l', 'naccl', 0, 5, True, ylabel="Phase Angles", hilb=True)

#phase, one mouse two regions
# sig.plot_phases_both(data, 6, 'Cg1l','naccl',0, 10, True)
