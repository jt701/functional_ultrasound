import os
import sys

#adding parent director to path so we can import from helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import helpers.helper as helper
import helpers.pre_processing as pre
import helpers.signal_plots as sig

#load data from relative path, apply hamming window
data = helper.load_data_np('matlab_files/dianni_data/salket_m_full.mat') 
data = pre.hamming(data)

#plots subplots for every mouse for region 8 and 13, utilizes 150 second shift (this time frame starts at +150)
#labels shows which region is which element
labels = ['AI L','GIDI L','S1 L','M1 L','M2 L','Cg1 L','PrL L','IL L','CPu L','NAcC L','NAcSh L',
    'NAcSh R','NAcC R','CPu R','IL R','PrL R','Cg1 R','M2 R','M1 R','S1 R','GIDI R','AI R']
sig.plot_subplots(data, 'cg1l', 'naccl', 0, 5, labels=True, hilb=False)



