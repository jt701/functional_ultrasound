import os
import sys

#adding parent director to path so we can import from helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import helpers.helper as helper
import helpers.pre_processing as pre

#load Tommasso data using helper function, apply hamming window
#here you can choose file from Tommasso script, can do your own splicing if you'd like
#relative path to where the matlab file is stored should be used
data = helper.load_data_np('matlab_files/dianni_data/salket_m_full.mat') 
data = data[:, :, 1350:1950]
data = pre.hamming(data)

#get_corr_matrix returns tuple
#first element is 30 by 30 array representing average correlations among all mice, second is standard deviation
corr_avg, corr_std = helper.get_corr_matrix(data)

#puts all corr matrices in array, useful for data analysis but not necessary for plotting
#helper.get_all_corr_matrix(data)

#plots correlations with labels, optional title field
helper.plot_corr_labels(corr_avg, title = "Correlations")









