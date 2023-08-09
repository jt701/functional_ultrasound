#functions to test GSR and placement with relation to bandpass filtering, saved averages/std to outputs/GSR
#p = pixel, r = roi, GSR= global signal regression, BP = bandpass filtering
#done on nalket m group, using -10 to 0 min, 0 - 10 min, 10 - 20 min
#BP is linear so roi is equivalent to pixel for it, masking technique only is useful for GSR pixel

import sys
import os

# appending parent directory to path
current_dir = os.path.dirname(os.path.abspath("/Users/josepht/functional-ultrasound-ketamine/data_analysis/expirements/GSR.py"))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import helper as helper
import pre_processing as pre
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#DATA FOR THIS EXPIREMENT
pix_path = "python_data/time_series_data/pixel_data/nalket_m_pix.npy"
roi_path = "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat"
masks_path = "matlab_files/Time_Series_Data/segment_masks/nalket_m_mask.mat"
areas_path = "matlab_files/Time_Series_Data/pix_area/pix_area_nalket_m.mat"

#HELPER FUNCTION FOR EXPIREMENT

def gsr():
    0


#makes destination directory in right folder, required name
def make_dest(name):
    dest = os.path.join(parent_dir, "outputs", "GSR", name)
    if not os.path.exists(dest):
        os.makedirs(dest)


#gets mean and std corr matrix and saves
#assumes data is num mice by num regions by num frames
def get_corr_save(data, directory, base_name):
    avg_matrix, std_matrix = helper.get_corr_matrix(data)
    avg_matrix = np.round(avg_matrix, 1) * 10
    std_matrix = np.round(std_matrix, 1) * 10
    mask = np.triu(np.ones_like(avg_matrix, dtype=bool))
    
    
    plt.figure()
    sns.heatmap(avg_matrix, cmap='coolwarm', annot=True, mask=mask)
    plt.xlabel("Brain Region 1")
    plt.ylabel("Brain Region 2")
    plt.title(name + " average correlation")

#calls helper for three time periods
def time_periods_save(data, directory):
    get_corr_save(data[])

#EXPIREMENT 

#control, BP on region of interest data
def no_filter():
    make_dest("no_filter")
    
    


    

    