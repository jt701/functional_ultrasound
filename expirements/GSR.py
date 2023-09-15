#functions to test GSR and placement with relation to bandpass filtering, saved averages/std to outputs/GSR
#p = pixel, r = roi, GSR= global signal regression, BP = bandpass filtering
#done on nalket m group, using -10 to 0 min, 0 - 10 min, 10 - 20 min
#data has been baseline adjusted, and spatiotemporally filtered beforehand
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
import time

#DATA FOR THIS EXPIREMENT
pix_path = "python_data/time_series_data/pixel_data/nalket_m_pix.npy"
roi_path = "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat"
masks_path = "matlab_files/Time_Series_Data/segment_masks/nalket_m_mask.mat"
areas_path = "matlab_files/Time_Series_Data/pix_area/pix_area_nalket_m.mat"
expirement_name = "GSR_nalket_m"

#HELPER FUNCTION FOR EXPIREMENT
#   not vectorizing all functions due to high memory usage, tradeoff in memory/processing speed
def load_pix():
    return helper.load_from_np(pix_path)

def load_roi():
    return helper.load_data_np(roi_path)

#directly modifies argument to conserve memory
def gsr(data):
    for i in range(data.shape[0]):
        data[i] = pre.global_signal_regression_proj(data[i])
        
def bandpass(data):
    for i in range(data.shape[0]):
        data[i] = pre.bandpass_filter(data[i], 0.01, 0.1, 4)

def mask_filter(pix_data):
    masks = helper.load_data_np(masks_path)
    for i in range(pix_data.shape[0]):
        pix_data[i] = pre.keep_roi_pixels(pix_data[i], masks[i])

def pix_to_roi(pix_data):
    masks = helper.load_data_np(masks_path)
    areas = helper.load_data_np(areas_path)
    
    num_mice = pix_data.shape[0]
    num_regions = masks.shape[1]
    num_frames = pix_data.shape[-1]
    
    roi_data = np.zeros((num_mice, num_regions, num_frames))
    
    for i in range(num_mice):
        roi_data[i] = pre.manual_pixel_to_ROI(pix_data[i], masks[i], areas[i])
    return roi_data

#makes destination directory in right folder, required name
#returns directory as well for ease of use
def make_dest(name):
    dest = os.path.join(parent_dir, "outputs", expirement_name, name)
    if not os.path.exists(dest):
        os.makedirs(dest)
    return dest


#gets mean and std corr matrix and saves
#assumes data is num mice by num regions by num frames
def get_corr_save(data, directory, base_name, title):
    avg_matrix, std_matrix = helper.get_corr_matrix(data)
    avg_matrix = np.round(avg_matrix, 1) * 10
    std_matrix = np.round(std_matrix, 1) * 10
    mask = np.triu(np.ones_like(avg_matrix, dtype=bool))
    
    #savign avg. figure
    plt.figure()
    sns.heatmap(avg_matrix, cmap='coolwarm', annot=True, mask=mask)
    plt.xlabel("Brain Region 1")
    plt.ylabel("Brain Region 2")
    plt.title(title + " avg")
    file_name = base_name + "_avg.png"
    path = os.path.join(directory, file_name)
    plt.savefig(path)
    plt.close()
    
    #saving std. figure
    plt.figure()
    sns.heatmap(std_matrix, cmap='coolwarm', annot=True, mask=mask)
    plt.xlabel("Brain Region 1")
    plt.ylabel("Brain Region 2")
    plt.title(title + " std")
    file_name = base_name + "_std.png"
    path = os.path.join(directory, file_name)
    plt.savefig(path)
    plt.close()

#calls helper for three time periods, assumes 9 by 30 by 4201 time series data
def time_periods_save(data, directory, curr_expirement):
    get_corr_save(data[:, :, 600:1200], directory, curr_expirement + "_-10to0min", curr_expirement + " -10 to 0 min")
    get_corr_save(data[:, :, 1200:1800], directory,curr_expirement + "_0to10min", curr_expirement + " 0 to 10 min")
    get_corr_save(data[:, :, 1800:2400], directory, curr_expirement + "_10to20min", curr_expirement + " 10 to 20 min")


#EXPIREMENT 

#control, BP on region of interest data
def no_filter():
    dest = make_dest("no_filter")
    roi_data = load_roi()
    time_periods_save(roi_data, dest, "no_filter")
# no_filter()

def bp_roi():
    dest = make_dest("bp_roi")
    roi_data = load_roi()
    bandpass(roi_data)
    time_periods_save(roi_data, dest, "bp_roi")

def gsr_roi():
    dest = make_dest("gsr_roi")
    roi_data = load_roi()
    gsr(roi_data)
    time_periods_save(roi_data, dest, "gsr_roi")
# gsr_roi()

def gsr_pix():
    dest = make_dest("gsr_pix")
    pix_data = load_pix()
    gsr(pix_data)
    roi_data = pix_to_roi(pix_data)
    time_periods_save(roi_data, dest, "gsr_pix")

# gsr_pix()
def gsrpix_bproi():
    dest = make_dest("gsrpix_bproi")
    pix_data = load_pix()
    gsr(pix_data)
    roi_data = pix_to_roi(pix_data)
    bandpass(roi_data)
    time_periods_save(roi_data, dest, "gsrpix_bproi")

# t = time.time()  
# gsrpix_bproi()
# print(time.time() - t)  

def gsrroi_bproi():
    dest = make_dest("gsrroi_bproi")
    data = load_roi()
    gsr(data)
    bandpass(data)
    time_periods_save(data, dest, "gsrroi_bproi")
# gsrroi_bproi()
    
def bppix_gsrpix():
    dest = make_dest("bppix_gsrpix")
    pix_data = load_pix()
    bandpass(pix_data)
    gsr(pix_data)
    roi_data = pix_to_roi(pix_data)
    time_periods_save(roi_data, dest, "bppix_gsrpix")
# bppix_gsrpix()

def bproi_gsroi():
    dest = make_dest("bproi_gsroi")
    roi_data = load_roi()
    bandpass(roi_data)
    gsr(roi_data)
    time_periods_save(roi_data, dest, "bproi_gsroi")

# bproi_gsroi()
def mask_gsrpix_bproi():
    dest = make_dest("mask_gsrpix_bproi")
    pix_data = load_pix()
    mask_filter(pix_data)
    gsr(pix_data)
    roi_data = pix_to_roi(pix_data)
    bandpass(roi_data)
    time_periods_save(roi_data, dest, "mask_gsrpix_bproi")
mask_gsrpix_bproi()

def mask_bppix_gsrpix():
    dest = make_dest("mask_bppix_gsrpix")
    pix_data = load_pix()
    mask_filter(pix_data)
    bandpass(pix_data)
    gsr(pix_data)
    roi_data = pix_to_roi(pix_data)
    time_periods_save(roi_data, dest, "mask_bppix_gsrpix")
mask_bppix_gsrpix()
    

    

    