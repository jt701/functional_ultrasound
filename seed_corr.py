import helper as helper
import matplotlib.pyplot as plt
import numpy as np
import pre_processing as pre


    
#takes in ROI path and pixel data path
#returns list of correlations using ROI as seed and correlating with every pixel
def region_pixel_corr(pixel_data, roi_data, region, start = 0, end = 10000):
    # pixel_data = helper.load_data_np(pixel_path)
    # roi_data = helper.load_data_np(roi_path)
    
    image_dim = pixel_data.shape[:2]
    corr_matrix = np.zeros(image_dim)
    num_mice = roi_data.shape[0]
    
    for mouse in range(num_mice):
        region_data = roi_data[mouse, region, start:end]
        for i in range(image_dim[0]):
            for j in range(image_dim[1]):
                corr = np.corrcoef(pixel_data[i, j, start:end], region_data)
                corr_matrix[i,j] += corr[0, 1]     
    corr_matrix /= num_mice
    return corr_matrix

#assumes pixel data is for one mouse
#assumes roi data is for one mouse
def mouse_pixel_corr(pixel_data, roi_data, region, start = 0, end = 10000):
    # pixel_data = helper.load_data_np(pixel_path)
    # roi_data = helper.load_data_np(roi_path)
    
    image_dim = pixel_data.shape[:2]
    corr_matrix = np.zeros(image_dim)
    for i in range(image_dim[0]):
        for j in range(image_dim[1]):
            corr = np.corrcoef(pixel_data[i, j, start:end], roi_data[region, start:end])
            corr_matrix[i,j] += corr[0, 1]     
    return corr_matrix


