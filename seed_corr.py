import helper as helper
import matplotlib.pyplot as plt
import numpy as np

pixel_data = helper.load_data_np('matlab_files/pixel_CBV_data/nalket_m_pix.mat')
roi_data = helper.load_data_np('matlab_files/ROI_CBV_data/reg_cbv_nalket_m.mat')
    
#takes in ROI path and pixel data path
#returns list of correlations using ROI as seed and correlating with every pixel
def region_pixel_corr(pixel_path, roi_path, region):
    pixel_data = helper.load_data_np(pixel_path)
    roi_data = helper.load_data_np(roi_path)
    
    image_dim = pixel_data.shape[:2]
    corr_matrix = np.zeros(image_dim)
    num_mice = roi_data.shape[0]
    
    for mouse in range(num_mice):
        region_data = roi_data[mouse, region, 1200:]
        for i in range(image_dim[0]):
            for j in range(image_dim[1]):
                corr = np.corrcoef(pixel_data[i, j, 1200:], region_data)
                corr_matrix[i,j] += corr[0, 1]     
    corr_matrix /= num_mice
    return corr_matrix



corr = region_pixel_corr('matlab_files/pixel_CBV_data/nalket_m_pix.mat','matlab_files/ROI_CBV_data/reg_cbv_nalket_m.mat', 17)
helper.plot_correlation(corr, min = 0, max = 1)
    
            
            
            
    
    
    
    
    















