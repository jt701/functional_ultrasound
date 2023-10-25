import sys
import os

# appending parent directory to path
current_dir = os.path.dirname(os.path.abspath("/Users/josepht/functional-ultrasound-ketamine/data_analysis/expirements/ROI_expirements.py"))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import helpers.helper as helper
import helpers.pre_processing as pre
import matplotlib.pyplot as plt
import helpers.stats as stats 

# REGION 13/27 show interesting connection
#write functions that take p value, finds correlations that are significant (look at multiple comparisons corrections)
def filter_only_plots():
    # data = helper.load_data_np(relative_path = "matlab_files/Time_Series_Data/ROI_CBV_Data/nalket_m_roi.mat")
    data = helper.load_data_np("matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")
    region_data = pre.bandpass_filter(data[0:9, :, 600:1400], 0.01, 0.1 , 4)
    #need to do gsr on each mouse individually
    # region_data = pre.global_signal_regression(region_data)
    # for mouse in range(region_data.shape[0]):
    #     region_data[mouse, :, :] = pre.global_signal_regression_proj(region_data[mouse, :, :])
    corr_matrix = helper.get_corr_matrix(region_data)[1]
    # # print(helper.num_over_corr(corr_matrix, 0.0))
    helper.plot_correlation_ROI(corr_matrix)
    
filter_only_plots()

# def plott():
#     data = helper.load_data_np(relative_path = "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")
#     plt.plot(data[4, 7, :])
#     plt.show()
# plott()