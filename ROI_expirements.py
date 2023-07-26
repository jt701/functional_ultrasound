#design
import helper as helper
import pre_processing as pre

def filter_only_plots():
    data = helper.load_data_np(relative_path = "matlab_files/Time_Series_Data/ROI_CBV_Data/reg_cbv_salket_m.mat")
    region_data = pre.bandpass_filter(data[:, :, 3600:4200], 0.01, 0.1 , 4)
    #need to do gsr on each mouse individually
    # region_data = pre.global_signal_regression(region_data)
    for mouse in range(region_data.shape[0]):
        region_data[mouse, :, :] = pre.global_signal_regression_proj(region_data[mouse, :, :])
    corr_matrix = (helper.get_corr_matrix(region_data)[0])
    # print(helper.num_over_corr(corr_matrix, 0.0))
    helper.plot_correlation_ROI(corr_matrix)
    
filter_only_plots()