#file to do expirements without losing past ideas
#put current expirement at top and run
#older expirements should be lower on page

import pre_processing as pre 
import helper as helper
import matplotlib.pyplot as plt

#test SVD and global signal reduction
def test_SVD():
    data = helper.load_data_np(relative_path = "matlab_files/Time_Series_Data/ROI_CBV_Data/reg_cbv_nalket_m.mat")
    region_data = pre.bandpass_filter(data, 0.01, 0.1 , 4)
    final_data = pre.global_signal_regression(region_data[4, :, :])
    # plt.plot(final_data[9, :])
    # plt.show()
    #WHY IS SLICE ERRORING, CHECK
    #ask gpt how to align brain atlas without compressing time series into single image
    corr = helper.get_corr_matrix_mouse(final_data)
    helper.plot_correlation(corr)

test_SVD()

