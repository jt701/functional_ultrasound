# file to do expirements without losing past ideas
# put current expirement at top and run
# older expirements should be lower on page
import numpy as np
import pre_processing as pre
import helper as helper
import matplotlib.pyplot as plt
import seed_corr as seed

# test SVD and global signal regression


def test_SVD():
    data = helper.load_data_np(
        relative_path="matlab_files/Time_Series_Data/ROI_CBV_Data/reg_cbv_salket_m.mat")
    region_data = pre.bandpass_filter(data, 0.01, 0.1, 4)
    final_data = pre.global_signal_regression(region_data[:, :, :])
    # plt.plot(final_data[9, :])
    # plt.show()

    # ask gpt how to align brain atlas without compressing time series into single image
    corr = helper.get_corr_matrix(final_data[:, :, :], start=2700, end=2800)[0]
    helper.plot_correlation(corr)

# test_SVD()

# test ROI plots,


def test_ROI_plots():
    data = helper.load_data_np(
        relative_path="matlab_files/Time_Series_Data/ROI_CBV_Data/reg_cbv_salket_m.mat")
    region_data = pre.bandpass_filter(data, 0.01, 0.1, 4)
    corr_matrix = helper.get_corr_matrix(region_data, start=1300, end=1450)[0]
    helper.plot_correlation_ROI(corr_matrix)


test_ROI_plots()

# saving large matlab as np
# tranposed, needs to be tranposed again upon loading for some weird reason


def save_large_np():
    cbv_data = helper.load_large_data(
        'matlab_files/Time_Series_Data/pixel_data/nalket_m_pix.mat')
    print('half way there')
    print(cbv_data.shape)
    np.save('python_data/time_series_data/pixel_data', cbv_data)

# save_large_np()

# is the pixel file even being loaded correctly???
# def test_seed():
#     pixel = helper.load_from_np("python_data/time_series_data/pixel_data/nalket_m_pix.npy")
#     pixel = pixel[0, :, :, :]
#     pixel = pre.bandpass_filter(pixel, 0.01, 0.1, 4)
#     # pixel = pre.global_signal_regression_proj(pixel)
#     roi = helper.load_data_np("matlab_files/Time_Series_Data/ROI_CBV_Data/reg_cbv_nalket_m.mat")[0, :, :]
#     roi = pre.bandpass_filter(roi, 0.01, 0.1, 4)
#     # roi = pre.global_signal_regression_proj(roi)
#     corr = seed.mouse_pixel_corr(pixel, roi, 17, start = 1600, end = 2000)
#     helper.plot_correlation(corr, min = -1, max = 1)

# test_seed()