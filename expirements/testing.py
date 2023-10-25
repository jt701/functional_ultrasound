# file to do expirements without losing past ideas
# put current expirement at bottom and run

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import time
import helpers.pre_processing as pre
import helpers.helper as helper
import matplotlib.pyplot as plt
import helpers.seed_corr as seed
import helpers.stats as stats
from sklearn.decomposition import FastICA, PCA


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
        relative_path="matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")
    # region_data = pre.bandpass_filter(data, 0.01, 0.1, 4)
    corr_matrix = helper.get_corr_matrix(data, start=1800, end=2400)[1]
    helper.plot_correlation_ROI(corr_matrix)


# test_ROI_plots()

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


def test_seed():
    pixel = helper.load_from_np(
        "python_data/time_series_data/pixel_data/nalket_m_pix.npy")
    pixel = pixel[0, :, :, :]
    pixel = pre.bandpass_filter(pixel, 0.01, 0.1, 4)
    # pixel = pre.global_signal_regression_proj(pixel)
    roi = helper.load_data_np(
        "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")[0, :, :]
    roi = pre.bandpass_filter(roi, 0.01, 0.1, 4)
    # roi = pre.global_signal_regression_proj(roi)
    corr = seed.mouse_pixel_corr(pixel, roi, 17, start=1600, end=2000)
    helper.plot_correlation(corr, min=-1, max=1)

# test_seed()

# checkingmaskdimensions


def maskTest():
    a = helper.load_data_np(
        "matlab_files/Time_Series_Data/segment_masks/nalket_m_mask.mat")
    print(a.shape)
# maskTest()


def save_pixel():
    helper.save_large_np("matlab_files/Time_Series_Data/pixel_data/nalket_f_pix.mat",
                         "python_data/time_series_data/pixel_data")
    print("done")
    helper.save_large_np("matlab_files/Time_Series_Data/pixel_data/nalket_m_pix.mat",
                         "python_data/time_series_data/pixel_data")
    print("done")
    helper.save_large_np("matlab_files/Time_Series_Data/pixel_data/salket_f_pix.mat",
                         "python_data/time_series_data/pixel_data")
    print("done")
    helper.save_large_np("matlab_files/Time_Series_Data/pixel_data/salket_m_pix.mat",
                         "python_data/time_series_data/pixel_data")
    print("done")
    helper.save_large_np("matlab_files/Time_Series_Data/pixel_data/nalsal_f_pix.mat",
                         "python_data/time_series_data/pixel_data")
    print("done")
    helper.save_large_np("matlab_files/Time_Series_Data/pixel_data/nalsal_m_pix.mat",
                         "python_data/time_series_data/pixel_data")
    print("done")
# save_pixel()


def test_tscore():
    data = helper.load_data_np(
        "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")
    # data = helper.load_data_np("matlab_files/Time_Series_Data/ROI_CBV_Data/nalket_m_roi.mat")

    region_data = pre.bandpass_filter(data[:, :, 1200:1800], 0.01, 0.1, 4)
    corr_matrix = helper.get_corr_matrix(region_data)
    sig_matrix = stats.get_sig_val_matrix(
        0.005, 9, corr_matrix[0], corr_matrix[1])
    helper.plot_correlation_ROI(sig_matrix)
# test_tscore()


def test_fisher_tscore():
    data = helper.load_data_np(
        "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")
    region_data = pre.bandpass_filter(data[:, :, 1200:1800], 0.01, 0.1, 4)
    corr_matrix = helper.get_all_corr_matrix(region_data)
    sig_matrix = stats.fisher_sig_val(corr_matrix, 0.01)
    helper.plot_correlation_ROI(sig_matrix)

# test_fisher_tscore()


def test_load_pix():
    data = helper.load_data_np(
        "matlab_files/Time_Series_Data/pix_area/pix_area_nalket_m.mat")
    print(data[2, :].shape)
    # data = data[1, 60, 57, :]
    # cleaned_data = pre.bandpass_filter(data, 0.01, 0.1, 4)
    # plt.plot(cleaned_data)
    # plt.show()

# test_load_pix()


def test_ROI_segmentation():
    # roi = helper.load_data_np("matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")[5, :, :]
    a = pre.pixel_to_ROI_vectorized("python_data/time_series_data/pixel_data/nalket_m_pix.npy",
                                    "matlab_files/Time_Series_Data/segment_masks/nalket_m_mask.mat",
                                    "matlab_files/Time_Series_Data/pix_area/pix_area_nalket_m.mat", 5)
    # print(np.sum(roi - a))
    a = pre.bandpass_filter(a[:, 1200:1800], 0.01, 0.1, 4)
    a = pre.global_signal_regression_proj(a)
    corr = helper.get_corr_matrix_mouse(a)
    helper.plot_correlation_ROI(corr)

# test_ROI_segmentation()


def gsr_pixel_test():
    t = time.time()
    gsr_pixel = helper.load_from_np(
        "python_data/time_series_data/pixel_data/nalket_m_pix.npy")[3, :, :, :]
    # gsr_pixel = pre.bandpass_filter(gsr_pixel, 0.01, 0.1, 4)
    # gsr_pixel = pre.global_signal_regression_proj(pixel)
    # gsr_pixel = pre.bandpass_filter(gsr_pixel, 0.01, 0.1, 4)
    roi = pre.pixel_to_ROI_pixdata(gsr_pixel, "matlab_files/Time_Series_Data/segment_masks/nalket_m_mask.mat",
                                   "matlab_files/Time_Series_Data/pix_area/pix_area_nalket_m.mat", 3)
    roi = pre.bandpass_filter(roi, 0.01, 0.1, 4)
    corr = helper.get_corr_matrix_mouse(roi)
    print(time.time() - t)
    helper.plot_correlation_ROI(corr)

# gsr_pixel_test()


def test_group_roi():
    two_pixels = helper.load_from_np(
        "python_data/time_series_data/pixel_data/nalket_m_pix.npy")[:2, :, :, :]
    mask = helper.load_data_np(
        "matlab_files/Time_Series_Data/segment_masks/nalket_m_mask.mat")[:2, :, :, :]
    pix_areas = helper.load_data_np(
        "matlab_files/Time_Series_Data/pix_area/pix_area_nalket_m.mat")[:2, :]
    two_pixels = pre.group_to_ROI(two_pixels, mask, pix_areas)

    rois = helper.load_from_np(
        "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")[:2, :, :]

    print(rois.shape)
    print(two_pixels.shape)
    print(np.sum(two_pixels - rois))

# test_group_roi()

# looping through bandpass is much faster
# vectorization optimizes within function but not for huge datasets and between subjects
# my datset is 40 million withoyt vectorization, 360 million with


def band_time_test():
    pix_data = helper.load_from_np(
        "python_data/time_series_data/pixel_data/nalket_m_pix.npy")

    t1 = time.time()
    pre.global_signal_regression_proj(pix_data)
    t2 = time.time()
    print(t2-t1)
    for i in range(pix_data.shape[0]):
        pre.global_signal_regression_proj(pix_data[i, :, :, :])
    t3 = time.time()
    print(t3 - t2)

# band_time_test()

# succesful, manual works as well as keep


def test_keep_roi():
    pix_data = helper.load_from_np(
        "python_data/time_series_data/pixel_data/nalket_m_pix.npy")[4, :, :, :]
    pix_area = helper.load_data_np(
        "matlab_files/Time_Series_Data/pix_area/pix_area_nalket_m.mat")[4, :]
    roi_data = helper.load_data_np(
        "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")[4, :, :]
    mask = helper.load_data_np(
        "matlab_files/Time_Series_Data/segment_masks/nalket_m_mask.mat")[4, :, :, :]
    keep = pre.keep_roi_pixels(pix_data, mask)
    new_roi = pre.manual_pixel_to_ROI(keep, mask, pix_area)
    print(np.sum(new_roi - roi_data))

# test_keep_roi()


def plot_region():
    data = helper.load_data_np(
        "matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")
    # compare 4 and 7 to bandpass

    # data[7] = pre.bandpass_filter(data[7], 0.01, 0.1, 4)
    plt.plot(data[2, 12, :])
    plt.show()


plot_region()

def test_dianni():
    data = helper.load_data_np('matlab_files/dianni_data/nalket_dianni.mat')
    hamming = helper.load_data_np('matlab_files/dianni_data/hamm_win.mat')
    windowed_data = hamming.T * data
    corr = helper.get_corr_matrix(windowed_data)[1]
    helper.plot_correlation_ROI(corr)


# test_dianni()

def dianni_center():
    data = helper.load_data_np('matlab_files/dianni_data/nalket_dianni.mat')
    hamming = helper.load_data_np('matlab_files/dianni_data/hamm_win.mat')
    means = np.mean(data, axis=2, keepdims=True)
    data = (data - means)/means
    windowed_data = hamming.T * data
    corr = helper.get_corr_matrix(windowed_data)[1]
    helper.plot_correlation_ROI(corr)


# dianni_center()

def add_hamming():
    data = helper.load_data_np(
        'matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat')
    hamming = helper.load_data_np('matlab_files/dianni_data/hamm_win.mat')
    data = data[:, :, 150:1051]
    data = pre.bandpass_filter(data, 0.01, 0.1, 4)
    data = data * hamming.T
    corr = helper.get_corr_matrix(data)[1]
    helper.plot_correlation_ROI(corr)


# add_hamming()
def gsr_dianni():
    data = helper.load_data_np('matlab_files/dianni_data/nalket_dianni.mat')
    for i in range(data.shape[0]):
        data[i] = pre.global_signal_regression_proj(data[i])
    hamming = helper.load_data_np('matlab_files/dianni_data/hamm_win.mat')
    windowed_data = hamming.T * data
    corr = helper.get_corr_matrix(windowed_data)[1]
    helper.plot_correlation_ROI(corr)
# gsr_dianni()

def ica_test():
    data = helper.load_data_np('matlab_files/dianni_data/nalket_dianni.mat')
    data = data[4, :, :].T
    data_centered = data - data.mean(axis=0)
    pca = PCA(whiten=True)
    data_whitened = pca.fit_transform(data_centered)
    num_components = 5
    ica = FastICA(n_components=num_components)
    ica.fit(data_whitened)
    components = ica.components_
    np.save('python_data/', components)
# ica_test()

# def ica_load():
#     ica = np.load('python_data/ica.npy')
#     print(ica[0])
# ica_load()

def signal():
    data = helper.load_data_np(
        'matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat')
    plt.figure()


