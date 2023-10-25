from scipy.signal import butter, filtfilt
import numpy as np
import time
from sklearn.decomposition import FastICA
from nilearn import plotting
import matplotlib.pyplot as plt
import scipy

# import sys
# sys.path.append('../../helper.py')
import helpers.helper as helper

#takes in data and assumes time is along last dimension
#applies bandpass filter fromm [lowcut, highcut] Hz of order 
def bandpass_filter(cbv_data, lowcut, highcut, order):
    #nyquist frequency, has to do with sampling
    sampling_frequenecy = 1
    nyquist = 0.5 * sampling_frequenecy
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, cbv_data, axis=-1)
    return filtered_data

#applies hamming window on data and returns new data
def hamming(data):
    hamming_data = scipy.signal.windows.hamming(data.shape[-1])
    new_data = hamming_data * data
    return new_data
#removes global signal, should only use on a singular mouse
#modeled by global signal and constant noise (np.ones)
def global_signal_regression(cbv_data):
    voxels_time_series = cbv_data.reshape(-1, cbv_data.shape[-1]).T
    U, sigma, V = np.linalg.svd(voxels_time_series, full_matrices=False)
    global_signal = U[:, 0]
    X = np.vstack((global_signal, np.ones(cbv_data.shape[-1])))
    regression_coeffs, _= np.linalg.lstsq(X.T, voxels_time_series, rcond=None)[:2]
    regressed_time_series = voxels_time_series - np.matmul(X.T, regression_coeffs)
    return regressed_time_series.T.reshape(cbv_data.shape)

#GSR using projection instead of regression
#reassigns voxel_time_series to save memory
def global_signal_regression_proj(cbv_data):
    voxels_time_series = cbv_data.reshape(-1, cbv_data.shape[-1])
    U, sigma, V = np.linalg.svd(voxels_time_series.T, full_matrices=False)
    global_signal = U[:, 0]
    global_signal = global_signal.reshape(global_signal.shape[0], 1)
    projection = np.matmul(voxels_time_series, global_signal)
    denoised_time_series = voxels_time_series - np.matmul(projection, global_signal.T)
    return denoised_time_series.reshape(cbv_data.shape)


#takes pixel data for single mouse and groups it to ROI
#pixel_data np file, segment mask matlab file, pix area matlab file
def pixel_to_ROI(pixel_data, segment_masks, pix_areas, data_idx):
    pix_data = helper.load_from_np(pixel_data)[data_idx, :, :, :]
    mask = helper.load_data_np(segment_masks)[data_idx, :, :, :]
    pix_area = helper.load_data_np(pix_areas)[data_idx, :]
    
    num_regions = mask.shape[0]
    num_frames = pix_data.shape[-1]
    roi_data = np.zeros((num_regions, num_frames))

    #iterate over every region, using for loop instead of vectorization for clarity
    for i in range(pix_area.shape[0]):
        masked_data = mask[i, :, :, np.newaxis] * pix_data
        roi_data[i, :] = np.sum(masked_data, axis=(0, 1)) / pix_area[i]
    return roi_data

def pixel_to_ROI_vectorized(pixel_data, segment_masks, pix_areas, data_idx):
    pix_data = helper.load_from_np(pixel_data)[data_idx, :, :, :]
    mask = helper.load_data_np(segment_masks)[data_idx, :, :, :]
    pix_area = helper.load_data_np(pix_areas)[data_idx, :]
    
    masked_data = mask[:, :, :, np.newaxis] * pix_data
    sum_pixels = np.sum(masked_data, axis=(1, 2))
    roi_data = sum_pixels / pix_area[:, np.newaxis]

    return roi_data

#takes in pixel_data for single mouse, allows for preprocessing of pixel data beforehand
def pixel_to_ROI_pixdata(pixel_data, masks, pix_areas, data_idx):
    mask = helper.load_data_np(masks)[data_idx, :, :, :]
    pix_area = helper.load_data_np(pix_areas)[data_idx, :]
    masked_data = mask[:, :, :, np.newaxis] * pixel_data
    sum_pixels = np.sum(masked_data, axis=(1, 2))
    roi_data = sum_pixels / pix_area[:, np.newaxis]

    return roi_data

#doesn't load any data for you, good for one mouse
def manual_pixel_to_ROI(pixel_data, mask, pix_area):
    masked_data = mask[:, :, :, np.newaxis] * pixel_data
    sum_pixels = np.sum(masked_data, axis=(1, 2))
    roi_data = sum_pixels / pix_area[:, np.newaxis]

    return roi_data

#takes entire group and returns the ROI data 
#very slow due to huge data set and new axes
def group_to_ROI(pix_data, masks, pix_areas):
    
    masks_reshaped = masks[:, :, np.newaxis, :, :]  # Shape: 9 mice x 30 regions x 1 x 98 x 92
    masked_data = masks_reshaped * pix_data[:, np.newaxis, :, :, :]  
    sum_pixels = np.sum(masked_data, axis=(2, 3))  
    roi_data = sum_pixels / pix_areas[:, :, np.newaxis]
    
    return roi_data

#keep roi pixels from mask only, one mouse at a time
def keep_roi_pixels(pix_data, masks):
    sum_masks = np.sum(masks, axis=0)
    binary_mask = sum_masks > 0
    masked_data = pix_data * binary_mask[:, :, np.newaxis]
    return masked_data

def ica(data, num_components):
    num_samples = data.shape[-1]
    fus_data_2d = data.reshape((-1, num_samples))

    ica = FastICA(n_components=num_components)

    components = ica.fit_transform(fus_data_2d.T)

    # Reshape components back to original shape
    independent_components = independent_components.T.reshape((num_components, -1))

    # Visualize ICA components using Nilearn
    for i in range(num_components):
        plotting.plot_stat_map(independent_components[i], display_mode='z', cut_coords=3, title=f'Component {i + 1}')
        plt.show()

# data = helper.load_data_np("matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")[4]
# data = bandpass_filter(data, 0.01, 0.1, 4)
# ICA(data)


        
    





