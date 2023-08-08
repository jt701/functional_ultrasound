from scipy.signal import butter, filtfilt
import numpy as np
import time

# import sys
# sys.path.append('../../helper.py')
import helper as helper 

#implement bandpass filter and global signal regression
#should there be trimming or additional filtering down before global signal regression???
#interesting to see if global regression would be more helpful with ROI data to ROI data because there isn't random pixel noise

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
def global_signal_regression_proj(cbv_data):
    voxels_time_series = cbv_data.reshape(-1, cbv_data.shape[-1])
    U, sigma, V = np.linalg.svd(voxels_time_series.T, full_matrices=False)
    global_signal = U[:, 0]
    global_signal = global_signal.reshape(global_signal.shape[0], 1)
    projection = np.matmul(voxels_time_series, global_signal)
    denoised_matrix = voxels_time_series - np.matmul(projection, global_signal.T)
    return denoised_matrix.reshape(cbv_data.shape)


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


# b = pixel_to_ROI_vectorized("python_data/time_series_data/pixel_data/nalket_m_pix.npy",
#              "matlab_files/Time_Series_Data/segment_masks/nalket_m_mask.mat",
#              "matlab_files/Time_Series_Data/pix_area/pix_area_nalket_m.mat", 5)


        
    





