from scipy.signal import butter, filtfilt
import numpy as np

import sys
sys.path.append('../../helper.py')
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

#removes global signal
#modeled by global signal and constant noise (np.ones)
def global_signal_regression(cbv_data):
    voxels_time_series = cbv_data.reshape(-1, cbv_data.shape[-1]).T
    U, sigma, V = np.linalg.svd(voxels_time_series, full_matrices=False)
    global_signal = U[:, 0]
    X = np.vstack((global_signal, np.ones(cbv_data.shape[-1])))
    regression_coeffs, _= np.linalg.lstsq(X.T, voxels_time_series, rcond=None)[:2]
    regressed_time_series = voxels_time_series - np.matmul(X.T, regression_coeffs)
    return regressed_time_series.T.reshape(cbv_data.shape)

def pixel_to_ROI(pixel_data, segment_masks, data_idx):
    #to implement
    print(0)


# cbv_data = helper.load_large_data('matlab_files/Time_Series_Data/pixel_data/nalket_m_pix.mat')
# print('half way there')
# np.save('python_data/time_series_data/pixel_data', cbv_data)


