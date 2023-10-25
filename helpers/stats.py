import numpy as np
from scipy.stats import t
import math
import helpers.helper as helper
import helpers.pre_processing as pre


# gets corr_value for which certain comparison is greater than with alpha = p_val
# assumes normal distribution, utilizes t test
def get_sig_val_matrix(p_val, num_samples, avg_matrix, std_matrix):
    df = num_samples - 1
    t_score = t.ppf(1 - p_val, df)
    std_multiplier = t_score / math.sqrt(num_samples)
    sig_matrix = avg_matrix - std_multiplier * std_matrix
    return sig_matrix

# gets sig val for single value


def get_sig_val(mean, std, num_samples, p_val):
    df = num_samples - 1
    t_score = t.ppf(1 - p_val, df)
    sig_val = mean - t_score * std / math.sqrt(num_samples)
    return sig_val


# returns fisher z converted matrix
# users error correction for -1 and 1 (converts them to about)+- 0.9999
def fisherz(matrix):
    transformed = np.arctanh(matrix)
    transformed = np.where(np.isposinf(transformed), 5, transformed)
    transformed = np.where(np.isneginf(transformed), -5, transformed)
    return transformed


def inverse_fisherz(matrix):
    return np.tanh(matrix)

# assumes matrix is n samples with sample being first dimension
# correlation matrix is accounted for in 2nd/3rd dimension


def fisher_sig_val(matrix, p_val):
    transformed_matrices = fisherz(matrix)
    num_samples = transformed_matrices.shape[0]
    avg_matrix = np.mean(transformed_matrices, axis=0)
    std_matrix = np.std(transformed_matrices, axis=0, ddof=1)
    sig_matrix = get_sig_val_matrix(p_val, num_samples, avg_matrix, std_matrix)

    return inverse_fisherz(sig_matrix)


# data = helper.load_data_np("matlab_files/Time_Series_Data/ROI_CBV_Data/rel_cbv_nalket_m.mat")
# region_data = pre.bandpass_filter(data[:, :, 1200:1800], 0.01, 0.1, 4)
# corr_matrix = helper.get_all_corr_matrix(region_data)
# sig_matrix = fisher_sig_val(corr_matrix, 0.01)
# helper.plot_correlation_ROI(sig_matrix)


