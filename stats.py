import numpy as np
from scipy.stats import t
import math


# gets corr_value for which certain comparison is greater than with alpha = p_val
# assumes normal distribution, utilizes t test
def get_sig_val_matrix(p_val, num_samples, avg_matrix, std_matrix):
    df = num_samples - 1
    t_score = t.ppf(1 - p_val, df)
    std_multiplier = t_score / math.sqrt(num_samples)
    sig_matrix = avg_matrix - std_multiplier * std_matrix
    return sig_matrix

#gets sig val for single value
def get_sig_val(mean, std, num_samples, p_val):
    df = num_samples - 1
    t_score = t.ppf(1 - p_val, df)
    sig_val = mean - t_score * std / math.sqrt(num_samples)
    return sig_val

#1 and -1 will return infinity
def fisherz(matrix):
    return np.arctanh(matrix)

def inverse_fisherz(matrix):
    return np.tanh(matrix)

#assumes matrix is n samples with sample being first dimension
#correlation matrix is accounted for in 2nd/3rd dimension
def fisher_sig_val(matrix, p_val):
    transformed_matrices = fisherz(matrix)
    num_samples = transformed_matrices.shape[0]
   
    avg_matrix = np.mean(transformed_matrices)
    std_matrix = np.std(transformed_matrices, ddof=1)
    sig_matrix = get_sig_val_matrix(p_val, num_samples, avg_matrix, std_matrix)
    
    return inverse_fisherz(sig_matrix)
    
    


