import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py


#loads data as np array
#assumes data exists in folder of same directory 
#assumes desired matlab variable is the file name
def load_data_np(relative_path, var_name = None):
    raw_data = loadmat(relative_path)
    if var_name:
        return np.array(raw_data[var_name])
    full_file_name = relative_path.split('/')[-1]
    file_name = full_file_name.split('.')[0] 
    mat_data = raw_data[file_name]
    return np.array(mat_data)

#loads large matlab data using h5py
def load_large_data (relative_path, var_name = None):
    with h5py.File(relative_path, 'r') as raw_data:
        raw_data = h5py.File(relative_path, 'r')
        if var_name:
            return np.array(raw_data[var_name])
        full_file_name = relative_path.split('/')[-1]
        file_name = full_file_name.split('.')[0] 
        mat_data = raw_data[file_name]
    return np.array(mat_data)



#gets correlation matrix for all mice and averages them out 
#mouse_num paramter allows you to select specific mouse only
#return averaged corr. matrix as well as std. corr matrix
def get_corr_matrix(mice_data, mouse_num = 'all'):
    matrices = []
    if mouse_num != 'all':
        mouse_data = mice_data[mouse_num, :, 1200:]
        return np.corrcoef(mouse_data), 0
    for i in range(mice_data.shape[0]):
        #need to exclude first 1200 baseline frames
        mouse_data = mice_data[i, :, 1200:]
        matrices.append(np.corrcoef(mouse_data))
    np_matrices = np.array(matrices)
    return np.mean(np_matrices, axis = 0), np.std(np_matrices, axis = 0)

#gets correlation data from singular mouse
def get_corr_matrix_mouse(mouse_data):
    return np.corrcoef(mouse_data[:, 1200:])

#gets num over certain correlation value, and indices
#excludes if they are in the same brain region
def num_over_corr(corr_matrix, value):
    indices = []
    num = 0
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if i // 2 == j // 2: #left/right brain region
                pass
            else:
                if corr_matrix[i, j] >= value:
                    num += 1
                    indices.append([i, j])
    return num, indices

#creates correlation matrix and plots via seaborn
def plot_correlation(corr_matrix, x_label = " ", y_label = " ", title = " ", min = 'omit', max= 'omit'):
    plt.figure()
    if min != "omit" and max != "omit":
        sns.heatmap(corr_matrix, cmap='coolwarm', vmin = min, vmax= max)
    else:
        sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

#generates plots based on file path and preferred labels
#default is use all mice data, can specify mouse_num if you'd like to just look at one
def generate_plot(file_path,  x_label = " ", y_label = " ", title = " ", mouse_num = "all"):
    data = load_data_np(file_path)
    matrix = get_corr_matrix(data, mouse_num)[0]
    plot_correlation(matrix, x_label = "Brain Region 1", y_label = "Brain Region 2", title = title)

#generates plots based on file paths and preferred labels, takes difference between the two
#default is use all mice data, can specify mouse_num if you'd like to just look at one
def generate_diff_plot(file_path1, file_path2,  x_label = " ", y_label = " ", title = " ", mouse_num = "all"):
    data = load_data_np(file_path1) - load_data_np(file_path2)
    matrix = get_corr_matrix(data, mouse_num)[0]
    plot_correlation(matrix, x_label = "Brain Region 1", y_label = "Brain Region 2", title = title)

    
#gets number over correlation value from file path
#speicfy for all or a specifc numbered mouse
def get_num_over_val(file_path, val, mouse_num = "all"):
    data = load_data_np(file_path)
    matrix = get_corr_matrix(data, mouse_num)[0]
    return num_over_corr(matrix, val)

#generates all plots
def generate_plots(*paths):
    for path in paths:
        full_file_name = path.split('/')[-1]
        file_name = full_file_name.split('.')[0] 
        generate_plot(path, title = file_name)

def generate_all_plots():
    generate_plots("matlab_files/ROI_CBV_data/reg_cbv_nalket_f.mat", "matlab_files/ROI_CBV_data/reg_cbv_nalket_m.mat", 
                   "matlab_files/ROI_CBV_data/reg_cbv_nalsal_f.mat" , "matlab_files/ROI_CBV_data/reg_cbv_nalsal_m.mat",
                   "matlab_files/ROI_CBV_data/reg_cbv_salket_f.mat", "matlab_files/ROI_CBV_data/reg_cbv_salket_m.mat")

