import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat

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


#gets correlation matrix for all mice and averages them out
#return averaged corr. matrix as well as std. corr matrix
def get_corr_matrix(mice_data):
    matrices = []
    for i in range(mice_data.shape[0]):
        #need to exclude first 1200 baseline frames
        mouse_data = mice_data[i, :, 1200:]
        matrices.append(np.corrcoef(mouse_data))
    np_matrices = np.array(matrices)
    return np.mean(np_matrices, axis = 0), np.std(np_matrices, axis = 0)

#gets num over certain correlation value, and indices
#excludes if they are in the same brain region
def num_over_corr(corr_matrix, value):
    indices = []
    num = 0
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if i // 2 == j // 2:
                pass
            else:
                if corr_matrix[i, j] >= value:
                    num += 1
                    indices.append([i, j])
    return num, indices

#creates correlation matrix and plots via seaborn
def plot_correlation(corr_matrix, x_label = " ", y_label = " ", title = " "):
    plt.figure()
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

#generates plots based on file path and preferred labels
def generate_plot(file_path,  x_label = " ", y_label = " ", title = " "):
    data = load_data_np(file_path)
    matrix = get_corr_matrix(data)[0]
    plot_correlation(matrix, x_label = "Brain Region 1", y_label = "Brain Region 2", title = title)
    
#gets number over correlation value from file path
def get_num_over_val(file_path, val):
    data = load_data_np(file_path)
    matrix = get_corr_matrix(data)[0]
    return num_over_corr(matrix, val)

#generates all plots
def generate_plots(*paths):
    for path in paths:
        full_file_name = path.split('/')[-1]
        file_name = full_file_name.split('.')[0] 
        generate_plot(path, title = file_name)
        
# generate_plot('matlab_files/ROI_CBV_data/reg_cbv_nalket_f.mat', title = "CBV for nalket_m")
# print(get_num_over_val('matlab_files/ROI_CBV_data/reg_cbv_nalket_f.mat', 0.97)[1])
def generate_all_plots():
    fig, axes = plt.subplots(nrows=2, ncols=3)
    generate_plots("matlab_files/ROI_CBV_data/reg_cbv_nalket_f.mat", "matlab_files/ROI_CBV_data/reg_cbv_nalket_m.mat", 
                   "matlab_files/ROI_CBV_data/reg_cbv_nalsal_f.mat" , "matlab_files/ROI_CBV_data/reg_cbv_nalsal_m.mat",
                   "matlab_files/ROI_CBV_data/reg_cbv_salket_f.mat", "matlab_files/ROI_CBV_data/reg_cbv_salket_m.mat")
generate_all_plots()