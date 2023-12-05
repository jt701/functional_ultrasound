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
#needs to be tranposed because of h5py workings
def load_large_data(relative_path, var_name = None):
    with h5py.File(relative_path, 'r') as raw_data:
        raw_data = h5py.File(relative_path, 'r')
        if var_name:
            return np.array(raw_data[var_name])
        full_file_name = relative_path.split('/')[-1]
        file_name = full_file_name.split('.')[0] 
        mat_data = raw_data[file_name]
    return np.array(mat_data)

#loads large files from np, need to tranposed if using h5py
def load_from_np(file_path):
    return np.load(file_path).T

#save large file from np given that they require h5py
def save_large_np(file_path, destination_folder):
    data = load_large_data(file_path)
    full_file_name = file_path.split('/')[-1]
    file_name = full_file_name.split('.')[0] 
    save_path = destination_folder + "/" + file_name
    np.save(save_path, data)
    
#gets indices for splicing, takes in center/length in minutes
#return left, right, shift
def get_splicing_info(center, length):
    right = center * 60 + length * 30 + 1200 
    left = center * 60 - length * 30 + 1200
    shift = left - 1200
    return left, right, shift

    
    
#gets correlation matrix for all mice and averages them out 
#mouse_num paramter allows you to select specific mouse only
#return averaged corr. matrix as well as std. corr matrix for frames from start to end
def get_corr_matrix(mice_data, mouse_num = -1):
    matrices = []
    if mouse_num != -1:
        return np.corrcoef(mice_data[mouse_num, :, :])
    for i in range(mice_data.shape[0]):
        mouse_data = mice_data[i, :, :]
        matrices.append(np.corrcoef(mouse_data))
    np_matrices = np.array(matrices)
    return np.mean(np_matrices, axis = 0), np.std(np_matrices, axis = 0, ddof=1) #ddof because we have sample 

#gets correlation data from singular mouse data
#start and end specify which frames
def get_corr_matrix_mouse(mouse_data, start = 0, end = 10000):
    end = min(end, mouse_data.shape[-1])
    return np.corrcoef(mouse_data[:, start: end])

#gets all corr matrix, samples is first dimension
def get_all_corr_matrix(mice_data):
    matrices = []
    for i in range(mice_data.shape[0]):
        mouse_data = mice_data[i, :, :]
        matrices.append(np.corrcoef(mouse_data))
    np_matrices = np.array(matrices)
    return np_matrices
    

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

#gets num over correlation for same brain region across hemispheres
#acts as control, expected to be high
def num_over_corr2(corr_matrix, value):
    indices = []
    num = 0
    for i in range(0, len(corr_matrix), 2):
        if corr_matrix[i][i+1] >= value:
            num += 1
            indices.append([i, i + 1])
    return num, indices

def plot_corr_labels(corr, title="Correlations"):
    plt.figure()
    sns.heatmap(corr, cmap='plasma', vmin=0.3, vmax=0.8)
    plt.xlabel("Region 1")
    plt.ylabel("Region 2")
    locs = [i + 0.5 for i in range(len(corr[0]))]
    labels = ['AI L','GIDI L','S1 L','M1 L','M2 L','Cg1 L','PrL L','IL L','CPu L','NAcC L','NAcSh L',
    'NAcSh R','NAcC R','CPu R','IL R','PrL R','Cg1 R','M2 R','M1 R','S1 R','GIDI R','AI R']
    plt.xticks(locs, labels, rotation=90)
    plt.yticks(locs, labels)
    plt.title(title)
    plt.show()

#creates correlation matrix and plots via seaborn
def plot_correlation(corr_matrix, x_label = " ", y_label = " ", title = " ", min = 'omit', max= 'omit'):
    plt.figure()
    if min != "omit" and max != "omit":
        sns.heatmap(corr_matrix, cmap='plasma', vmin = min, vmax= max)
    else:
        sns.heatmap(corr_matrix, cmap='plasma')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

#plots numbers along with colors for correlation matrices
#plots from 1-10 by multiplying by 10, makes more readable
def plot_correlation_ROI(corr_matrix, x_label = "Region 1", y_label = "Region 2"):
    corr_matrix = np.round(corr_matrix, 1) * 10
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, mask=mask)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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

