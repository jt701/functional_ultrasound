import helper as helper
import matplotlib.pyplot as plt
import numpy as np
import scipy

def plot_regions(data, mouse, region1, region2, shift=0, xlab="Region 1", ylab="Region 2", title = "Region 1 vs Region 2"):
    x_axis = np.arange(data.shape[-1]) + shift
    plt.plot(x_axis, data[mouse, region1, :], label=xlab, color='blue')
    plt.plot(x_axis, data[mouse, region2, :], label=ylab, color='red')

    plt.xlabel('Time(sec) post ketamine')
    plt.ylabel('Signal')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_corr(data):
    corr = helper.get_corr_matrix(data)[0]
    helper.plot_correlation_ROI(corr)
    
def plot_subplot(ax, data, mouse, region1, region2, shift=100, xlab="Region 1", ylab="Region 2", title = "Region 1 vs Region 2"):
    x_axis = np.arange(data.shape[-1]) + shift
    
    ax.plot(x_axis, data[mouse, region1, :], label=xlab, color='blue')
    ax.plot(x_axis, data[mouse, region2, :], label=ylab, color='red')
    
    ax.set_xlabel('Time(sec) post ketamine')
    ax.set_ylabel('Signal')
    ax.set_title(title, fontsize=8)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # ax.legend()
    
    return ax
def plot_subplots(data, region1, region2, shift, title=""):
    num_mice = data.shape[0]
    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            if 3*i + j > num_mice - 1:
                continue
            mouse_num = 3*i + j
            title = "Mouse" + str(mouse_num)
            plot_subplot(axes[i,j], data, mouse_num, region1, region2, shift, title=title, xlab="RE1", ylab="RE2")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    
# data = helper.load_data_np('matlab_files/dianni_data/nalket_dianni.mat')
data_baseline = helper.load_data_np('matlab_files/dianni_data/nalket_dianni_baseline.mat')
# hamming = helper.load_data_np('matlab_files/dianni_data/hamm_win.mat')
hamming = scipy.signal.windows.hamming(data_baseline.shape[-1])
windowed_data = hamming.T * data_baseline
# plot_subplots(windowed_data, 6, 14, 150, title="Low Correlation: 2.5-17.5min")



# plot_regions(windowed_data, 7, 2, 14)


corr = helper.get_corr_matrix(windowed_data)[0]
helper.plot_correlation_ROI(corr)
#high 8/12, low 2/14


#to do
# 1. align my data and recreate in comparison to Tommasso's
# 2. read more literature
# 3. Create standardized functions to do this
# 4. Make my correlation maps have his labels and neat and hamming
        #dictioniary with names could be convenient
# 5. Look into his detrending, 4th degree polynomials, etc. 
# 6. basleina nd 1 normlaization
