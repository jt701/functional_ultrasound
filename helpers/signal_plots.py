import helpers.helper as helper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import helpers.pre_processing as pre
import math
from scipy.signal import hilbert


#plots all regions for mouse group in 3 by 3 plot
def plot_regions(data, mouse, region1, region2, shift=0, xlab="Region 1", ylab="Region 2", title="Region 1 vs Region 2"):
    x_axis = np.arange(data.shape[-1]) + shift
    plt.plot(x_axis, data[mouse, region1, :], label=xlab, color='blue')
    plt.plot(x_axis, data[mouse, region2, :], label=ylab, color='red')

    plt.xlabel('Time(sec) post ketamine')
    plt.ylabel('Signal')
    plt.title(title)
    plt.legend()
    plt.show()

#plot corr, can reorder if necessary (if not using Tommasso's data)
def plot_corr(data, title="Correlations", reorder=False):
    corr = helper.get_corr_matrix(data)[0]
    if reorder:
        corr = reorder_corr(corr)
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
    
#gets indices for splicing, takes in center/length in minutes
#return left, right, shift
def get_splicing_info(center, length):
    right = center * 60 + length * 30 + 1200 
    left = center * 60 - length * 30 + 1200
    shift = left - 1200
    return left, right, shift

#maps label to index
#allows for errors with capitalization and spacing
def label_to_index(label):
    label_dict = {
    'ail': 0,
    'gidil': 1,
    's1l': 2,
    'm1l': 3,
    'm2l': 4,
    'cg1l': 5,
    'prll': 6,
    'ill': 7,
    'cpul': 8,
    'naccl': 9,
    'nacshl': 10,
    'nacshr': 11,
    'naccr': 12,
    'cpur': 13,
    'ilr': 14,
    'prlr': 15,
    'cg1r': 16,
    'm2r': 17,
    'm1r': 18,
    's1r': 19,
    'gidir': 20,
    'air': 21
    }
    key = label.replace(" ", "").lower()
    return label_dict[key]
    

#plotting subplot, helper function
def plot_subplot(ax, data, mouse, region1, region2, shift=100, title="Region 1 vs Region 2", ylabel="Signal"):
    x_axis = (np.arange(data.shape[-1]) + shift) / 60
    
    labels = ['AI L','GIDI L','S1 L','M1 L','M2 L','Cg1 L','PrL L','IL L','CPu L','NAcC L','NAcSh L',
    'NAcSh R','NAcC R','CPu R','IL R','PrL R','Cg1 R','M2 R','M1 R','S1 R','GIDI R','AI R']

    ax.plot(x_axis, data[mouse, region1, :], label=labels[region1], color='blue')
    ax.plot(x_axis, data[mouse, region2, :], label=labels[region2], color='red')

    ax.set_xlabel('Time(min) post ketamine')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize='small')
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # ax.legend()

    return ax

def plot_subplot_phase(ax, data, mouse, region1, region2, shift=100):
    x_axis = (np.arange(data.shape[-1]) + shift) / 60
    phase_difference = np.arccos(np.cos(data[mouse, region2, :] - data[mouse, region1, :])) * 180/math.pi
    labels = ['AI L','GIDI L','S1 L','M1 L','M2 L','Cg1 L','PrL L','IL L','CPu L','NAcC L','NAcSh L',
    'NAcSh R','NAcC R','CPu R','IL R','PrL R','Cg1 R','M2 R','M1 R','S1 R','GIDI R','AI R']
    
    ax.plot(x_axis, phase_difference, color='blue')
    r1 = labels[region1]
    r2 = labels[region2]
    title = "Phase Difference: " + r2 + " - " + r1
    ax.set_xlabel('Time(min) post ketamine')
    ax.set_ylabel("Phase Difference")
    ax.set_title(title, fontsize=8)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # ax.legend()

    return ax

def plot_subplots_phase(data, region1, region2, center, length, labels=False):
    if labels:
        region1 = label_to_index(region1)
        region2 = label_to_index(region2)
    left, right, shift = get_splicing_info(center, length)
    data = data[:, :, left:right]
    data = pre.hamming(data)
    hilbert_transform = hilbert(data, axis=-1)
    data = np.angle(hilbert_transform) 
    num_mice = data.shape[0]
    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            if 3*i + j > num_mice - 1:
                continue
            mouse_num = 3*i + j
            plot_subplot_phase(axes[i, j], data, mouse_num, region1,
                         region2, shift)
    title_big="Phase Differences"
    fig.suptitle(title_big)
    plt.tight_layout()
    plt.show()

def phase_diff(data, mouse, region1, region2, center, length, labels=False):
    
    if labels:
        region1 = label_to_index(region1)
        region2 = label_to_index(region2)
    left, right, shift = get_splicing_info(center, length)
    data = data[:, :, left:right]
    data = pre.hamming(data)
    hilbert_transform = hilbert(data, axis=-1)
    data = np.angle(hilbert_transform) 
    x_axis = (np.arange(data.shape[-1]) + shift) / 60
    phase_difference = np.arccos(np.cos(data[mouse, region2, :] - data[mouse, region1, :])) * 180/math.pi
    labels = ['AI L','GIDI L','S1 L','M1 L','M2 L','Cg1 L','PrL L','IL L','CPu L','NAcC L','NAcSh L',
    'NAcSh R','NAcC R','CPu R','IL R','PrL R','Cg1 R','M2 R','M1 R','S1 R','GIDI R','AI R']
    
    plt.plot(x_axis, phase_difference, color='blue')
    r1 = labels[region1]
    r2 = labels[region2]
    title = "Phase Difference: " + r2 + " - " + r1
    plt.xlabel('Time(min) post ketamine')
    plt.ylabel("Phase Difference")
    plt.title(title, fontsize=8)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # ax.legend()
    plt.show()

#plot several subplots
def plot_subplots(data, region1, region2, center, length, labels=False, title_big = "", ylabel="Signal", hilb=False):
    if labels:
        region1 = label_to_index(region1)
        region2 = label_to_index(region2)
    left, right, shift = get_splicing_info(center, length)
    data = data[:, :, left:right]
    data = pre.hamming(data)
    if hilb:
        hilbert_transform = hilbert(data, axis=-1)
        data = (np.angle(hilbert_transform))*180/math.pi
    num_mice = data.shape[0]
    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            if 3*i + j > num_mice - 1:
                continue
            mouse_num = 3*i + j
            title = "Mouse" + str(mouse_num)
            plot_subplot(axes[i, j], data, mouse_num, region1,
                         region2, shift, title=title, ylabel=ylabel)
    fig.suptitle(title_big)
    plt.tight_layout()
    plt.show()
    
#plot several subplots
def plot_phases_both(data, mouse_num, region1, region2, center, length, labels=False, unwrap = False, title_big = ""):
    if labels:
        region1 = label_to_index(region1)
        region2 = label_to_index(region2)
    left, right, shift = get_splicing_info(center, length)
    data = data[:, :, left:right] 
    data = pre.hamming(data)
    hilbert_transform = hilbert(data, axis=-1)
    if unwrap:
        data = np.unwrap(np.angle(hilbert_transform)) * 180/math.pi
    else:
         data = np.angle(hilbert_transform) * 180/math.pi
    fig, axes = plt.subplots(figsize=(12, 8))
    plot_subplot(axes, data, mouse_num, region1,
                    region2, shift, title="Phase Plots", ylabel="Phase Angle")
    fig.suptitle(title_big)
    plt.tight_layout()
    plt.show()

#reorder correlation matrix to be more brain oriented, only when going from my data to Tommasso's
def reorder_corr(corr):
    brain_ordering = ['AI L','GIDI L','S1 L','M1 L','M2 L','Cg1 L','PrL L','IL L','CPu L','NAcC L','NAcSh L',
    'NAcSh R','NAcC R','CPu R','IL R','PrL R','Cg1 R','M2 R','M1 R','S1 R','GIDI R','AI R']
    old_ordering = [
    'DP L', 'DP R', 'IL L', 'IL R', 'PrL L', 'PrL R', 'Cg1 L', 'Cg1 R',
    'M2 L', 'M2 R', 'M1 L', 'M1 R', 'S1 L', 'S1 R', 'GIDI L', 'GIDI R',
    'Cl L', 'Cl R', 'CPu L', 'CPu R', 'NAcC L', 'NAcC R', 'NAcSh L', 'NAcSh R',
    'S1DZ L', 'S1DZ R', 'S1J L', 'S1J R', 'AI L', 'AI R'
    ]
    old_dict = {old_ordering[i]:i for i in range(len(old_ordering))}
    
    num_reg = len(brain_ordering)
    result = np.zeros((num_reg, num_reg))
    for i in range(num_reg):
        for j in range(num_reg):
            first_region = brain_ordering[i]
            sec_region = brain_ordering[j]
            first_index = old_dict[first_region]
            sec_index = old_dict[sec_region]
            result[i,j] = corr[first_index, sec_index]
    return result

#normalize based off baseline data using standard deviations
def normalize_to_baseline(baseline, sample):
    baseline_means = np.mean(baseline, axis=2)
    baseline_std = np.std(baseline, axis=2)
    baseline_means = baseline_means[:, :, np.newaxis]
    baseline_std = baseline_std[:, :, np.newaxis] 
    normalized_data = (sample - baseline_means) / baseline_std
    return normalized_data

#normalize sample to itself
def normalize_to_sample(sample):
    sample_mean = np.mean(sample, axis=2)
    sample_std = np.std(sample, axis=2)
    sample_mean = sample_mean[:, :, np.newaxis]
    sample_std = sample_std[:, :, np.newaxis]
    return (sample-sample_mean) / sample_std
            
#plotting subplots of Dianni data

def load_dianni():
    data = helper.load_data_np('matlab_files/dianni_data/nalket_dianni.mat') #di ianni data
    hamming = helper.load_data_np('matlab_files/dianni_data/hamm_win.mat')
    windowed_data = hamming.T * data
    plot_corr(windowed_data)

    # data_baseline = helper.load_data_np('matlab_files/dianni_data/nalket_dianni_baseline.mat')
    # hamming_baseline = scipy.signal.windows.hamming(data_baseline.shape[-1])

    # plot_subplots(windowed_data, 2, 10, 150, title_big="Low Correlation: 2.5-17.5min")
# load_dianni()

#my data corr and plots
def plot_mine():
    data = helper.load_data_np('matlab_files/Time_Series_Data/ROI_CBV_Data/salket_m_roi.mat')
    data = pre.bandpass_filter(data, 0.01, 0.1, 5)[:, :, 2500:3500]
    hamming = scipy.signal.windows.hamming(data.shape[-1])
    data = hamming.T * data
    plot_corr(data, reorder=True)
    # plot_corr(data, reorder=True)
    # corr = helper.get_corr_matrix(data)[0]
    # corr = reorder_corr(corr)
    # helper.plot_correlation_ROI(corr)
# plot_regions(windowed_data, 7, 2, 14)

# plot_mine()
# def random_corr_matrix():
#     corr = helper.get_corr_matrix(windowed_data)[0]
#     plot_corr(windowed_data)
#     high 8/12, low 2/14

def normalize_test():
    data_baseline = helper.load_data_np('matlab_files/dianni_data/nalket_dianni_baseline.mat')
    data = helper.load_data_np('matlab_files/dianni_data/nalket_dianni.mat')
    hamming_baseline = scipy.signal.windows.hamming(data_baseline.shape[-1])
    hamming_data = scipy.signal.windows.hamming(data.shape[-1])
    data_baseline = hamming_baseline.T * data_baseline
    data = hamming_data.T * data
    # data = normalize_to_baseline(data_baseline, data) 
    data = normalize_to_sample(data)
    plot_subplots(data, 8, 13, 150)
    # plot_corr(data, title="Correlations 2.5-17.5min")

# normalize_test()
