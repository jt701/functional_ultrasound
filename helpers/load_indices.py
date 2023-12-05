import helper

def test():
    a = helper.load_data_np('matlab_files/dianni_data/plot_rois.mat')
    print(a)
    a = list(a)
