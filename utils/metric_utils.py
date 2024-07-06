import numpy as np

def accuracy_total_error(acc_list):
    n_task = len(acc_list)
    return 100 * (1.0 * n_task - sum(acc_list))

def normalized_gain(acc_list, ref_acc_list):
    ref_acc = np.array(ref_acc_list)
    ref_error = 1.0 - ref_acc
    acc_list = np.array(acc_list)
    acc_error = 1.0 - acc_list
    res = np.mean((ref_error - acc_error) / ref_error) 
    return 100 * res
