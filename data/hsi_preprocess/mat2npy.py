import os
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data_mat_dir = '/home/disk/data/HSI/'

datasets = ['KSC', 'PaviaU', 'Indian_pines', 'HoustonU','Salinas']
dataset_HSIs = ['KSC', 'paviaU', 'indian_pines_corrected', 'IGRSS_2013','salinas_corrected']
dataset_gts = ['KSC_gt', 'paviaU_gt', 'indian_pines_gt', 'IGRSS_2013_gt','salinas_gt']

def convert(dataset, dataset_HSI, dataset_gt):
    dataset_mat_dir = data_mat_dir + '{}/{}.mat'.format(dataset, dataset)
    dataset_gt_dir = data_mat_dir + '{}/{}_gt.mat'.format(dataset, dataset)

    if dataset in ['Indian_pines', 'PaviaU', 'KSC','Salinas']:
        HSI_data = sio.loadmat(dataset_mat_dir)[dataset_HSI]
        HSI_gt = sio.loadmat(dataset_gt_dir)[dataset_gt]

    else:
        HSI_data = h5py.File(dataset_mat_dir)[dataset_HSI][:]
        HSI_data = HSI_data.transpose(1,2,0)
        HSI_gt = h5py.File(dataset_gt_dir)[dataset_gt][:]

    np.save(dataset_mat_dir.replace('.mat', '_data.npy'), HSI_data)
    np.save(dataset_mat_dir.replace('.mat', '_label.npy'), HSI_gt)

    print('{} convert done!'.format(dataset))

if __name__ == "__main__":

    for dataset, dataset_HSI, dataset_gt in zip(datasets, dataset_HSIs, dataset_gts):
        convert(dataset, dataset_HSI, dataset_gt)






