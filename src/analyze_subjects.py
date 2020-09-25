#!/usr/bin/env python

import rosbag
import matplotlib.pyplot as plt

import numpy as np
import os

from subjects import Subject, Filter

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from labelling import *

def main():

    save_path = "src/Dataset/5_Gest2_Ppl2"
    subj = []
    filters = []

    #Subject 1
    subj.append(Subject("Federica", "src/Dataset/5_gest_Fede1/5_gest_Fede1.bag"))
    filters.append((
        Filter(255, [0, 300]),
        Filter(570, [516, 721]),
        Filter(585, [1029, 1243]),
        Filter(425, [1430, 4571]),
        Filter(590, [4575, 5585]),
        Filter(1550, [5733, 5955]),
        Filter(390, [5940, -1]),
    ))

    # #Subject 2
    # subj.append(Subject("Stefano", "src/Dataset/5_gest_Ste2/5_gest_Ste2.bag"))
    # filters.append((Filter(722, [0, -1]), ))

    # #Subject 3
    # subj.append(Subject("Chiara", "src/Dataset/5_gest_Chia1/5_gest_Chia1.bag"))
    # filters.append((
    #     Filter(400, [0, 809]),
    #     Filter(504, [809, 1261]),
    #     Filter(585, [1484, 4809]),
    #     Filter(1120, [4809, 5307]),
    #     Filter(662, [5307, -1]),
    # ))

    # #Subject 4
    # subj.append(Subject("Jessica", "src/Dataset/5_gest_Je1/5_gest_Je1.bag"))
    # filters.append((
    #     Filter(1005, [0, 6825]),
    #     Filter(2265, [6825, -1]),
    # ))

    # #Subject 5
    # subj.append(Subject("Stefano2", "src/Dataset/5_gest_Ste2/5_gest_Ste2.bag"))
    # filters.append((
    #     Filter(530, [0, -1]),
    # ))

    # #Subject 6
    # subj.append(Subject("Stefano3", "src/Dataset/5_gest_Ste2/5_gest_Ste2.bag"))
    # filters.append((Filter(740, [0, -1]), ))

    # #Subject 7
    # subj.append(Subject("Stefano5", "src/Dataset/5_Gest_Matteo/5_Gest_Matteo.bag"))
    # filters.append((
    #     Filter(590, [0, -1]),
    # ))

    # #Subject 8
    # subj.append(Subject("Stefano6", "src/Dataset/5_Gest_Ste6/5_Gest_Ste6.bag"))
    # filters.append((
    #     Filter(245, [0, 363]),
    #     Filter(257, [478, 960]),
    #     Filter(265, [1015, 1391]),
    #     Filter(300, [1510, 2808]),
    #     Filter(716, [2900, -1]),
    # ))

    # #Subject 9
    # subj.append(Subject("Matteo", "src/Dataset/5_Gest_Matteo/5_Gest_Matteo.bag"))
    # filters.append((
    #     Filter(366, [0, 2903]),
    #     Filter(421, [2904, 5370]),
    #     Filter(567, [5371, 5819]),
    #     Filter(512, [5892, 7089]),
    #     Filter(488, [7089, 7861]),
    #     Filter(604, [7862, -1])
    # ))

    # #Subject 10
    # subj.append(Subject("Cristina2", "src/Dataset/5_Gest_Cri2/5_Gest_Cri2.bag"))
    # filters.append((
    #     Filter(205, [0, 266]),
    #     Filter(222, [498, 734]),
    #     Filter(213, [989, 1218]),
    #     Filter(203, [1508, 1751]),
    #     Filter(289, [1991, 2213]),
    #     Filter(220, [2507, 2743]),
    #     Filter(222, [2848, 3849]),
    #     Filter(233, [3850, 7838]),
    #     Filter(350, [7839, -1]),
    # ))

    #Subject 13
    subj.append(Subject("StefanoDx2", "src/Dataset/5_Gest2_Ste_Dx2/5_Gest2_Ste_Dx2.bag"))
    filters.append((
        Filter(850, [0, 1637]),
        Filter(1112, [1637, 1670]),
        Filter(850, [1670, 5445]),
        Filter(1027, [5045, 8514]),
        Filter(720, [8514, -1])
    ))

    # #Subject 14
    # subj.append(Subject("StefanoSx2", "src/Dataset/5_Gest2_Ste_Sx2/5_Gest2_Ste_Sx2.bag"))
    # filters.append((
    #     Filter(606, [0, 2025]),
    #     Filter(712, [2025, 3841]),
    #     Filter(495, [3841, 4035]),
    #     Filter(712, [4035, 9017]),
    #     Filter(584, [9017, 9487]),
    #     Filter(854, [9636, -1])
    # ))

    #Subject 15
    subj.append(Subject("StefanoDx", "src/Dataset/5_Gest2_Ste_Dx/5_Gest2_Ste_Dx.bag"))
    filters.append((
        Filter(490, [0, -1]),
    ))

    # #Subject 16
    # subj.append(Subject("StefanoSx", "src/Dataset/5_Gest2_Ste_Sx/5_Gest2_Ste_Sx.bag"))
    # filters.append((
    #     Filter(473, [0, 451]),
    #     Filter(310, [451, 556]),
    #     Filter(516, [556, 1883]),
    #     Filter(700, [1883, 2138]),
    #     Filter(516, [2138, 2381]),
    #     Filter(777, [2381, 4372]),
    #     Filter(890, [4372, 8646]),
    #     Filter(450, [8646, 10866]),
    #     Filter(640, [10866, -1]),
    # ))

    #Subject 16
    subj.append(Subject("Roberto1", "src/Dataset/5_Gest2_Roberto1/5_Gest2_Roberto1.bag"))
    filters.append((
        Filter(675, [0, 2740]),
        Filter(3365, [2740, 3103]),
        Filter(675, [3103, 4434]),
        Filter(1065, [4434, 7876]),
        Filter(426, [7876, 9110]),
        Filter(785, [9110, 9240]),
        Filter(495, [9240, -1])
    ))

    #Subject 16
    subj.append(Subject("Roberto2", "src/Dataset/5_Gest2_Roberto2/5_Gest2_Roberto2.bag"))
    filters.append((
        Filter(518, [0, 2920]),
        Filter(771, [2920, 7410]),
        Filter(490, [7410, -1])
    ))










    n_classes = 5

    #Create val_set, test_set, train_set
    for i in range(len(subj)):
        filt_channel = subj[i].applyFilter(filters[i])
        val_set1, test_set1, train_set1, gest1, val_labels1, test_labels1, train_labels1 = subj[i].createDataset(filt_channel)
        print(gest1.idx_gesture)
        val_label1 = labelling(val_set1.shape[0], n_classes)
        test_label1 = labelling(test_set1.shape[0], n_classes)
        train_label1 = labelling(train_set1.shape[0], n_classes)

        subj[i].plotChannels(filt_channel)
        plt.show()
        subj[i].plotDataset(gest1.idx_gesture, filt_channel)
        plt.show()

        # print(val_labels1)
        # print(test_labels1)
        # print(train_labels1)

        if i == 0:
            val_set = val_set1
            test_set = test_set1
            train_set = train_set1
            val_label = val_label1
            test_label = test_label1
            train_label = train_label1
        else:
            val_set = np.append(val_set, val_set1, axis=0)
            test_set = np.append(test_set, test_set1, axis=0)
            train_set = np.append(train_set, train_set1, axis=0)
            val_label = np.append(val_label, val_label1, axis=0)
            test_label = np.append(test_label, test_label1, axis=0)
            train_label = np.append(train_label, train_label1, axis=0)


    print(test_label)

    print("***")
    print("The GLOBAL shapes of the MERGED val_set, test_set, train_set are respectively:")
    print(val_set.shape)
    print(test_set.shape)
    print(train_set.shape)

    dataset = np.concatenate((val_set,test_set,train_set), axis=0)
    print("max: ", dataset.max())
    print("min: ", dataset.min())

    dataset = np.reshape(dataset, (dataset.shape[0], 120))
    print(dataset.shape)


    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)

    plt.hist(dataset, density=True, bins=20)
    plt.show()
    plt.hist(dataset_scaled, density=True, bins=20)
    plt.show()

    print(dataset_scaled.shape)


    dataset_scaled = np.reshape(dataset_scaled,
                                (dataset_scaled.shape[0], 8, 15))

    val_set = dataset_scaled[0:val_set.shape[0], :, :]
    test_set = dataset_scaled[val_set.shape[0]:
        val_set.shape[0] + test_set.shape[0], :, :]
    train_set = dataset_scaled[val_set.shape[0] + test_set.shape[0]:
        val_set.shape[0] + test_set.shape[0] + train_set.shape[0], :, :]


    try:
        os.mkdir(save_path)
    except OSError:
        print("Creation of the directory %s failed" % save_path)
    else:
        print("Successfully created the directory %s " % save_path)
    np.save(save_path + "/val_set", val_set)
    np.save(save_path + "/test_set", test_set)
    np.save(save_path + "/train_set", train_set)
    np.save(save_path + "/val_labels", val_label)
    np.save(save_path + "/test_labels", test_label)
    np.save(save_path + "/train_labels", train_label)

    from sklearn.externals.joblib import dump, load
    dump(scaler, save_path+'/std_scaler.bin', compress=True)



if __name__ == "__main__":
    main()