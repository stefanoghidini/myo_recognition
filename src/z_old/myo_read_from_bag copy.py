#!/usr/bin/env python

import rosbag
import matplotlib.pyplot as plt
import numpy as np

class Filter:
    def __init__(self, max_value=0, limits_zero=[]):
        self.max_value = max_value
        self.limits_zero = limits_zero

    def getMaxValue(self):
        return self.max_value

    def getLimitsZero(self):
        return self.limits_zero


def createDataset(bag_file, filters):
    # bag = rosbag.Bag("src/Dataset/5_gest_Cri1/5_gest_Cri1.bag")

    bag = rosbag.Bag(bag_file)

    channel = np.zeros((8, 1), dtype="int32")
    # channel = np.reshape(channel, (1,1))

    for topic, msg, t in bag.read_messages(topics=['/myo_raw/myo_emg']):
        data = np.array(msg.data, dtype="int32")
        data = np.reshape(data, (8,1))

        channel = np.append(channel, data, axis=1)

    bag.close()


    channel_sum = channel.sum(axis=0)
    plt.plot(channel_sum)
    plt.show()
    channel_sum = np.reshape(channel_sum, (1, channel_sum.shape[0]))

    #Create filter to remove rest and noise position data. Filter executed on the sum data of channels
    filt = np.ones((1, channel_sum.shape[1]))
    for i in range(len(filters)):
        limits_zero = filters[i].getLimitsZero()
        max_value = filters[i].getMaxValue()
        if limits_zero[1] == -1:
            limits_zero[1] = channel_sum.shape[1]
        for j in range(limits_zero[0], limits_zero[1]):
            if channel_sum[0, j] <= max_value:
                filt[0,j] = 0

    #Multiply channels for filters
    filt_channel = np.empty((8, channel.shape[1]), dtype="int32")
    for i in range(channel.shape[0]):
        filt_channel[i,:] = channel[i, :] * filt
        plt.plot(filt_channel[i, :])

    #Create Validation, Test, Train Data
    c = 0 #Number of total repetition
    c1 = 0 #Number of repetition on a single exercise
    end_ges_idxs = [0]
    check = True

    val_set = np.empty((1, 8, 15), dtype="int32")
    test_set = np.empty((1, 8, 15), dtype="int32")
    train_set = np.empty((1, 8, 15), dtype="int32")

    for i in range(0,filt_channel.shape[1]-6,6):

        if (filt_channel[:, i:i + 15].shape == (8,15)) :
            matrix = filt_channel[:, i:i + 15]
        matrix_sum = matrix.sum()

        matrix = np.reshape(matrix, (1,8,15))

        #Count starting of each repetition
        if matrix_sum != 0 :
            if check == True:
                c += 1
                plt.plot(i, 5,'bo')
                c1 += 1
                if c1 > 5:
                    end_ges_idxs.append(i)
                    c1 = 1
                check = False
        else:
            check = True

        #First repetition becomes val_set
        if (c1 == 1 and matrix_sum != 0):
            val_set = np.append(val_set, matrix, axis=0)
        #Second repetition becomes test_set
        elif (c1 == 2 and matrix_sum != 0):
            test_set = np.append(test_set, matrix, axis=0)
        #Other repetitions become train_set
        else:
            if matrix_sum != 0 :
                train_set = np.append(train_set, matrix, axis=0)

    #Append last index of channels to have a reference of last gesture
    end_ges_idxs.append(filt_channel.shape[1])

    plt.show()


    print("Bag file: " + bag_file)
    print("Number of repetition is: " + str(c))

    for i in range(len(end_ges_idxs)-1):
        plt.subplot(5,1,i+1)
        for j in range(channel.shape[0]):
            plt.plot(filt_channel[j,end_ges_idxs[i]:end_ges_idxs[i+1]])
    plt.show()

    return val_set, test_set, train_set



def main():

    filters = (
        Filter(255, [0, 300]),
        Filter(570, [516, 721]),
        Filter(585, [1029, 1243]),
        Filter(425, [1430, 4571]),
        Filter(590, [4575, 5585]),
        Filter(1550, [5733, 5955]),
        Filter(390, [5940, -1]),
    )
    bag_file = "src/Dataset/5_gest_Fede1/5_gest_Fede1.bag"
    val_set1, test_set1, train_set1 = createDataset(bag_file, filters)


    max_value = 630
    filt_corr = FilterCorrector(max_value)
    bag_file = "src/Dataset/5_gest_Ste1/5_gest_Ste1.bag"
    val_set2, test_set2, train_set2 = createDataset(bag_file, filt_corr)

    # bag_file = "src/Dataset/5_gest_Ste1/5_gest_Ste1.bag"
    # val_set1, test_set1, train_set1 = createDataset(bag_file, filt_corr)


    val_set = np.append(val_set1, val_set2, axis=0)
    test_set = np.append(test_set1, test_set2, axis=0)
    train_set = np.append(train_set1, train_set2, axis=0)




    np.save("src/Dataset/5_gest_Ppl2_1/val_set",val_set)
    np.save("src/Dataset/5_gest_Ppl2_1/test_set", test_set)
    np.save("src/Dataset/5_gest_Ppl2_1/train_set", train_set)

    print(val_set1.shape)
    print(test_set1.shape)
    print(train_set1.shape)

    print(val_set.shape)
    print(test_set.shape)
    print(train_set.shape)

if __name__ == "__main__":
    main()