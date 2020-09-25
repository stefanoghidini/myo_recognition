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

class Gestures:
    def __init__(self):
        self.count_move = 0 #Count number of each arm movements
        self.count_rep = 0 #Count number of repetition for each gesture
        self.count_gest = 0
        self.idx_gesture = [0] #Index of the end of each gesture


class Subject:
    def __init__(self, name, bag_file):
        """Create a subject object using EMG data from the specified bag_file,
         red from the /myo_raw/myo_emg topic"""

        self.name = name
        self.channel = np.zeros((8, 1), dtype="int32")

        bag = rosbag.Bag(bag_file)

        for topic, msg, t in bag.read_messages(topics=['/myo_raw/myo_emg']):
            data = np.array(msg.data, dtype="int32")
            data = np.reshape(data, (8, 1))
            self.channel = np.append(self.channel, data, axis=1)

        bag.close()

        print("***")
        print("Created subject object named " + name + " using EMG data from: " + bag_file)



    def applyFilter(self, filters):
        channel_sum = self.channel.sum(axis=0)
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
                    filt[0, j] = 0

        #Multiply channels for filters
        filt_channel = np.empty((8, self.channel.shape[1]), dtype="int32")
        for i in range(self.channel.shape[0]):
            filt_channel[i, :] = self.channel[i, :] * filt

        return filt_channel


    def createDataset(self, filt_channel):
        #Create Validation, Test, Train Data
        gest = Gestures()
        check = True

        val_set = np.zeros((1, 8, 15), dtype="float64")
        test_set = np.zeros((1, 8, 15), dtype="float64")
        train_set = np.zeros((1, 8, 15), dtype="float64")

        val_labels = []
        test_labels = []
        train_labels = []

        for i in range(0,filt_channel.shape[1]-6,6): #Step era 6!!!!

            if (filt_channel[:, i:i + 15].shape == (8,15)) :
                matrix = filt_channel[:, i:i + 15]
            matrix_sum = matrix.sum()

            matrix = np.reshape(matrix, (1,8,15))

            #Count starting of each repetition
            if matrix_sum != 0 :
                if check == True:
                    gest.count_rep += 1
                    gest.count_move += 1
                    if gest.count_rep > 5:
                        gest.idx_gesture.append(i)
                        gest.count_rep = 1
                        gest.count_gest += 1
                    check = False
            else:
                check = True

            #First repetition becomes val_set
            if (gest.count_rep == 1 and matrix_sum != 0):
                val_set = np.append(val_set, matrix, axis=0)
                val_labels.append(gest.count_gest)
            #Second repetition becomes test_set
            elif (gest.count_rep == 2 and matrix_sum != 0):
                test_set = np.append(test_set, matrix, axis=0)
                test_labels.append(gest.count_gest)
            #Other repetitions become train_set
            else:
                if matrix_sum != 0 :
                    train_set = np.append(train_set, matrix, axis=0)
                    train_labels.append(gest.count_gest)

        gest.idx_gesture.append(self.channel.shape[1])

        print("***")
        print("Succesfully created a dataset for subject: " + self.name)
        print("Number of total arm movements is: " + str(gest.count_move))
        print("The shapes of val_set, test_set, train_set are respectively:")
        print(val_set.shape)
        print(test_set.shape)
        print(train_set.shape)

        return val_set, test_set, train_set, gest, val_labels, test_labels, train_labels


    def plotDataset(self, idx_gesture, channel):
        #Append last index of channels to have a reference of last gesture
        for i in range(len(idx_gesture) - 1):
            plt.subplot(5, 1, i + 1)
            for j in range(channel.shape[0]):
                plt.plot(channel[j, idx_gesture[i]:idx_gesture[i + 1]])


    def plotChannelSum(self):
        """Plot the sum of not filtered EMG channels data read from bag file
        specified in the constructor"""
        channel_sum = self.channel.sum(axis=0)
        plt.title(self.name)
        plt.xlabel("# samples")
        plt.ylabel("Amplitude")
        plt.plot(channel_sum)

    def plotChannels(self, channel):
        """Plot the EMG channel data specified in the argument channel"""
        for i in range(channel.shape[0]):
            plt.plot(channel[i, :])
            plt.legend(loc="upper left")
        plt.title(self.name)
        plt.xlabel("# samples")
        plt.ylabel("Amplitude")
