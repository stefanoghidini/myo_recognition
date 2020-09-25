#!/usr/bin/env python

from ros_myo.msg import EmgArray
import rospy
import numpy as np
from std_msgs.msg import String
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.externals.joblib import dump, load


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class MyoInterface :

    channels = np.empty((8, 1), dtype="int32")
    count = 0


    def __init__(self):
        print("create class")
        rospy.Subscriber("myo_raw/myo_emg", EmgArray, self.callback, queue_size=1)
        self.img = np.zeros((8, 15))
        self.matrix = np.zeros((8, 15))

    def getChannels(self):
        return self.channels

    def getMatrix(self):
        # self.img = (self.img - 194.305684870188) / 164.66917199229

        return self.img


    def callback(self,data):
        self.channels = np.array(data.data)
        self.channels = self.channels.reshape((8, 1))

        self.matrix = np.append(self.matrix, self.channels, axis=1)
        self.matrix = np.delete(self.matrix, 0, axis=1)

        self.count += 1

        if self.count >= 6:
            self.img = self.matrix
            self.count = 0




def main():
    print("Start main")

    model_name = "5_Gest2_Ppl2"
    model = load_model("src/Models/Myo_Model_" + model_name + ".h5")
    model.summary()
    sc = load("src/Dataset/"+model_name+"/std_scaler.bin")

    rospy.init_node('listener', anonymous=True)
    r = rospy.Rate(50)  # 200hz

    myo = MyoInterface()

    while not rospy.is_shutdown() :

        img = myo.getMatrix()

        img = np.reshape(img,(1,120))
        img = sc.transform(img)
        img = np.reshape(img,(1, 1, 8, 15))
        img = np.transpose(img, (0, 2, 3, 1))
        predict = model.predict_classes(img)

        print(predict)
        print("***")

        r.sleep()

    # rospy.spin()


if __name__ == "__main__":
    main()
