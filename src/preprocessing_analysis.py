import rosbag
import matplotlib.pyplot as plt
import numpy as np

from subjects import Subject, Gestures

def main():

    """ This script helps to analyze the sum of raw data among all channels
    of a subject. In this way you can observe the possibile limits to apply
    filters in other scripts"""

    subj = []
    # subj.append(Subject("Federica", "src/Dataset/5_gest_Fede1/5_gest_Fede1.bag"))
    # subj.append(Subject("Stefano", "src/Dataset/5_gest_Ste1/5_gest_Ste1.bag"))
    # subj.append(Subject("Chiara", "src/Dataset/5_gest_Chia1/5_gest_Chia1.bag"))
    # subj.append(Subject("Jessica", "src/Dataset/5_gest_Je1/5_gest_Je1.bag"))
    # subj.append(Subject("Stefano2", "src/Dataset/5_gest_Ste2/5_gest_Ste2.bag"))
    # subj.append(Subject("Stefano3", "src/Dataset/5_gest_Ste3/5_gest_Ste3.bag"))
    # subj.append(Subject("Stefano5", "src/Dataset/5_Gest_Ste5/5_Gest_Ste5.bag"))
    # subj.append(Subject("Stefano6", "src/Dataset/5_Gest_Ste6/5_Gest_Ste6.bag"))
    # subj.append(Subject("Cristina2", "src/Dataset/5_Gest_Cri2/5_Gest_Cri2.bag"))
    # subj.append(Subject("Matteo", "src/Dataset/5_Gest_Matteo/5_Gest_Matteo.bag"))

    # subj.append(Subject("StefanoDx", "src/Dataset/5_Gest2_Ste_Dx/5_Gest2_Ste_Dx.bag"))
    # subj.append(Subject("StefanoSx", "src/Dataset/5_Gest2_Ste_Sx/5_Gest2_Ste_Sx.bag"))
    subj.append(Subject("StefanoDx2", "src/Dataset/5_Gest2_Ste_Dx2/5_Gest2_Ste_Dx2.bag"))
    # subj.append(Subject("StefanoSx", "src/Dataset/5_Gest2_Ste_Sx2/5_Gest2_Ste_Sx2.bag"))
    # subj.append(Subject("Roberto1", "src/Dataset/5_Gest2_Roberto1/5_Gest2_Roberto1.bag"))
    # subj.append(Subject("Roberto2", "src/Dataset/5_Gest2_Roberto2/5_Gest2_Roberto2.bag"))

    for i in range(len(subj)):
        plt.subplot(len(subj),1,i+1)
        subj[i].plotChannelSum()

    plt.show()


if __name__ == "__main__":
    main()