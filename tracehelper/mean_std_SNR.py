import numpy as np
from ASCAD_train_models import load_ascad
import matplotlib.pyplot as plt
import matplotlib
import tkinter


class mean_std:

    def __init__(self, ascad):
        self.testX = np.arange(700)
        self.mean(ascad)
        self.std(ascad)
        self.mean_std_special(ascad)
        self.do_SNR()

    def mean(self, ascad):
        (X_profiling, Y_profiling), (X_attack,
                                     Y_attack), (Metadata_profiling,
                                                 Metadata_attack) = ascad

        # MEAN
        plt.clf()
        self.testY = np.mean(X_profiling, axis=0)
        plt.plot(self.testX, self.testY)
        #plt.plot(self.testX, X_profiling[0])
        #plt.plot(self.testX, X_profiling[1])
        #plt.plot(self.testX, X_profiling[2])
        #plt.legend(["mean", "0", "1", "2"])
        # plt.show()
        plt.savefig("figure/mean.png")

    def std(self, ascad):
        (X_profiling, Y_profiling), (X_attack,
                                     Y_attack), (Metadata_profiling,
                                                 Metadata_attack) = ascad

        self.testDeviation = np.std(X_profiling, axis=0)
        plt.clf()
        plt.plot(self.testX, self.testDeviation)
        plt.savefig("figure/std.png")

    def mean_std_special(self, ascad):
        (X_profiling, Y_profiling), (X_attack,
                                     Y_attack), (Metadata_profiling,
                                                 Metadata_attack) = ascad
        # MEAN+STD_Special
        stdDevplus = self.testY + self.testDeviation
        stdDevminus = self.testY - self.testDeviation
        plt.clf()
        # plt.figure().set_figwidth(50, forward=True)
        plt.fill_between(self.testX, stdDevminus, stdDevplus)
        plt.plot(self.testX, self.testY, color="r")
        plt.savefig("figure/mean_dev_special.png")

    def do_SNR(self):
        plt.clf()
        plt.plot(self.testX, np.log10(
            self.testY/self.testDeviation)*10, color="r")
        # plt.show()
        plt.savefig("figure/SNR.png")
