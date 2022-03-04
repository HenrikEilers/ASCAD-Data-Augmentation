import numpy as np
from ASCAD_train_models import load_ascad
import matplotlib.pyplot as plt
import matplotlib
import tkinter


class mean_std_special:

    def __init__(self, ascad):
        self.testX = np.arange(700)
        self.ident_mean(ascad)
        self.hamming_mean(ascad)
        self.ident_std(ascad)
        self.hamming_std(ascad)

    def ident_mean(self, ascad):
        (X_profiling, Y_profiling), (X_attack,
                                     Y_attack), (Metadata_profiling,
                                                 Metadata_attack) = ascad
        # segmentation
        self.segmented_ident_plot = {}
        for index, value in enumerate(X_profiling):
            indexindex = Y_profiling[index]
            if indexindex not in self.segmented_ident_plot:
                self.segmented_ident_plot[indexindex] = list()

            self.segmented_ident_plot[indexindex].append(value)

        # mean calculation
        plt.clf()
        retVal = list()
        for value in self.segmented_ident_plot:
            value = np.asarray(self.segmented_ident_plot[value])
            value = np.mean(value, axis=0)
            retVal.append(value)
            plt.plot(self.testX, value)
        plt.xlim([490, 500])
        plt.ylim([0, 20])
        plt.ylabel("Erwartungswert")
        plt.xlabel("Zeitabschnitt im Trace")
        plt.show()
        plt.savefig("figure/std_ident.png")

    def hamming_mean(self, ascad):
        (X_profiling, Y_profiling), (X_attack,
                                     Y_attack), (Metadata_profiling,
                                                 Metadata_attack) = ascad
        # segmentation
        self.segmented_hamming_plot = {}
        for index, value in enumerate(X_profiling):
            indexindex = bin(Y_profiling[index]).count("1")
            if indexindex not in self.segmented_hamming_plot:
                self.segmented_hamming_plot[indexindex] = list()

            self.segmented_hamming_plot[indexindex].append(value)

        # mean calculation
        plt.clf()
        retVal = list()
        for value in self.segmented_hamming_plot:
            value = np.asarray(self.segmented_hamming_plot[value])
            value = np.mean(value, axis=0)
            retVal.append(value)
            plt.plot(self.testX, value)
        plt.xlim([490, 500])
        plt.ylim([0, 20])
        plt.ylabel("Erwartungswert")
        plt.xlabel("Zeitabschnitt im Trace")
        plt.show()
        plt.savefig("figure/mean_hamming.png")

        plt.clf()
        plt.plot(self.testX, np.mean(self.segmented_hamming_plot[0], axis=0))
        #plt.plot(self.testX, self.segmented_hamming_plot[0][0])
        #plt.plot(self.testX, self.segmented_hamming_plot[0][1])
        #plt.plot(self.testX, self.segmented_hamming_plot[0][2])
        #plt.plot(self.testX, self.segmented_hamming_plot[0][3])
        #plt.plot(self.testX, self.segmented_hamming_plot[0][4])
        #plt.plot(self.testX, self.segmented_hamming_plot[0][5])
        #plt.legend(["mean", "0", "1", "2"])
        # plt.show()

    def ident_std(self, ascad):
        (X_profiling, Y_profiling), (X_attack,
                                     Y_attack), (Metadata_profiling,
                                                 Metadata_attack) = ascad
        plt.clf()
        retVal = list()
        for value in self.segmented_hamming_plot:
            value = np.asarray(self.segmented_hamming_plot[value])
            value = np.std(value, axis=0)
            retVal.append(value)
            plt.plot(self.testX, value)
        plt.savefig("figure/std_ident.png")

    def hamming_std(self, ascad):
        (X_profiling, Y_profiling), (X_attack,
                                     Y_attack), (Metadata_profiling,
                                                 Metadata_attack) = ascad
        plt.clf()
        retVal = list()
        for value in self.segmented_hamming_plot:
            value = np.asarray(self.segmented_hamming_plot[value])
            value = np.std(value, axis=0)
            retVal.append(value)
            plt.plot(self.testX, value)
        plt.savefig("figure/std_hamming.png")
