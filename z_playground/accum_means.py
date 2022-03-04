import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter


matplotlib.use('TkAgg')

attempts = ["ident", "hamming", "added_v1", "SMOTE"]
nets = ["cnn", "mlp"]


for net in nets:
    plt.clf()
    for attempt in attempts:

        tmpRank = np.loadtxt("means/"+attempt+"-"+net +
                             "-mean_rank.csv", delimiter=',')
        x_test = np.arange(0, len(tmpRank)*25, 25)

        plt.plot(x_test, tmpRank, label=attempt)
        plt.xlim([0.0, 2000])
        plt.ylim([0.0, 200])
        plt.ylabel("Rang")
        plt.xlabel("Traceanzahl")
        plt.legend(["Schlüsselidentität", "Hamming-Gewicht-Modell",
                    "Erwartungswert-Modellierung", "SMOTE"])

    plt.savefig("means/"+net + "-mean_rank.png")
    # plt.show()
