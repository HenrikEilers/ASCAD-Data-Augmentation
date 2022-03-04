from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter


def do_snr(ascad):

    (X_profiling, Y_profiling), (X_attack,
                                 Y_attack), (Metadata_profiling,
                                             Metadata_attack) = ascad

    testX = np.arange(700)

    # legend = ["Output S-Box", "Output S-Box Ohne Maske1",
    #          "Maske1", "Output S-Box Ohne Maske2", "Maske2", ]
    #legend = ["Hamming-Gewicht-Modell", "Schlüsselidentität"]
    count = 1
    toTestArray = [list()]  # , list(), list(), list(), list()]
    for index, value in enumerate(Y_profiling):
        toTestArray[0].append(value)
        #toTestArray[1].append(value ^ Metadata_profiling[index]["masks"][15])
        # toTestArray[2].append(Metadata_profiling[index]["masks"][15])
        #toTestArray[3].append(value ^ Metadata_profiling[index]["masks"][0])
        # toTestArray[4].append(Metadata_profiling[index]["masks"][0])
    final = np.zeros((5, 700))
    for hamming in [True, False]:
        count = 1
        for Y_profiling in toTestArray:
            # segmentation
            segmented_ident_plot = {}
            for index, value in enumerate(X_profiling):
                if hamming == True:
                    indexindex = bin(Y_profiling[index]).count("1")
                else:
                    indexindex = Y_profiling[index]
                if indexindex not in segmented_ident_plot:
                    segmented_ident_plot[indexindex] = list()

                segmented_ident_plot[indexindex].append(value)

            retVal = list()
            for value in segmented_ident_plot:
                value = np.asarray(segmented_ident_plot[value])
                retVal.append(value)

            meanTMP = mean(retVal)
            meanTMP = np.asarray(meanTMP)
            varTMP = var(retVal)
            varTMP = np.asarray(varTMP)
            final = np.var(meanTMP, axis=0)/np.mean(varTMP, axis=0)
            color = ""
            if count == 1:
                if hamming == True:
                    plt.plot(testX, final, color="red" if hamming ==
                             True else "blue", label="Hamming-Gewicht-Modell")
                else:
                    plt.plot(testX, final, color="red" if hamming ==
                             True else "blue", label="Schlüsselidentität")
            else:
                plt.plot(testX, final, color="red" if hamming ==
                         True else "blue")
            count += 1
    plt.legend()
    plt.ylabel("SNR")
    plt.xlabel("Zeitabschnitt im Trace")
    plt.show()
    # print(final)

    plt.clf()
    # , color="red" if hamming == True else "blue")
    plt.plot(np.arange(700), X_profiling[0])
    # plt.legend(legend)
    plt.ylabel("Seitenkanal Messung")
    plt.xlabel("Zeit")
    plt.show()
    print(Y_profiling[0])
    print(Metadata_profiling[0]['plaintext'][2])
    print(Metadata_profiling[0]['key'][2])
    print(Metadata_profiling[0]["masks"][0])
    print(Metadata_profiling[0]["masks"][15])

    array1 = np.zeros(256)
    for tmp in toTestArray[0]:
        array1[tmp] = array1[tmp]+1
    array1 = array1/50000
    plt.clf()
    # , color="red" if hamming == True else "blue")
    plt.plot(np.arange(256), array1)
    # plt.legend(legend)
    plt.ylabel("Auftrittswahrscheinlichkeiten")
    plt.xlabel("Labelkanidat")
    plt.ylim(0, 0.2)
    plt.show()
    print(Y_profiling[0])
    print(Metadata_profiling[0]['plaintext'][2])
    print(Metadata_profiling[0]['key'][2])
    print(Metadata_profiling[0]["masks"][0])
    print(Metadata_profiling[0]["masks"][15])


def var(data):
    data1 = list()
    for index, value in enumerate(data):
        data1.append(np.var(value, axis=0))
    return data1


def mean(data):
    data1 = list()
    for index, value in enumerate(data):
        data1.append(np.mean(value, axis=0))
    return data1
