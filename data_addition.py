# load traces
from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter

import ASCAD_train_models
import ASCAD_test_models
from tracehelper.special_mean_std import mean_std_special

matplotlib.use('TkAgg')


def create_data(mean, std, amount):
    retValue = np.zeros((len(mean), amount), dtype=int)
    for datapointIndex in range(len(mean)):
        retValue[datapointIndex] = np.random.normal(
            loc=mean[datapointIndex], scale=std[datapointIndex], size=amount)
    retValue = retValue.T
    return retValue


def create_data_SMOTE(org_taces, amount):
    k = len(org_taces)-1
    numattrs = len(org_taces[0])
    sample = org_taces
    newindex = 0
    synthetic = np.zeros((amount, len(sample[0])))
    N = amount
    T = len(sample)

    nnarray_range = np.arange(0, len(sample))

    overflowAmount = amount % len(sample)
    normalAmount = int(amount/len(sample))

    for index, value in enumerate(sample):
        if index < overflowAmount:
            i_amount = normalAmount+1
        else:
            i_amount = normalAmount
        nnarray = nnarray = np.concatenate(
            (nnarray_range[0:index], nnarray_range[index+1:T]))
        (newindex, synthetic) = populate_SMOTE(i_amount,
                                               index, sample, newindex, nnarray, synthetic, k, amount, normalAmount)

    return synthetic


def populate_SMOTE(i_amount, i, sample, newindex, nnarray, synthetic, k, amount, normalAmount):
    while i_amount != 0:
        nn = int(np.around(np.random.uniform(0, k-1)))
        for attr in np.arange(0, len(sample[i])):
            dif = sample[nnarray[nn]][attr] - sample[i][attr]
            gap = np.random.uniform(0, 1)
            synthetic[newindex][attr] = sample[i][attr] + \
                int(np.around(gap*dif))
        i_amount = i_amount-1
        newindex = newindex + 1
    return (newindex, synthetic)


def add_data(ascad):
    (X_profiling, Y_profiling), (X_attack,
                                 Y_attack), (Metadata_profiling,
                                             Metadata_attack) = ascad

    ret = list()
    for i in Y_profiling:
        # print(bin(i).count("1"))
        ret.append(bin(i).count("1"))

# segmenting
    segmented_hamming_plot = {}
    tmptmp = np.zeros(9)
    for index, value in enumerate(X_profiling):
        indexindex = bin(Y_profiling[index]).count("1")
        if indexindex not in segmented_hamming_plot:
            segmented_hamming_plot[indexindex] = list()
        tmptmp[indexindex] += 1
        segmented_hamming_plot[indexindex].append(value)

    step = 1
    XTest = np.arange(0, 9, step)

    # plt.clf()
    total_plot_length = np.zeros(9, dtype=int)  # alle traces
    total_plot_means = np.zeros((9, 700))  # mean aller traces
    total_plot_std = np.zeros((9, 700))  # std aller traces
    total = 0

    for value in segmented_hamming_plot:
        total_plot_length[value] = int(len(segmented_hamming_plot[value]))
        total_plot_means[value] = np.mean(
            np.asarray(segmented_hamming_plot[value]), axis=0)
        total_plot_std[value] = np.std(
            np.asarray(segmented_hamming_plot[value]), axis=0)
        total += len(segmented_hamming_plot[value])/9
    total = int(np.around(total))

    # plt.bar(XTest, total_plot_length, color="red", hatch="/")
    target_plot = np.full(9, total)  # uniformverteilung
    # plt.bar(XTest, target_plot, color="yellow")
    reduced_plot = np.where(total_plot_length < total,
                            total_plot_length, total)  # reduzierte traces
    # plt.bar(XTest, reduced_plot, color="blue")
    amount_to_add_plot = np.where(
        total_plot_length < total, total-total_plot_length, 0)  # Anzahlt der Hinzuzufügenden traces
    # plt.legend(["entfernte Traces", "hinzugefügte Traces", "unveränderte Traces"])
    # plt.ylabel("Trace Anzahl")
    # plt.xlabel("Hamming Gewicht")

    additional_data_plot = list()
    for index in range(9):
        # additional_data_plot.append(create_data(total_plot_means[index],
        #                                        total_plot_std[index], amount_to_add_plot[index]))
        additional_data_plot.append(create_data_SMOTE(
            segmented_hamming_plot[index], amount_to_add_plot[index]))

    plt.clf()
    XTest_700 = np.arange(700)

    plt.plot(XTest_700, additional_data_plot[0]
             [0], color="red", label="generierte Traces")
    plt.plot(XTest_700, additional_data_plot[0][1], color="red")
    plt.plot(XTest_700, additional_data_plot[0][2], color="red")

    plt.plot(
        XTest_700, segmented_hamming_plot[0][0], color="blue", label="originale Traces")
    plt.plot(XTest_700, segmented_hamming_plot[0][1], color="blue")
    plt.plot(XTest_700, segmented_hamming_plot[0][2], color="blue")

    plt.xlabel("Zeitabschnitt im Trace")
    plt.xlim([455, 475])
    plt.ylim([-5, 20])
    plt.legend()
    # plt.show()

    plt.clf()

    isSet = False
    for value in segmented_hamming_plot:
        if isSet == False:
            isSet = True
            final_data_plot = np.copy(np.asarray(
                segmented_hamming_plot[value]))[:total]
            final_data_plot_y = np.full(
                len(segmented_hamming_plot[value]), value)[:total]
        else:
            final_data_plot = np.concatenate((
                final_data_plot, np.asarray(segmented_hamming_plot[value][:total])))
            final_data_plot_y = np.concatenate((
                final_data_plot_y, np.full(len(segmented_hamming_plot[value]), value)[:total]))

        if len(additional_data_plot[value]) != 0:
            final_data_plot = np.concatenate(
                (final_data_plot, additional_data_plot[value]))
            final_data_plot_y = np.concatenate((
                final_data_plot_y, np.full(len(additional_data_plot[value]), value)[:total]))
    return (final_data_plot, final_data_plot_y)

################


ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
ascad = ASCAD_test_models.load_ascad(ascad_database, load_metadata=True)
(X_profiling, Y_profiling), (X_attack,
                             Y_attack), (Metadata_profiling,
                                         Metadata_attack) = ascad
x_data = np.copy(X_profiling)
y_data = np.copy(Y_profiling)
meta_data = np.copy(Metadata_profiling)
step = 1/10

# for index, value in enumerate(y_data):
#    y_data[index] = bin(value).count("1")

networkType = ["cnn", "mlp"]
# networkType = ["cnn2"]
for network in networkType:
    ranklist = list()
    scorelist = list()
    for index, value in enumerate(np.arange(0, 1, step)):
        print("step "+str(index+1)+" of "+str(int(1/step)))

        indexStart = int(np.around(len(x_data)*value))
        indexEnd = int(np.around(len(x_data)*(value+step)))
        X_profiling = np.concatenate((x_data[:indexStart], x_data[indexEnd:]))
        X_attack = x_data[indexStart:indexEnd]

        Y_profiling = np.concatenate((y_data[:indexStart], y_data[indexEnd:]))
        Y_attack = y_data[indexStart:indexEnd]

        print("step" + str(index)+": " + str(Y_attack[0]))
        print("step" + str(index)+": " + str(y_data[indexStart]))

        Metadata_profiling = np.concatenate(
            (meta_data[:indexStart], meta_data[indexEnd:]))
        Metadata_attack = meta_data[indexStart:indexEnd]

        ascad1 = (X_profiling, Y_profiling), (X_attack,
                                              Y_attack), (Metadata_profiling,
                                                          Metadata_attack)

        testing = False
        if testing == True:
            final_data_plot = X_profiling
            final_data_plot_y = Y_profiling
        else:
            (final_data_plot, final_data_plot_y) = add_data(ascad1)

        ascad2 = (final_data_plot, final_data_plot_y), (X_attack,
                                                        Y_attack), (Metadata_profiling,
                                                                    Metadata_attack)

        training_model_upper_dir = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/crossval/"
        save_file1 = "results17/SMOTE/sync/"

        attempt = "SMOTE"
        training_model_name = str(network) + "-" + \
            str(attempt)+"-"+str(index)+".h5"

        save_file1 = "results17/"+str(attempt)+"/sync/"

        training_model_dir = training_model_upper_dir + \
            "/"+attempt+"/"+network+"/"+training_model_name
        print(training_model_dir)

        hamming = True

        if testing != True:
            if "mlp" == network:
                ASCAD_train_models.main(ascad_loaded=True, ascad_data=ascad2, hamming_ready=True, ascad_database=ascad_database,
                                        training_model=training_model_dir, network_type=network, hammingWeight=True,  epochs=400, batch_size=100)
            else:
                ASCAD_train_models.main(ascad_loaded=True, ascad_data=ascad2, hamming_ready=True, ascad_database=ascad_database,
                                        training_model=training_model_dir, network_type=network, hammingWeight=True)
        else:
            save_file = save_file1+network+"/"
            (rank, score) = ASCAD_test_models.main(ascad_loaded=True, ascad_data=ascad2, save_file=save_file+network + "-"
                                                   + attempt+"_"+str(value*10), hammingWeight=True, model_file=training_model_dir, ascad_database=ascad_database)
            ranklist.append(rank)
            scorelist.append(score)

    if testing == True:
        hamming = hamming
        ranklist = np.asarray(ranklist)
        scorelist = np.asarray(scorelist)

        rankFinalMean = np.mean(ranklist, axis=0)
        rankFinalStd = np.std(ranklist, axis=0)

        x_test = np.arange(0, len(rankFinalMean)*25, 25)
        stdDevplus = rankFinalMean + rankFinalStd
        stdDevplus = np.where(stdDevplus > 256, 256, stdDevplus)
        stdDevminus = rankFinalMean - rankFinalStd
        stdDevminus = np.where(stdDevminus < 0, 0, stdDevminus)

        plt.clf()
        plt.fill_between(x_test, stdDevminus, stdDevplus)
        plt.plot(x_test, rankFinalMean, color="r")
        plt.xlim([0.0, 2000])
        plt.ylim([0.0, 200])
        plt.ylabel("rank")
        plt.xlabel("traces")
        if hamming == True:
            plt.savefig(save_file1+"/"+network+"/" +
                        network+"-mean_rank_hamming")
        else:
            plt.savefig(save_file1+"/"+network+"/"+network+"-mean_rank_ident")
        # plt.show()
        plt.clf()

        scorelist = np.asarray(scorelist)

        scoreFinalMean = np.mean(scorelist, axis=0)
        scoreFinalStd = np.std(scorelist, axis=0)

        x_test = np.arange(0, len(scoreFinalMean))
        stdDevplus = scoreFinalMean + scoreFinalStd
        stdDevminus = scoreFinalMean - scoreFinalStd

        plt.clf()
        plt.fill_between(x_test, stdDevminus, stdDevplus)
        plt.plot(x_test, scoreFinalMean, color="r")
        plt.ylabel("Schlüssel Score")
        plt.xlabel("Schlüssel Kanidaten")
        if hamming == True:
            plt.savefig(save_file1+"/"+network+"/" +
                        network+"-mean_score_hamming")
        else:
            plt.savefig(save_file1+"/"+network+"/"+network+"-mean_score_ident")
        # plt.show()
        plt.clf()
        np.savetxt("means/"+attempt+"-"+network+"-mean_rank.csv",
                   rankFinalMean, delimiter=',')
print("test")
