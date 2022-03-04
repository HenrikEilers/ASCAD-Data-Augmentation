import ASCAD_train_models
import ASCAD_test_models

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter

from tracehelper.special_mean_std import mean_std_special

matplotlib.use('TkAgg')

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

networkType = ["cnn",  "mlp"]
#networkType = ["cnn2"]
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

        ascad2 = (X_profiling, Y_profiling), (X_attack,
                                              Y_attack), (Metadata_profiling,
                                                          Metadata_attack)

        training_model_upper_dir = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/crossval/"
        save_file1 = "results17/ident/sync/"
        testing = True
        attempt = "ident"
        training_model_name = str(network) + "-" + \
            str(attempt)+"-"+str(index)+".h5"

        save_file1 = "results17/"+str(attempt)+"/sync/"

        training_model_dir = training_model_upper_dir + \
            "/"+attempt+"/"+network+"/"+training_model_name
        print(training_model_dir)

        hamming = False
        if testing != True:
            if "mlp" == network:
                ASCAD_train_models.main(ascad_loaded=True, ascad_data=ascad2, hamming_ready=False, ascad_database=ascad_database,
                                        training_model=training_model_dir, network_type=network, hammingWeight=hamming,  epochs=400, batch_size=100)
            else:
                ASCAD_train_models.main(ascad_loaded=True, ascad_data=ascad2, hamming_ready=False, ascad_database=ascad_database,
                                        training_model=training_model_dir, network_type=network, hammingWeight=hamming)
        else:
            save_file = save_file1+network+"/"
            (rank, score) = ASCAD_test_models.main(ascad_loaded=True, ascad_data=ascad2, save_file=save_file+network +
                                                   "-hamming_"+str(value*10), hammingWeight=hamming, model_file=training_model_dir, ascad_database=ascad_database)
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
