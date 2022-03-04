# load traces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter

import ASCAD_train_models
import ASCAD_test_models


from ASCAD_train_models import load_ascad
matplotlib.use('TkAgg')


def norm(x, mu, std):
    return (1/(std * np.sqrt(2 * np.pi)) *
            np.exp(- (x - mu)**2 / (2 * std**2)))


ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
ascad = load_ascad(ascad_database, load_metadata=True)
(X_profiling, Y_profiling), (X_attack,
                             Y_attack), (Metadata_profiling,
                                         Metadata_attack) = ascad

ret = list()
for i in Y_profiling:
    # print(bin(i).count("1"))
    ret.append(bin(i).count("1"))


segmented_hamming_plot = {}
tmptmp = np.zeros(9)
for index, value in enumerate(X_profiling):
    indexindex = bin(Y_profiling[index]).count("1")
    if indexindex not in segmented_hamming_plot:
        segmented_hamming_plot[indexindex] = list()
    tmptmp[indexindex] += 1
    segmented_hamming_plot[indexindex].append(value)

targetLength = (
    len(segmented_hamming_plot[0])+len(segmented_hamming_plot[8]))/2

overload_length = np.zeros(9)
for key in segmented_hamming_plot:
    if key != 0 or key != 8:
        overload_length[key] = len(
            segmented_hamming_plot[key])-targetLength
    else:
        overload_length[key] = 0

tmp = np.asarray(ret)
tmpMean = np.mean(tmp)
tmpStd = np.std(tmp)
step = 1
XTest = np.arange(0, 9, step)
#plt.xlim([-1, 9])
# plt.plot(XTest, norm(XTest, tmpMean, tmpStd)*len(tmp),
# color='r')
length_final = list()
for i in np.arange(1, 0, -0.1):
    tmptmptmp = tmptmp-(overload_length*i)
    length_final.append(tmptmptmp)
    if i*10 % 2 == 0:
        plt.plot(XTest, tmptmptmp,  # align='left',
                 #bins=np.arange(0, 10, 1),
                 color='r')
    else:
        plt.plot(XTest, tmptmptmp,  # align='left',
                 #bins=np.arange(0, 10, 1),
                 color='b')
# plt.show()


networkType = ["cnn", "cnn2", "mlp"]


for network in networkType:
    print(network)

training_model1 = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/reduced_stepped/sync/"
ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
save_file1 = "results20/reduced/sync/"
testing = True

for index, length in enumerate(length_final):
    for network in networkType:
        training_model = training_model1+network+"/"
        if testing != True:
            if "mlp" == network:
                ASCAD_train_models.main(ascad_database=ascad_database, training_model=training_model+network +
                                        "-hamming-reduced"+str((10-index)*10)+".h5", network_type=network, hammingWeight=True, save_file=save_file+network+"-Hamming-reduced"+str((10-index)*10), length=length, epochs=400, batch_size=100)
            else:
                ASCAD_train_models.main(ascad_database=ascad_database, training_model=training_model+network +
                                        "-hamming-reduced"+str((10-index)*10)+".h5", network_type=network, hammingWeight=True, save_file=save_file+network+"-Hamming-reduced"+str((10-index)*10), length=length)
        else:
            import ASCAD_train_models
            save_file = save_file1+network+"/"
            ASCAD_test_models.main(save_file=save_file+network +
                                   "-hamming-reduced"+str((10-index)*10)+".h5", hammingWeight=True, model_file=training_model+network +
                                   "-hamming-reduced"+str((10-index)*10)+".h5", ascad_database=ascad_database)

print(norm(0, tmpMean, tmpStd)*50000)
print(np.sum(norm(XTest, tmpMean, tmpStd)*step*50000))
