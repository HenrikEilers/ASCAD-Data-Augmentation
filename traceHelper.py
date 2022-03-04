# load traces
import numpy as np
from ASCAD_train_models import load_ascad
import matplotlib.pyplot as plt
import matplotlib
import tkinter
from tracehelper.s_box_test import s_box_test
from tracehelper.mean_std_SNR import mean_std


matplotlib.use('TkAgg')


ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_variable_key/ASCAD_data/ASCAD_databases/ascad-variable.h5"
ascad = load_ascad(ascad_database, load_metadata=True)
(X_profiling, Y_profiling), (X_attack,
                             Y_attack), (Metadata_profiling,
                                         Metadata_attack) = ascad

s_box_test(ascad)
mean_std(ascad)

testX = np.arange(1400)


# Ident mean varinat2
plotArrayHamming = {}
indexArrayHamming = [0]*256
cmap = plt.cm.get_cmap("hsv", 9)

plt.clf()
#plt.figure().set_figwidth(50, forward=True)

for index, value in enumerate(X_profiling):
    indexindex = Y_profiling[index]
    if indexindex not in plotArrayHamming:
        plotArrayHamming[indexindex] = list()

    plotArrayHamming[indexindex].append(value)
    indexArrayHamming[indexindex] += 1


#indexArrayHamming = np.split(np.repeat([indexArrayHamming], 1400), 9)

retVal = list()
#plotArrayHamming = plotArrayHamming/indexArrayHamming
for value in plotArrayHamming:
    value = np.asarray(plotArrayHamming[value])
    value = np.mean(value, axis=0)
    retVal.append(value)
    plt.plot(testX, value)

retretVal = np.zeros(1400)
for i in range(1400):
    tmpVAl = 0
    for k1 in retVal:
        for k2 in retVal:
            if abs(k1[i] - k2[i]) > tmpVAl:
                tmpVAl = abs(k1[i] - k2[i])
    retretVal[i] = tmpVAl
plt.plot(testX, retretVal*150)


plt.savefig("figure/mean_ident_var2.png")
# plt.show()


# Indent Mean
plotArrayIdent = np.zeros((256, 1400))
indexArrayIdent = [0]*256
cmap = plt.cm.get_cmap("hsv", 256)


plt.clf()
#plt.figure().set_figwidth(50, forward=True)

for index, value in enumerate(X_profiling):
    indexindex = Y_profiling[index]
    plotArrayIdent[indexindex] += value
    indexArrayIdent[indexindex] += 1

indexArrayIdent = np.split(np.repeat([indexArrayIdent], 1400), 256)

plotArrayIdent = plotArrayIdent/indexArrayIdent
for index, value in enumerate(plotArrayIdent):
    plt.plot(testX, value)

retArray = list()
for i in range(1400):
    retArray.append(0)
    for k1 in plotArrayIdent:
        for k2 in plotArrayIdent:
            retArray[i] += np.absolute(k1[i]-k2[i])/256

tmp = np.argmax(np.asarray(retArray))
print(tmp)

plt.axvline(tmp, color='r')

plt.savefig("figure/mean_ident.png")
plt.show()


# Hamming mean
plotArrayHamming = np.zeros((9, 1400))
indexArrayHamming = [0]*9
cmap = plt.cm.get_cmap("hsv", 9)

plt.clf()
#plt.figure().set_figwidth(50, forward=True)

for index, value in enumerate(X_profiling):
    indexindex = bin(Y_profiling[index]).count("1")
    plotArrayHamming[indexindex] += value
    indexArrayHamming[indexindex] += 1

indexArrayHamming = np.split(np.repeat([indexArrayHamming], 1400), 9)

plotArrayHamming = plotArrayHamming/indexArrayHamming
for index, value in enumerate(plotArrayHamming):
    plt.plot(testX, value)

plt.savefig("figure/mean_hamming.png")


# std Ident
plotArrayIdentStd = np.zeros((256, 1400))
indexArrayIdentStd = [0]*256
cmap = plt.cm.get_cmap("hsv", 256)

plt.clf()
#plt.figure().set_figwidth(50, forward=True)

for index, value in enumerate(X_profiling):
    indexindex = Y_profiling[index]
    plotArrayIdentStd[indexindex] += np.absolute(
        value-plotArrayIdent[indexindex])
    indexArrayIdentStd[indexindex] += 1

indexArrayIdentStd = np.split(np.repeat([indexArrayIdentStd], 1400), 256)

plotArrayIdentStd = plotArrayIdentStd/indexArrayIdentStd
for index, value in enumerate(plotArrayIdentStd):
    plt.plot(testX, value)

plt.savefig("figure/ident_std.png")


# std Hamming
plotArrayHammingStd = np.zeros((9, 1400))
indexArrayHammingStd = [0]*9
cmap = plt.cm.get_cmap("hsv", 9)

plt.clf()
#plt.figure().set_figwidth(50, forward=True)

for index, value in enumerate(X_profiling):
    indexindex = bin(Y_profiling[index]).count("1")
    plotArrayHammingStd[indexindex] += np.absolute(
        value-plotArrayHamming[indexindex])
    indexArrayHammingStd[indexindex] += 1

indexArrayHammingStd = np.split(np.repeat([indexArrayHammingStd], 1400), 9)

plotArrayHammingStd = plotArrayHammingStd/indexArrayHammingStd
for index, value in enumerate(plotArrayHammingStd):
    plt.plot(testX, value)

plt.savefig("figure/hamming_std.png")
