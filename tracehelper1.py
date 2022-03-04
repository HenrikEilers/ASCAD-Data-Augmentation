# load traces
import numpy as np
from ASCAD_train_models import load_ascad
import matplotlib.pyplot as plt
import matplotlib
import tkinter
from tracehelper.s_box_test import s_box_test
from tracehelper.mean_std_SNR import mean_std
from tracehelper.snr_and_stuff import do_snr
from tracehelper.special_mean_std import mean_std_special


matplotlib.use('TkAgg')


ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
ascad = load_ascad(ascad_database, load_metadata=True)
(X_profiling, Y_profiling), (X_attack,
                             Y_attack), (Metadata_profiling,
                                         Metadata_attack) = ascad

for index, value in enumerate(Y_profiling):
    tmp = Metadata_profiling[index]["masks"]
    #Y_profiling[index] = value ^ Metadata_profiling[index]["masks"][15]
    #Y_profiling[index] = Metadata_profiling[index]["masks"][0]

ascad = (X_profiling, Y_profiling), (X_attack,
                                     Y_attack), (Metadata_profiling,
                                                 Metadata_attack)

s_box_test(ascad)
do_snr(ascad)
mean_std(ascad)
mean_std_special(ascad)
