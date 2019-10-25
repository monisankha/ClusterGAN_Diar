import numpy
import tensorflow as tf
import argparse
import numpy as np
import os
import kaldi_io
import os, glob

pathname = '/Data/Monisankha/Data/moni_code/moni_xvector_embedding/Manoj_xvector/bosc_clinic/Data_bosc_clinic/'

# No of speaker of each test file

for i in range(0, 27):
    itr = str(i + 1)
    pathname1 = pathname + 'data' + itr + '/'
    for filename in sorted(glob.glob(os.path.join(pathname1, '*.rttm'))):  # The .rttm file that is to be diarized
        file = open(filename, 'r')
        x = file.readlines()
        list_spk = []
        for it in range(len(x)):
            a = x[it].split(' ')
            b = a[7]
            list_spk.append(b)
        uniq_spk = np.unique(list_spk)
        no_uniq_spk = len(uniq_spk)
    np.save(pathname1 + '/no_of_spk.npy', no_uniq_spk)

for i in range(0, 27):
    itr = str(i + 1)
    data = []  # np.array([])
    pathname1 = pathname + 'data' + itr + '/tmpkaldidir_boscclinic/xvectors/'
    file = open(pathname1 + '/xvector.scp', 'r')
    x = file.readlines()
    d = np.empty((len(x), 512))
    for i in range(0, len(x)):
        a = x[i]
        d[i, :] = kaldi_io.read_vec_flt(a.strip().split()[1])
    data.append(d)
    data = np.concatenate(data)
    np.save(pathname + 'data' + itr + '/data.npy', data)

x = 7