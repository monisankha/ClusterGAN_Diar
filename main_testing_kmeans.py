import os, glob
import tensorflow as tf
import numpy
import numpy as np
import scipy.signal
import argparse
import importlib
import util

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from make_rttm import main

def kaldi_format_for_rttm_der(pathname, filename, pred_lbl):

    path = pathname + '/tmpkaldidir_ami/xvectors/'

    pred_lbl = 1 + pred_lbl
    segments= path + '/segments'

    # Smoothing on the segment labels

    smoothing_win = numpy.arange(1, 702, 30)
    out1 = []
    for i in range(0, len(smoothing_win)):
        out = scipy.signal.medfilt(pred_lbl, smoothing_win[i])
        uniq_lbl = np.unique(out)
        if len(uniq_lbl) == 1:
            continue
        else:
            out1.append(out)
        # out = 1 + out

    vr2 = []
    for i in range(0, len(out1)):
        pred_lbl = out1[i]
        file = open(segments, 'r')
        x = file.readlines()
        file1 = open(path + '/labels', 'w')
        for j in range(len(x)):
            a = x[j].split(' ')
            b = a[0]
            lbl = str(int(pred_lbl[j]))
            file1.write("{} {}\n".format(b, lbl))
        file1.close()

        labels = path + '/labels'

        rttmfolder = path + str(i + 1) + '/'
        if not os.path.exists(rttmfolder):
            os.makedirs(rttmfolder)

        rttmpath = rttmfolder + '/predict.rttm'

        rttm_channel = 0

        # RTTM file generation

        main(segments, labels, rttmpath, rttm_channel)

        # DER calculation

        vr = os.popen('perl md-eval.pl -1 -c 0.25 -r ' + pathname + filename + '.rttm -s ' + rttmpath + ' 2>&1| grep "OVERALL" | cut -f 2 -d "=" || echo "Failure during eval"').readlines()
        vr1 = vr[0][:6]
        vr2.append(vr1)

    # Save DER

    derpath = pathname
    if not os.path.exists(derpath):
        os.makedirs(derpath)

    file = open(derpath + '/DER_kmeans_no_of_spk_simple.txt', 'w')
    for row in range(len(vr2)):
        der = vr2[row]
        file.write("%s\n" % der)
    file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--pathname', type=str, default='/Data/Monisankha/Data/moni_code/Moni_AMI_All/ClusterGAN/AMI_16k/AMI_dev_asr_16k/')

    args = parser.parse_args()

    pathname = args.pathname

    # for i in range(0, 14):
    #
    #     itr = str(i + 1)
    #     pathname1 = pathname + 'data' + itr + '/'
    #     data = np.load(pathname1 + '/data.npy')
    #
    #     #no_of_spk = int(np.load(pathname1 + '/no_of_spk.npy'))
    #     file = open(pathname1 + '/no_of_spk_sc_simple.txt', 'r')
    #     no_of_spk = int(file.readlines()[0])
    #
    #     km = KMeans(n_clusters=no_of_spk, random_state=0).fit(data)
    #     pred_lbl = km.labels_
    #
    #     for filename in sorted(glob.glob(os.path.join(pathname1, '*.wav'))):  # The .wav file to diarize
    #      file_name = filename.split('/')[-1][:-4]
    #
    #     kaldi_format_for_rttm_der(pathname1, file_name, pred_lbl)

    der = []
    for i in range(0, 14):
        itr = str(i + 1)
        pathname1 = pathname + 'data' + itr + '/'
        file = open(pathname1 + 'DER_kmeans_no_of_spk.txt', 'r')
        x = file.readlines()
        l = len(x)
        a = x[0].strip()
        der.append(a)

    der = np.asarray(der, dtype=float)
    der_avg = np.mean(der)
    der_std = np.std(der)









