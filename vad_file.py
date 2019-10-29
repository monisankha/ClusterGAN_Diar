import os, glob
import numpy as np
import decimal
import argparse
import scipy.signal

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def vad_file(wavFile, rttmFile):

    frame_rate = 100
    file = open(rttmFile, 'r')
    duration = os.popen('soxi -D ' + wavFile).readlines()[0][:-1]
    total_frame = float(duration) * float(frame_rate)
    x = file.readlines()
    l = len(x)  # No of lines in the rttm file
    vad = [0] * int(total_frame)

    for it in range(l):
        a = x[it].split(' ')  # First line read of rttm file
        f1 = int(round_half_up(float(a[3]) * frame_rate))   # Starting frame index
        f2 = int(round_half_up((float(a[3]) + float(a[4])) * frame_rate))  # Ending frame index
        vad[f1:f2] = [1] * (f2-f1)

    vad = np.asarray(vad)
    return vad

def data_prep_vad(wavFile, rttmFile):

    path = os.getcwd()

    logger = open(os.path.join(path, "wavList"), 'w')
    logger.write("{:s}\n".format(wavFile))
    logger.close()

    file_name = wavFile.split('/')[-1][:-4]

    vad = vad_file(wavFile, rttmFile)
    iter_path = path + '/vad/kaldiVAD/'

    if not os.path.exists(iter_path):
        os.makedirs(iter_path)

    logger1 = open(os.path.join(iter_path, file_name + ".csv"), 'w')
    for i1 in range(len(vad)):
        logger1.write("{:d}\n".format(vad[i1]))
    logger1.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--wavFile', type=str)
    parser.add_argument('--rttmFile', type=str)

    args = parser.parse_args()

    wavFile = args.wavFile
    rttmFile = args.rttmFile

    data_prep_vad(wavFile, rttmFile)






