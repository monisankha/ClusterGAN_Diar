import os, glob
import numpy as np
import decimal
import argparse
import scipy.signal

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def vad_file(pathname, wavfilename):

    frame_rate = 100
    file = open(pathname + wavfilename + '.rttm', 'r')
    duration = os.popen('soxi -D ' + pathname + wavfilename + '.wav').readlines()[0][:-1]
    total_frame = float(duration) * float(frame_rate)
    x = file.readlines()
    l = len(x)  # No of lines in the rttm file
    vad = [0] * int(total_frame)

    #a = x[0].split(' ')
    #f2 = float(0)
    #a = x[l-1].split(' ')
    #f3 = float(a[3]) + float(a[4])
    #input_file = pathname + wavfilename + '.wav'
    #output_file = pathname + 'mod/' + wavfilename + '.wav'
    #b = 'sox ' + input_file + ' ' + output_file + ' trim ' + str(f2) + ' ' + str(f3)
    #os.system('sox ' + input_file + ' ' + output_file + ' trim ' + str(f2) + ' ' + str(f3))

    for it in range(l):
        a = x[it].split(' ')  # First line read of rttm file
        f1 = int(round_half_up(float(a[3]) * frame_rate))   # Starting frame index
        f2 = int(round_half_up((float(a[3]) + float(a[4])) * frame_rate))  # Ending frame index
        vad[f1:f2] = [1] * (f2-f1)

    vad = np.asarray(vad)
    return vad

def data_prep_boscclinic_testing(pathname, it):

    path = pathname + '/data' + str(it) + '/'

    print(path)

    filename_list = []
    wavfile_list = []
    for filename in sorted(glob.glob(os.path.join(path, '*.wav'))):  # The .wav file to diarize
        wavfile_list.append(filename)
        file_name1 = filename.split('/')[-1][:-4]
        filename_list.append(file_name1)

    filename_list = np.asarray(filename_list)
    wavfile_list = np.asarray(wavfile_list)

    os.makedirs(path, exist_ok=True)

    logger = open(os.path.join(path, "wavList_boscclinic"), 'w')

    for i in range(len(wavfile_list)):
        wavefile = filename_list[i]
        wavefile1 = wavfile_list[i]
        print(wavefile1)
        vad = vad_file(path, wavefile)
        iter_path = path + '/vad_boscclinic/kaldiVAD/'
        os.makedirs(iter_path, exist_ok=True)
        logger1 = open(os.path.join(iter_path, wavefile + ".csv"), 'w')
        for i1 in range(len(vad)):
            logger1.write("{:d}\n".format(vad[i1]))
        logger1.close()
        logger.write("{:s}\n".format(wavefile1))

    logger.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--pathname', type=str)
    parser.add_argument('--data', type=str)

    args = parser.parse_args()

    pathname = args.pathname
    it = args.data

    data_prep_boscclinic_testing(pathname, it)



