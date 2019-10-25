import os, glob
import tensorflow as tf
import numpy
import numpy as np
import scipy.signal
import argparse
import importlib
import util

from sklearn.cluster import KMeans
from make_rttm import main
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

import metric
from sklearn.manifold import SpectralEmbedding

from spectral_clustering import predict

def recon_enc(timestamp, sampler, z_dim, beta_cycle_label, beta_cycle_gen, data, no_of_spk, batch_size):

    sess = tf.Session()

    checkpoint_dir = 'checkpoint_dir/ami/{}_{}_z{}_cyc{}_gen{}'.format(timestamp, sampler,
                                                                   z_dim, beta_cycle_label,
                                                                   beta_cycle_gen)

    imported_meta = tf.train.import_meta_graph(checkpoint_dir + '/model.ckpt.meta')
    imported_meta.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    print('Restored model weights.')

    x = tf.get_collection("x")[0]
    z_infer_gen = tf.get_collection("z_infer_gen")[0]
    z_infer_label = tf.get_collection("z_infer_label")[0]

    # ==================
    # Testing
    # ==================

    data_recon = data

    num_pts_to_plot = data_recon.shape[0]
    recon_batch_size = batch_size
    latent = np.zeros(shape=(num_pts_to_plot, z_dim))

    # print('Data Shape = {}, Labels Shape = {}'.format(data_recon.shape, label_recon.shape))

    for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):
        if (b + 1) * recon_batch_size > num_pts_to_plot:
            pt_indx = np.arange(b * recon_batch_size, num_pts_to_plot)
        else:
            pt_indx = np.arange(b * recon_batch_size, (b + 1) * recon_batch_size)
        xtrue = data_recon[pt_indx, :]

        zhats_gen, zhats_label = sess.run([z_infer_gen, z_infer_label],
                                                 feed_dict={x: xtrue})

        latent[pt_indx, :] = np.concatenate((zhats_gen, zhats_label), axis=1)
        #latent[pt_indx, :] = zhats_gen

    #latent_rep = latent
    #km = KMeans(n_clusters=no_of_spk, random_state=0).fit(latent_rep)
    #pred_lbl = km.labels_
    #pred_lbl = predict(latent_rep, no_of_spk)

    return latent

def kaldi_format_for_rttm_der(pathname, filename, pred_lbl, iter):

    path = pathname + '/tmpkaldidir_boscclinic/xvectors/'

    pred_lbl = 1 + pred_lbl
    segments= path + '/segments'

    # Smoothing on the segment labels

    smoothing_win = numpy.arange(1, 2, 30)
    out1 = []
    for i in range(0, len(smoothing_win)):
        out = scipy.signal.medfilt(pred_lbl, smoothing_win[i])
        # out = 1 + out
        out1.append(out)

    vr2 = []
    for i in range(0, len(smoothing_win)):
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

    derpath = pathname + iter
    if not os.path.exists(derpath):
        os.makedirs(derpath)

    file = open(derpath + '/DER_boscclinic_fusion_asr4.txt', 'w')
    for row in range(len(vr2)):
        der = vr2[row]
        file.write("%s\n" % der)
    file.close()

def eval_cluster(pathname1, labels_pred, labels_true, no_of_spk, timestamp, z_dim, sampler, beta_cycle_label, beta_cycle_gen):

    purity = metric.compute_purity(labels_pred, labels_true)
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    print(' #Points = {}, K = {}, Purity = {},  NMI = {}, ARI = {},'.format(labels_pred.shape[0], no_of_spk, purity, nmi, ari))

    with open(pathname1 + '/Result.txt', 'a+') as f:
        f.write('{}, K = {}, z_dim = {}, beta_label = {}, beta_gen = {}, sampler = {}, Purity = {}, NMI = {}, ARI = {}\n'
                .format(timestamp, no_of_spk, z_dim, beta_cycle_label, beta_cycle_gen,
                        sampler, purity, nmi, ari))
        f.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--pathname', type=str, default='/Data/Monisankha/Data/moni_code/moni_xvector_embedding/Manoj_xvector/bosc_clinic/Data_bosc_clinic/')
    parser.add_argument('--data', type=str, default='ami_xvector')
    parser.add_argument('--model', type=str, default='clus_wgan_new')
    parser.add_argument('--sampler', type=str, default='one_hot')
    parser.add_argument('--dz', type=int, default=30)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--beta_n', type=float, default=2.0)
    parser.add_argument('--beta_c', type=float, default=10.0)

    args = parser.parse_args()

    pathname = args.pathname
    model = importlib.import_module(args.data + '.' + args.model)
    dim_gen = args.dz
    batch_size = args.bs
    beta_cycle_gen = args.beta_n
    beta_cycle_label = args.beta_c
    sampler = args.sampler

    n_cat = 1
    no_of_spk = 155
    z_dim = dim_gen + no_of_spk * n_cat
    d_net = model.Discriminator()
    g_net = model.Generator(z_dim=z_dim)
    enc_net = model.Encoder(z_dim=z_dim, dim_gen=dim_gen)

    # for i in range(0, 27):
    #
    #     path = '/Data/Monisankha/Data/moni_code/Moni_AMI_All/ClusterGAN/Timestamp/'
    #
    #     itr = str(i + 1)
    #     pathname1 = pathname + 'data' + itr + '/'
    #     data = np.load(pathname1 + '/data.npy')
    #
    #     timestamp = np.load(path + 'timestamp_asr4.npy')
    #
    #     for j in range(1, len(timestamp)):
    #         no_of_spk1 = int(np.load(pathname1 + '/no_of_spk.npy'))
    #         xs = data
    #         zs = util.sample_Z
    #         timestamp1 = timestamp[j]
    #
    #         latent = recon_enc(timestamp1, sampler, z_dim, beta_cycle_label, beta_cycle_gen, data, no_of_spk1,
    #                            batch_size)  # model load and prediction
    #
    #         data1 = np.concatenate((0.2 * data, 0.8 * latent), axis=1)
    #
    #         km = KMeans(n_clusters=no_of_spk1, init="k-means++", max_iter=300, random_state=0).fit(data1)
    #         pred_lbl = km.labels_
    #
    #         #pred_lbl = recon_enc(timestamp1, sampler, z_dim, beta_cycle_label, beta_cycle_gen, data, no_of_spk1, batch_size)  # model load and prediction
    #
    #     #lbl_true = np.load(pathname1 + '/spk_lbl.npy')
    #     #lbl_true = numpy.transpose(lbl_true)
    #     #lbl_true = lbl_true.reshape(np.shape(pred_lbl)[0])
    #
    #     #eval_cluster(pathname1, pred_lbl, lbl_true, no_of_spk1, timestamp, z_dim, sampler, beta_cycle_label, beta_cycle_gen)  # model load and prediction
    #
    #     #np.save(pathname1 + '/pred.npy', pred_lbl)
    #
    #         for filename in sorted(glob.glob(os.path.join(pathname1, '*.wav'))):  # The .wav file to diarize
    #             file_name = filename.split('/')[-1][:-4]
    #
    #         iter = str(j + 1)
    #
    #         kaldi_format_for_rttm_der(pathname1, file_name, pred_lbl, iter)

    der = []
    for i in range(0, 27):
        if i == 10 or i == 11 or i == 12:
            continue
        else:
            itr = str(i + 1)
            pathname1 = pathname + 'data' + itr + '/' + str(6) + '/'
            file = open(pathname1 + 'DER_boscclinic_fusion_asr_icsi3.txt', 'r')
            x = file.readlines()
            a = x[0].strip()
            der.append(a)

    der = np.asarray(der, dtype=float)
    der_avg = np.mean(der)
    der_std = np.std(der)











