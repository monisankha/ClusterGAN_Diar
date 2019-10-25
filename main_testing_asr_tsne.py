import os, glob
import tensorflow as tf
import numpy
import numpy as np
import scipy.signal
import argparse
import importlib
import util
import time
import pickle
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.cluster import KMeans
from make_rttm import main
from sklearn.manifold import SpectralEmbedding

from spectral_clustering import predict

def recon_enc(timestamp, sampler, z_dim, beta_cycle_label, beta_cycle_gen, data, batch_size):

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

    return latent


if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--pathname', type=str, default='/Data/Monisankha/Data/moni_code/Moni_AMI_All/ClusterGAN/AMI_16k/AMI_dev_asr_16k/')
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

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 15

    for i in range(10, 11):

        path = '/Data/Monisankha/Data/moni_code/Moni_AMI_All/ClusterGAN/Timestamp/'
        itr = str(i + 1)
        pathname1 = pathname + 'data' + itr + '/'
        data = np.load(pathname1 + '/data.npy')
        timestamp = np.load(path + 'timestamp_asr4.npy')
        timestamp1 = timestamp[5]
        no_of_spk1 = int(np.load(pathname1 + '/no_of_spk.npy'))
        xs = data
        zs = util.sample_Z
        act_spk_lbl = np.load(pathname1 + '/spk_lbl.npy')

        ##########################################
        ## t-SNE plot for x-vector embedding
        ##########################################

        time_start = time.time()
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(data)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_pca_results = tsne.fit_transform(pca_result_50)

        feat_cols = ['pixel' + str(i) for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=feat_cols)
        df['y'] = act_spk_lbl
        df['label'] = df['y'].apply(lambda i: str(i))

        df_subset = df.copy()

        df_subset['x-vec-one'] = tsne_pca_results[:, 0]
        df_subset['x-vec-two'] = tsne_pca_results[:, 1]

        #plt.figure(figsize=(5, 5))
        #plt.rcParams.update({'font.size': 22})

        plt.rc('font', size=BIGGER_SIZE)
        plt.rc('axes', labelsize=BIGGER_SIZE)
        plt.rc('xtick', labelsize=BIGGER_SIZE)
        plt.rc('ytick', labelsize=BIGGER_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)

        fig = plt.figure(figsize=(14, 3))
        ax1 = plt.subplot(1, 3, 1)
        #ax1.tick_params(labelsize=14)

        #ax1.tick_params(axis="x", labelsize=14)
        #ax1.tick_params(axis="y", labelsize=14)
        #plt.rcParams['xtick.labelsize'] = 8
        sns.scatterplot(
            x="x-vec-one", y="x-vec-two",
            hue="y",
            #palette=sns.color_palette(sns.hls_palette(8, l=0.5, s=0.5), no_of_spk1),
            palette=sns.color_palette("hls", no_of_spk1),
            data=df_subset,
            alpha=0.3, linewidths=10,
            ax = ax1
        )
        #plt.savefig(pathname1 + 'data_xvec' + itr + '.png')

        ##########################################
        ## t-SNE plot for proposed embedding
        ##########################################
        latent = recon_enc(timestamp1, sampler, z_dim, beta_cycle_label, beta_cycle_gen, data, batch_size)  # model load and prediction

        time_start = time.time()
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(latent)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_pca_results = tsne.fit_transform(pca_result_50)


        feat_cols = ['pixel' + str(i) for i in range(latent.shape[1])]
        df = pd.DataFrame(latent, columns=feat_cols)
        df['y'] = act_spk_lbl
        df['label'] = df['y'].apply(lambda i: str(i))

        df_subset = df.copy()

        df_subset['proposed-one'] = tsne_pca_results[:, 0]
        df_subset['proposed-two'] = tsne_pca_results[:, 1]

        #plt.figure(figsize=(5, 5))
        ax2 = plt.subplot(1, 3, 2)
        #ax2.tick_params(labelsize=14)
        sns.scatterplot(
            #x="proposed-one", y="proposed-two",
            hue="y",
            #palette=sns.color_palette(sns.hls_palette(8, l=0.5, s=0.5), no_of_spk1),
            palette=sns.color_palette("hls", no_of_spk1),
            data=df_subset,
            alpha=0.3, linewidths=10,
            ax=ax2
        )
        #plt.savefig(pathname1 + 'data_propos' + itr + '.png')

        ##########################################
        ## t-SNE plot for fused embedding
        ##########################################

        data1 = np.concatenate((0.2*data, 0.8*latent), axis=1)

        time_start = time.time()
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(data1)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_pca_results = tsne.fit_transform(pca_result_50)

        feat_cols = ['pixel' + str(i) for i in range(data1.shape[1])]
        df = pd.DataFrame(data1, columns=feat_cols)
        df['y'] = act_spk_lbl
        df['label'] = df['y'].apply(lambda i: str(i))

        df_subset = df.copy()

        df_subset['fused-one'] = tsne_pca_results[:, 0]
        df_subset['fused-two'] = tsne_pca_results[:, 1]

        #plt.figure(figsize=(5, 5))
        ax3 = plt.subplot(1, 3, 3)
        ax3.tick_params(labelsize=14)
        sns.scatterplot(
            #x="fused-one", y="fused-two",
            hue="y",
            #palette=sns.color_palette(sns.hls_palette(8, l=0.5, s=0.5), no_of_spk1),
            palette=sns.color_palette("hls", no_of_spk1),
            data=df_subset,
            alpha=0.3, linewidths=10,
            ax=ax3
        )
        #plt.savefig(pathname1 + 'data_fused' + itr + '.png')
        plt.savefig(pathname1 + 'data_asr4_6_new' + itr + '.png')
        pickle.dump(fig, open('figure.pickle', 'wb'))

    x = 7










