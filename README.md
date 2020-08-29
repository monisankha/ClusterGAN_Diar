# ClusterGAN_Diar
In this work, we propose deep latent space clustering for speaker diarization using generative adversarial network (GAN) back-projection with the help of an encoder network. The proposed diarization system is trained jointly with GAN loss, latent variable recovery loss, and a clustering-specific loss. It uses x-vector speaker embeddings at the input, while the latent variables are sampled from a combination of continuous random variables and discrete one-hot encoded variables using the original speaker labels. This repository only contains the inference part based on the available pre-trained model.

# Paper published
Pal, Monisankha, et al. ["Speaker diarization using latent space clustering in generative adversarial network."](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053952) ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.

# Dependencies
The code has been tested with the following versions of packages.

Python 2.7.12; Tensorflow 1.4.0; Numpy 1.16.4; Scikit-learn 0.20.3

# Datasets
AMI-dev, eval; ADOS-mod3, BOSCC-clinic

# Timestamp
Timestamp.npy contains the time-index when the trained model was saved.
timestamp_asr_icsi3->Trained on AMI-train + ICSI, timestamp_asr4->Trained on AMI-train

# Testing using pre-trained models
Pre-trained models are saved in checkpoint_dir.

main_testing.py  <--wavFile (Path of .wav)>  <--rttmFile (Path of .rttm)>  <--kaldidir (Kaldi directory)>  <--no_of_spk (No. of. spk in the wavefile, oracle)>  <--rttm_output (Path where predicted rttm to be saved)>  <--der_output (Path where output DER to be saved)>

# Example
python main_testing.py --wavFile /Data/git_dir_final/ClusterGAN_Diar/AMI_ES2004c.wav --rttmFile /Data/git_dir_final/ClusterGAN_Diar/AMI_ES2004c.rttm --kaldidir /home/moni/kaldi/ --no_of_spk 4 --rttm_output /Data/git_dir_final/ClusterGAN_Diar/rttm/ --der_output /Data/git_dir_final/ClusterGAN_Diar/der/

# Diarizaion Performance
DER of the x-vector baseline on AMI_ES2004c.wav: 20.52%; 
DER of the fused system (x-vector + ClusterGAN embedding) on AMI_ES2004c.wav: 1.68% (using timestamp_asr_icsi3, 30k model)

*** For any query pls contact at monisankha.pal@gmail.com

