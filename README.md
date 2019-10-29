# ClusterGAN_Diar
Latent space clustering in GANs.

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

