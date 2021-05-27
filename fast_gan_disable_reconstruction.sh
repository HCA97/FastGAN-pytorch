#block(name=gan_disable_Rloss_hca, threads=4, memory=16000, subtasks=1, gpus=1, hours=14)
	echo $CUDA_VISIBLE_DEVICES
    source /home/s7hialtu/anaconda3/etc/profile.d/conda.sh
	conda activate fast-gan
	python3 train.py --path /scratch/s7hialtu/fast_gan/fast_gan_cars --im_size 256 --name potsdam_cars_disable_reconstruction --disable_reconstruction --update_tensorboard 525 

