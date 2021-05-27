#block(name=job_fast_gan_hca, threads=4, memory=25000, subtasks=1, gpus=1, hours=7)
	echo $CUDA_VISIBLE_DEVICES
    source /home/s7hialtu/anaconda3/etc/profile.d/conda.sh
	conda activate fast-gan
	python3 train.py --path /scratch/s7hialtu/fast_gan/fast_gan_cars --im_size 256 --start_iter 15000 --name potsdam_cars --ckpt /scratch/s7hialtu/fast_gan/train_results/potsdam_cars/models/all_15000.pth 
