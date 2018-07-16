#!/bin/bash
echo `date`: This job is running on ${HOSTNAME}
UsingGPU=$(awk '/^UsingGPU/{print $3}' test_config.cfg)
scale=$(awk '/^scale/{print $3}' test_config.cfg)
test_data_path=$(awk '/^test_data_path/{print $3}' test_config.cfg)
checkpoint_path=$(awk '/^checkpoint_path/{print $3}' test_config.cfg)
epoch_num=$(awk '/^epoch_num/{print $3}' test_config.cfg)
image_name=$(awk '/^image_name/{print $3}' test_config.cfg)

image_save_path=$(awk '/^image_save_path/{print $3}' test_config.cfg)
psnr_save_path=$(awk '/^psnr_save_path/{print $3}' test_config.cfg)

if [ $UsingGPU -eq 1 ]
	then
	# Create links and directories as needed by the actual job.
	export PATH=/usr/local/cuda-8.0/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
	export CPATH=/usr/local/include
	export CUDA_HOME=/usr/local/cuda-8.0
	nvidia-smi	

	source /tools/tensorflow/ENV_TF_GPU/bin/activate	

	#cd /scratch/nianlong	

	/tools/tensorflow/ENV_TF_GPU/bin/python3 test.py --scale $scale --test_path $test_data_path --ckpt_path $checkpoint_path --epoch_num $epoch_num --image_name $image_name --image_save_path $image_save_path --psnr_save_path $psnr_save_path	
	
	

	deactivate	

	# #qsub -l h_vmem=2500M -q MegaKnechte -pe openmp 3-3 run.sh 2 data/train/291/SHEVCscale2/ None

else

	source /tools/tensorflow/ENV_TF_CPU/bin/activate

	/tools/tensorflow/ENV_TF_CPU/bin/python3 test.py --scale $scale --test_path $test_data_path --ckpt_path $checkpoint_path --epoch_num $epoch_num --image_name $image_name --image_save_path $image_save_path --psnr_save_path $psnr_save_path
	
	deactivate

fi