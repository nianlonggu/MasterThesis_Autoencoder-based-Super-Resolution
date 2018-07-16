#!/bin/bash
echo `date`: This job is running on ${HOSTNAME}

UsingGPU=$(awk '/^UsingGPU/{print $3}' train_config.cfg)
scale=$(awk '/^scale/{print $3}' train_config.cfg)
training_data_path=$(awk '/^training_data_path/{print $3}' train_config.cfg)
checkpoint_path=$(awk '/^checkpoint_path/{print $3}' train_config.cfg)


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

	/tools/tensorflow/ENV_TF_GPU/bin/python3 SR.py  --scale $scale --train_data_path $training_data_path --ckpt_path $checkpoint_path	

	deactivate	

	# #qsub -l h_vmem=2500M -q MegaKnechte -pe openmp 3-3 run.sh 2 data/train/291/SHEVCscale2/ None

else
	source /tools/tensorflow/ENV_TF_CPU/bin/activate

	/tools/tensorflow/ENV_TF_CPU/bin/python3 SR.py  --scale $scale --train_data_path $training_data_path --ckpt_path $checkpoint_path	

	deactivate
fi