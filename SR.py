import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
from sr_utils import *
import time
from model import *
import os, glob, re, signal, sys, argparse, threading, time
import _thread



parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path")
parser.add_argument("--scale")
parser.add_argument("--train_data_path")
args = parser.parse_args()
ckpt_path = args.ckpt_path
SCALE=int(args.scale)
TRAIN_DATA_PATH = args.train_data_path

HR_IMG_SIZE=60
LR_IMG_SIZE=int(HR_IMG_SIZE/SCALE)
MINI_BATCH_SIZE = 64
START_LR = 0.0003
DECAY_RATE = 0.3
DECAY_STEP_SIZE = 50
MAX_EPOCH = 80




def create_paceholders():
    
    Y=tf.placeholder(tf.float32, [None, HR_IMG_SIZE, HR_IMG_SIZE, 1 ] )
    X_lr=tf.placeholder( tf.float32, [ None, LR_IMG_SIZE, LR_IMG_SIZE, 1 ] )
    return Y, X_lr


def compute_cost( Y_hat, Y  ):

    # cost= tf.reduce_sum( tf.nn.l2_loss(tf.subtract( Y_hat,Y )  )  )
    cost= tf.reduce_sum(tf.squared_difference(Y_hat, Y  )  )
    return cost




if __name__ == '__main__':

	#train_list contrains a list of <input slice, ref_slice> pairs, len(train_list) means the total number of examples
	print('start loding file')
	train_list=get_train_list(TRAIN_DATA_PATH,  scale=SCALE )
		
	print('loading finished, length=', len(train_list))

	# Y means the raw image
	Y, X_lr= create_paceholders()


	# shared_model = tf.make_template('shared_model', sr_model)
	# Y_hat, parameters=shared_model(X , SCALE )
	Y_hat, Y_downsampled = sr_model(  Y, X_lr )

	cost=compute_cost(Y_hat,Y)

	#store mse cost into tensorboard so as to monitor
	cost_mse= cost *(255**2) / MINI_BATCH_SIZE/ (HR_IMG_SIZE**2)
	tf.summary.scalar("cost", cost_mse)

	lr_cost = compute_cost( Y_downsampled, X_lr)

	## change the form of loss function, to make the downsampled image not far from the SHEVC downsampled image.
	alpha =0.8

	loss = alpha * cost + (1 - alpha)*lr_cost


	"""
	learning rate = START_LR * (DECAY_RATE)**( global_step*MINI_BATCH_SIZE / len(train_list)*DECAY_STEP_SIZE  )
	in one epoch the step number = len(train_list)/MINI_BATCH_SIZE
	so this is equivalent to:
		per DECAY_STEP_SIZE epochs, the learning rate decay by DECAY_RATE 
	"""
	global_step 	= tf.Variable(0, trainable=False)
	learning_rate 	= tf.train.exponential_decay(START_LR, global_step*MINI_BATCH_SIZE, len(train_list)*DECAY_STEP_SIZE, DECAY_RATE, staircase=True)
	tf.summary.scalar("learning_rate", learning_rate)
	#after running train each time, the global step will increase by one
	opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


	# save every thing that can be saved
	saver = tf.train.Saver(var_list=None, max_to_keep=0)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)   
	with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1, gpu_options=gpu_options)) as sess:

	# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

	# with tf.Session(config=session_conf) as sess:
		#TensorBoard open log with "tensorboard --logdir=logs"
		# log_save_path = '/tmp/logs/'+os.getcwd().split('/')[-1]+'/'+TRAIN_DATA_PATH.split('/')[-2]+'/'
		ckpt_save_path = 'checkpoints/'+TRAIN_DATA_PATH.split('/')[-2]+'/'

		# if not os.path.exists(log_save_path):
		# 	os.makedirs(log_save_path)

		if not os.path.exists(ckpt_save_path):
			os.makedirs(ckpt_save_path)

		#merge all the monitored scalars
		merged = tf.summary.merge_all()
		# file_writer = tf.summary.FileWriter(log_save_path, sess.graph)

		init=tf.global_variables_initializer()
		sess.run(init)


		start_epoch =0
		if ckpt_path:
			if ckpt_path != 'None':
				os.system("echo "+"restore model...")
				saver.restore(sess, ckpt_path)
				#"checkpoints/SR_epoch_001.ckpt"
				start_epoch = int( ckpt_path.split( '.ckpt' )[0].split('_')[-1] ) +1 
				os.system("echo "+"Done")

		for epoch in range(start_epoch, MAX_EPOCH):
			random.shuffle(train_list)
			max_step=len(train_list)//MINI_BATCH_SIZE
			for step in range(max_step):

				offset = step*MINI_BATCH_SIZE
				mini_batch_X, mini_batch_Y, mini_batch_upscaled_X = get_mini_batch(train_list, offset, MINI_BATCH_SIZE )
				feed_dict={  Y:mini_batch_Y, X_lr:mini_batch_X }
				_, mini_batch_cost, mini_c_mse, train_output, lr, g_step ,summary  =sess.run( [ opt, cost,cost_mse, Y_hat, learning_rate, global_step, merged ] , feed_dict=feed_dict )
				if step % 10 ==0:
					os.system("echo "+str(g_step))
					os.system("echo "+"[epoch %2.4f] cost %.4f\t learning rate %.5f"%(epoch+(float(step)*MINI_BATCH_SIZE/len(train_list)), mini_c_mse, lr))
				# file_writer.add_summary(summary, step+epoch*max_step )
				del mini_batch_X, mini_batch_Y, mini_batch_upscaled_X
			#os.system("echo "+str(epoch))


			saver.save(sess, ckpt_save_path+"SR_epoch_%03d.ckpt" % epoch )

			
			


    

