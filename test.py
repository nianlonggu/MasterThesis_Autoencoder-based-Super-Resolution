import os, glob, re, signal, sys, argparse, threading, time
import math
import random
import numpy as np
import scipy.io
from sr_utils import *
from model import *
from PIL import Image
import time
import matplotlib.pyplot as plt
from psnr import *
import scipy
import pickle
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--test_path")
parser.add_argument("--ckpt_path")
parser.add_argument("--image_name")
parser.add_argument("--scale")
parser.add_argument("--epoch_num")
parser.add_argument("--image_save_path")
parser.add_argument("--psnr_save_path")
args = parser.parse_args()
TEST_DATA_PATH = args.test_path
ckpt_path = args.ckpt_path
image_name = args.image_name
SCALE = int(args.scale)
epoch_num=args.epoch_num
image_save_path=args.image_save_path
psnr_save_path=args.psnr_save_path


# image_save_path = 'data/test_out/'+TEST_DATA_PATH.split('/')[-3]+'/'+TEST_DATA_PATH.split('/')[-2]+'/'
# psnr_save_path='psnr/' + TEST_DATA_PATH.split('/')[-3]+'/'+TEST_DATA_PATH.split('/')[-2]+'/'

if not os.path.exists(image_save_path    ):
			os.makedirs(image_save_path  )

if not os.path.exists(psnr_save_path    ):
			os.makedirs(psnr_save_path  )


if __name__ == '__main__':
	model_list = sorted( glob.glob( ckpt_path +"SR_epoch_*" ) )
	model_list = [ fn.split('.data')[0] for fn in model_list if not ( fn.endswith("meta") or fn.endswith("index") ) ]

	test_list = get_test_list( TEST_DATA_PATH, SCALE )

	# print(len(test_list))

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)   
	with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1, gpu_options=gpu_options)) as sess:


		raw_tensor = tf.placeholder( tf.float32, shape=(1, None, None, 1) )
		SHEVC_lr_tensor = tf.placeholder(tf.float32, shape=(1,None, None,1) )

		#downsampling
		downscaled_tensor = downscale_model( raw_tensor, SHEVC_lr_tensor )

		#upscaling
		upscaled_tensor = upscale_model(   downscaled_tensor )

		saver = tf.train.Saver(var_list=None, max_to_keep=0)
		
		# init = tf.global_variables_initializer()
		# sess.run(init)	

		if epoch_num =='all':	

			psnr_esnr_dict=dict()
			"""
			psnr_esnr_dict contains the psnr and esnr_upperhalf and esnr_lowerhalf of the averaged value of test data, for each epoch to the tested.
			it is used to plot the training curve of the autoencoder model

			"""

			for i in range(len(test_list)):
				psnr_sr_list={'total':[],'upperhalf':[], 'lowerhalf':[]}
				psnr_interpolation_list ={'total':[],'upperhalf':[], 'lowerhalf':[]}
				psnr_esnr_dict[test_list[i][3]]={ "sr": psnr_sr_list, "interpolation" :psnr_interpolation_list  }


			for model_ckpt in model_list:
				epoch = int(model_ckpt.split('epoch_')[1].split('.ckpt')[0])
				print("Testing model:", model_ckpt)
				saver.restore( sess, model_ckpt )
				psnr_sr_list_tmp =[]
				psnr_interpolation_list_tmp=[]


				for i in range(len(test_list)):

					# input, ref, and upsacale are all 4d matrix
					lr_img, ref_img ,upscaled_img= get_test_image( test_list, i  )
					image_y_hat_sr, image_y_lr = sess.run( [upscaled_tensor, downscaled_tensor ] , feed_dict={ raw_tensor: ref_img, SHEVC_lr_tensor: lr_img }   ) 
					## here must use matlab_uint8 self defined function
					image_y_hat_sr = matlab_uint8(image_y_hat_sr.squeeze()*255)
					image_y_lr = matlab_uint8( image_y_lr.squeeze() *255  )
					ref_img = matlab_uint8(ref_img.squeeze()*255)
					image_y_hat_interpolation = matlab_uint8(upscaled_img.squeeze()*255)


					# This shave is used to produce comparable results with the SRCNN model, since in their model they also use this shaving operation 
					ref_img=shave( ref_img, [SCALE, SCALE] )
					image_y_hat_interpolation=shave(image_y_hat_interpolation,[SCALE,SCALE])
					image_y_hat_sr=shave(image_y_hat_sr,[SCALE,SCALE])
					image_y_lr= shave( image_y_lr, [1,1])

					psnr_interpolation = psnr( image_y_hat_interpolation, ref_img )
					psnr_sr = psnr( image_y_hat_sr, ref_img )

					esnr_upperhalf_interpolation = esnr_upperhalf( image_y_hat_interpolation, ref_img )
					esnr_upperhalf_sr = esnr_upperhalf(image_y_hat_sr, ref_img)

					esnr_lowerhalf_interpolation = esnr_lowerhalf( image_y_hat_interpolation, ref_img )
					esnr_lowerhalf_sr = esnr_lowerhalf(image_y_hat_sr, ref_img)					

					psnr_esnr_dict[ test_list[i][3]]['sr']['total'].append( psnr_sr )
					psnr_esnr_dict[ test_list[i][3]]['interpolation']['total'].append(psnr_interpolation)

					psnr_esnr_dict[ test_list[i][3]]['sr']['upperhalf'].append( esnr_upperhalf_sr )
					psnr_esnr_dict[ test_list[i][3]]['interpolation']['upperhalf'].append(esnr_upperhalf_interpolation)

					psnr_esnr_dict[ test_list[i][3]]['sr']['lowerhalf'].append( esnr_lowerhalf_sr )
					psnr_esnr_dict[ test_list[i][3]]['interpolation']['lowerhalf'].append(esnr_lowerhalf_interpolation)					

					print( test_list[i][3] ,'\tinterpolation:', psnr_interpolation, '\tsr:',psnr_sr )
					psnr_sr_list_tmp.append(psnr_sr)
					psnr_interpolation_list_tmp.append(psnr_interpolation)

					if epoch % 10 ==0:
						scipy.misc.imsave(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_2interpolation.png',  image_y_hat_interpolation  )
						scipy.misc.imsave(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_1sr.png',  image_y_hat_sr  )
						scipy.misc.imsave(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_0raw.png',  ref_img )
						scipy.misc.imsave(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_3downsampled.png', image_y_lr  )
						# scipy.io.savemat(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_3downsampled', {'image': image_y_lr }  )
				print('mean psnr interpolation:','%.4f' % (sum(psnr_interpolation_list_tmp)/len(psnr_interpolation_list_tmp)),'mean psnr sr:','%.4f' % (sum(psnr_sr_list_tmp)/len(psnr_sr_list_tmp)))		
			with open(psnr_save_path+'psnr_esnr_dict', 'wb') as f:
				pickle.dump(psnr_esnr_dict, f)

		else:
			epoch_num= int(epoch_num)
			for model_name in model_list:
				if str(epoch_num) in model_name:
					model_ckpt=model_name
					break
			epoch = int(model_ckpt.split('epoch_')[1].split('.ckpt')[0])
			print("Testing model:", model_ckpt)
			saver.restore( sess, model_ckpt )

			for i in range(len(test_list)):

				if image_name not in test_list[i][3]:
					continue

				# input, ref, and upsacale are all 4d matrix
				lr_img, ref_img ,upscaled_img= get_test_image( test_list, i  )

				image_y_hat_sr, image_y_lr = sess.run( [upscaled_tensor, downscaled_tensor ] , feed_dict={ raw_tensor: ref_img, SHEVC_lr_tensor:lr_img }   ) 

				## here must use matlab_uint8 self defined function
				image_y_hat_sr = matlab_uint8(image_y_hat_sr.squeeze()*255)
				image_y_lr = matlab_uint8( image_y_lr.squeeze() *255  )
				ref_img = matlab_uint8(ref_img.squeeze()*255)
				image_y_hat_interpolation = matlab_uint8(np.rint(upscaled_img.squeeze()*255))

				ref_img=shave( ref_img, [SCALE, SCALE] )
				image_y_hat_interpolation=shave(image_y_hat_interpolation,[SCALE,SCALE])
				image_y_hat_sr=shave(image_y_hat_sr,[SCALE,SCALE])
				image_y_lr= shave( image_y_lr, [1,1])

				psnr_interpolation = psnr( image_y_hat_interpolation, ref_img )
				psnr_sr = psnr( image_y_hat_sr, ref_img )

				esnr_upperhalf_interpolation = esnr_upperhalf( image_y_hat_interpolation, ref_img )
				esnr_upperhalf_sr = esnr_upperhalf(image_y_hat_sr, ref_img)

				esnr_lowerhalf_interpolation = esnr_lowerhalf( image_y_hat_interpolation, ref_img )
				esnr_lowerhalf_sr = esnr_lowerhalf(image_y_hat_sr, ref_img)					


				# psnr_sr_list.append( psnr_sr )
				# psnr_interpolation_list.append(psnr_interpolation)
				print( test_list[i][3] , '\tpsnr_interpolation:\t\t', '%.4f' % psnr_interpolation, '\t\tpsnr_sr:\t\t','%.4f' % psnr_sr )
				print( test_list[i][3] , '\tesnr_lowerhalf_interpolation:\t', '%.4f' % esnr_lowerhalf_interpolation, '\t\tesnr_lowerhalf_sr:\t','%.4f' % esnr_lowerhalf_sr )
				print( test_list[i][3] , '\tesnr_upperhalf_interpolation:\t','%.4f' % esnr_upperhalf_interpolation, '\t\tesnr_upperhalf_sr:\t','%.4f' % esnr_upperhalf_sr )

				scipy.misc.imsave(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_2interpolation.png',  image_y_hat_interpolation  )
				scipy.misc.imsave(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_1sr.png',  image_y_hat_sr  )
				scipy.misc.imsave(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_0raw.png',  ref_img )
				scipy.misc.imsave(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_3downsampled.png', image_y_lr )
				# scipy.io.savemat(image_save_path+'epoch'+str(epoch)+'_'+test_list[i][3].split('/')[-1].split('.')[0]+'_3downsampled', {'image': image_y_lr }  )

				# plt.figure( test_list[i][3]+'_raw')
				# plt.imshow( ref_img )
				# plt.figure( test_list[i][3]+'_sr')
				# plt.imshow( image_y_hat_sr)
				# plt.figure( test_list[i][3]+'_interpolation')
				# plt.imshow( image_y_hat_interpolation )
				# plt.figure( test_list[i][3]+'_downsampled')
				# plt.imshow( matlab_uint8( image_y_lr.squeeze()*255 ) )
				# plt.show()





