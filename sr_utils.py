import os, glob, re, signal, sys, argparse, threading, time
import math
import random
import numpy as np
import scipy.misc
import scipy
import scipy.io
import matplotlib.pyplot as plt

#for email alert
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


#this method is a emulation of matlab uint8() function, so as to produce consistent outcome with the matlab code of SRCNN
#however, the np.rint always round to nearest even number, which makes this func not compeltely identical 
def matlab_uint8( nd_arr ):
	return np.rint( np.minimum(np.maximum( nd_arr , 0),255)  ).astype(np.uint8)

def shave( imgs, border ):
	height, width = imgs.shape[0], imgs.shape[1]
	imgs = imgs[ border[0]: height-border[0], border[1]:width- border[1] ]
	return imgs





def get_train_list(path, scale):
	l = glob.glob(os.path.join(path,"*"))
	# print (len(l))
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	# print (len(l))
	train_list = []
	tag_lr = '_'+str(scale)+'_lr.mat'
	tag_interp = '_'+str(scale)+'_interp.mat'
	ind=0
	total_num = len(l)
	os.system('echo loading file ...')
	for f in l:
		if os.path.exists(f):
			if os.path.exists(f[:-4]+tag_lr)  and os.path.exists(f[:-4]+tag_interp)  : 
				input_img = scipy.io.loadmat( f[:-4]+tag_lr)['patch']
				ref_img = scipy.io.loadmat(f)['patch']
				upscaled_img =scipy.io.loadmat(f[:-4]+tag_interp)['patch']
				train_list.append( [input_img, ref_img, upscaled_img ])
		ind +=1
		if ind % 100 ==0:
			os.system('echo '+ "%02.f" % (ind*100.0/total_num ))
	return train_list

def get_mini_batch( train_list, offset, mini_batch_size ):
	mini_batch_input=[]
	mini_batch_ref=[]
	mini_batch_upscaled=[]
	for i in range( offset, offset+mini_batch_size ):
		input_img = train_list[i][0]
		ref_img = train_list[i][1]
		upscaled_img = train_list[i][2]

		mini_batch_input.append( input_img )
		mini_batch_ref.append( ref_img)
		mini_batch_upscaled.append( upscaled_img )

	mini_batch_input=np.asarray( mini_batch_input )
	mini_batch_ref=np.asarray(mini_batch_ref)
	mini_batch_upscaled=np.asarray(mini_batch_upscaled)

	mini_batch_input = mini_batch_input.reshape( mini_batch_input.shape[0], mini_batch_input.shape[1], mini_batch_input.shape[2], 1 )
	mini_batch_ref = mini_batch_ref.reshape( mini_batch_ref.shape[0], mini_batch_ref.shape[1], mini_batch_ref.shape[2], 1 )
	mini_batch_upscaled = mini_batch_upscaled.reshape( mini_batch_upscaled.shape[0], mini_batch_upscaled.shape[1], mini_batch_upscaled.shape[2], 1 )
	## test if the data is right
	# test_batch=scipy.misc.imresize( (mini_batch_ref[0,:,:,0]*255).astype(np.uint8), 1/2, interp="bicubic" )
	# print(((mini_batch_input[0,:,:,0]*255).astype(np.uint8))== test_batch )

	return mini_batch_input, mini_batch_ref,mini_batch_upscaled


def get_test_list(path, scale):
	
	file_name_list= glob.glob(os.path.join(path,"*"))
	f_list = [ f for f in file_name_list if '_lr.mat' not in f and '_interp.mat' not in f ]
	test_list = []
	tag_lr = '_'+str(scale)+'_lr.mat'
	tag_interp = '_'+str(scale)+'_interp.mat'

	for f in f_list:
		if os.path.exists(f):
			if os.path.exists(f[:-4]+tag_lr)  and os.path.exists(f[:-4]+tag_interp)  : 
				input_img = scipy.io.loadmat( f[:-4]+tag_lr)['image']
				ref_img = scipy.io.loadmat(f)['image']
				upscaled_img =scipy.io.loadmat(f[:-4]+tag_interp)['image']
				test_list.append([ input_img,ref_img, upscaled_img  , f.split('/')[-1].split('.')[0] ])

	return test_list


def get_test_image(test_list, index ):
	input_img =  test_list[index][0]
	ref_img = test_list[index][1]
	upscaled_img =test_list[index][2]

	input_img= input_img.reshape( (1, input_img.shape[0], input_img.shape[1], 1  ) )
	ref_img = ref_img.reshape( (1, ref_img.shape[0], ref_img.shape[1], 1  ) )
	upscaled_img = upscaled_img.reshape( (1, upscaled_img.shape[0], upscaled_img.shape[1], 1  ) )

	return input_img, ref_img, upscaled_img
