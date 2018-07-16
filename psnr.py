
import numpy as np
import math

def psnr(target, ref):
	#assume that target and ref are uint8 type, and of N x N shape

	target_data = np.array(target).astype(np.float)
	ref_data = np.array(ref).astype(np.float)
	
	diff = ref_data - target_data
	diff = diff.flatten()
	rmse = np.sqrt( np.mean(diff ** 2.) ) 
	return 20*np.log10(255.0/rmse)

def esnr_upperhalf(target, ref):
	F_t =np.fft.fftshift( np.fft.fft2(target) )
	F_r =np.fft.fftshift( np.fft.fft2(ref)  )

	F_zeros= np.zeros( ref.shape )

	F_t[ int(F_t.shape[0]/2 - F_t.shape[0]/4+1) : int(F_t.shape[0]/2 + F_t.shape[0]/4),  int(F_t.shape[1]/2 - F_t.shape[1]/4+1) :int( F_t.shape[1]/2 + F_t.shape[1]/4 )    ] =  0
	F_r[ int(F_r.shape[0]/2 - F_r.shape[0]/4+1) : int(F_r.shape[0]/2 + F_r.shape[0]/4), int( F_r.shape[1]/2 - F_r.shape[1]/4+1) : int(F_r.shape[1]/2 + F_r.shape[1]/4 )    ] =  0

	diff= np.abs(F_t- F_r)
	diff = diff.flatten()
	error_energy =  np.sum(diff**2.)

	raw_spectrum = np.abs(F_r)
	raw_spectrum = raw_spectrum.flatten() 
	raw_energy = np.sum(raw_spectrum**2.)

	return 10 * np.log10(raw_energy/error_energy)

def esnr_lowerhalf(target, ref):
	F_t =np.fft.fftshift( np.fft.fft2(target) )
	F_r =np.fft.fftshift( np.fft.fft2(ref)  )

	F=F_t

	F[  int( F.shape[0]/2 + F.shape[0]/4): F.shape[0]   , :    ]=0
	F[ :                                                , int( F.shape[1]/2 + F.shape[1]/4 ):F.shape[1] ]   =0
	F[ 0:int( F.shape[0]/2+1- F.shape[0]/4 )            , :    ]  =0
	F[ :                                                , 0:int( F.shape[1]/2+1-F.shape[1]/4 ) ] =0

	F_t=F

	F=F_r

	F[  int( F.shape[0]/2 + F.shape[0]/4): F.shape[0]   , :    ]=0
	F[ :                                                , int( F.shape[1]/2 + F.shape[1]/4 ):F.shape[1] ]   =0
	F[ 0:int( F.shape[0]/2+1- F.shape[0]/4 )            , :    ]  =0
	F[ :                                                , 0:int( F.shape[1]/2+1-F.shape[1]/4 ) ] =0

	F_r=F	


	# temp_F_t = np.zeros(ref.shape).astype(np.complex128)
	# temp_F_r = np.zeros(ref.shape).astype(np.complex128)

	# cP = [int(np.floor((ref.shape[0]+1)/2.0)),int(np.floor((ref.shape[0]+1)/2.0))]
	# temp_F_t[cP[0]- int(ref.shape[0]/4): cP[0]+int(ref.shape[0]/4), cP[1]- int(ref.shape[1]/4): cP[1]+int(ref.shape[1]/4) ] =  F_t[ cP[0]- int(ref.shape[0]/4): cP[0]+int(ref.shape[0]/4), cP[1]- int(ref.shape[1]/4): cP[1]+int(ref.shape[1]/4)   ] 
	# temp_F_r[cP[0]- int(ref.shape[0]/4): cP[0]+int(ref.shape[0]/4), cP[1]- int(ref.shape[1]/4): cP[1]+int(ref.shape[1]/4) ] =  F_r[ cP[0]- int(ref.shape[0]/4): cP[0]+int(ref.shape[0]/4), cP[1]- int(ref.shape[1]/4): cP[1]+int(ref.shape[1]/4)   ]   
	# F_t = temp_F_t
	# F_r = temp_F_r

	diff= np.abs(F_t- F_r)
	diff = diff.flatten()

	error_energy =  np.sum(diff**2.)
	raw_spectrum = np.abs(F_r)
	raw_spectrum = raw_spectrum.flatten() 
	raw_energy = np.sum(raw_spectrum**2.)

	return 10 * np.log10(raw_energy/error_energy)
