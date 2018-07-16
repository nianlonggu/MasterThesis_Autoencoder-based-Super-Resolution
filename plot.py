
import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--plot_path")
args = parser.parse_args()
plot_path = args.plot_path



dict_path= plot_path+'psnr_esnr_dict'

with open( dict_path, 'rb' ) as f:
	psnr_esnr_dict=pickle.load(f)


image_name_list= psnr_esnr_dict.keys()

fig=plt.figure(1)
# index=0

psnr_interpolation_list=[]
psnr_sr_list=[]

esnr_upperhalf_interpolation_list =[]
esnr_upperhalf_sr_list =[]

esnr_lowerhalf_interpolation_list =[]
esnr_lowerhalf_sr_list =[]

for img_name in image_name_list:
	# index+=1
	# fig.add_subplot( 4,4,index )
	psnr_interpolation_list.append( psnr_esnr_dict[img_name]['interpolation']['total'] )
	psnr_sr_list.append( psnr_esnr_dict[img_name]['sr']['total'] )

	esnr_upperhalf_interpolation_list.append( psnr_esnr_dict[img_name]['interpolation']['upperhalf'] )
	esnr_upperhalf_sr_list.append( psnr_esnr_dict[img_name]['sr']['upperhalf'] )

	esnr_lowerhalf_interpolation_list.append( psnr_esnr_dict[img_name]['interpolation']['lowerhalf'] )
	esnr_lowerhalf_sr_list.append( psnr_esnr_dict[img_name]['sr']['lowerhalf'] )
	# plt.plot( psnr_esnr_dict[img_name]['sr'], 'g--', label="sr")
	# plt.plot( psnr_esnr_dict[img_name]['interpolation'], 'r--', label="interpolation")
	# # plt.tight_layout()
	# plt.grid()
	# plt.xlabel('epoch')
	# plt.ylabel('psnr')
	# plt.legend()
	# plt.title(img_name)

psnr_interpolation_list= np.asarray( psnr_interpolation_list )
psnr_sr_list= np.asarray( psnr_sr_list )
mean_psnr_interpolation = np.mean( psnr_interpolation_list, axis=0 )
mean_psnr_sr =np.mean( psnr_sr_list, axis=0 )

esnr_upperhalf_interpolation_list= np.asarray( esnr_upperhalf_interpolation_list )
esnr_upperhalf_sr_list= np.asarray( esnr_upperhalf_sr_list )
mean_esnr_upperhalf_interpolation = np.mean( esnr_upperhalf_interpolation_list, axis=0 )
mean_esnr_upperhalf_sr =np.mean( esnr_upperhalf_sr_list, axis=0 )

esnr_lowerhalf_interpolation_list= np.asarray( esnr_lowerhalf_interpolation_list )
esnr_lowerhalf_sr_list= np.asarray( esnr_lowerhalf_sr_list )
mean_esnr_lowerhalf_interpolation = np.mean( esnr_lowerhalf_interpolation_list, axis=0 )
mean_esnr_lowerhalf_sr =np.mean( esnr_lowerhalf_sr_list, axis=0 )

# index +=1
# fig.add_subplot( 4,4,index )
plt.plot( mean_psnr_sr, 'g-', label="autoencoder")
plt.plot( mean_psnr_interpolation, 'r-', label="SHVC-interpolation")
# plt.tight_layout()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('psnr (dB)')
plt.legend()
plt.title('Average PSNR overall spectrum')

# print(' interpolation  ',mean_psnr_interpolation)
# print(' sr  ', mean_psnr_sr)

plt.figure(2)
plt.plot( mean_esnr_upperhalf_sr, 'g-', label="autoencoder")
plt.plot( mean_esnr_upperhalf_interpolation, 'r-', label="SHVC-interpolation")
# plt.tight_layout()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('esnr upperhalf (dB)')
plt.legend()
plt.title('Average ESNR upperhalf')

# print(' interpolation  ',mean_psnr_interpolation)
# print(' sr  ', mean_psnr_sr)

plt.figure(3)
plt.plot( mean_esnr_lowerhalf_sr, 'g-', label="autoencoder")
plt.plot( mean_esnr_lowerhalf_interpolation, 'r-', label="SHVC-interpolation")
# plt.tight_layout()
plt.grid()
plt.xlabel('epoch')
plt.ylabel('esnr lowerhalf (dB)')
plt.legend()
plt.title('Average ESNR lowerhalf')

plt.show()
# help(fig.add_subplot)

