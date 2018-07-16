# Autoencoder based Single Image Super Resolution
## Introduction
Single image super resolution (SISR) is an ill-posed problem in computer vision and shows
its potential in the context of video coding. Since the SRCNN [1] model was first proposed, it
has become the current research focus in this field that training deep-learning based model
to perform super resolution.

The current flow of deep-learning based super resolution is shown in Figure 1. First bicubic/SHVC method is used to 
down-sample the raw image into low-resolution image, then the low-resolution image is up-scaled via interpolation
method. The interpolated image is used for the training and testing of deep-learning models.
![generalstructuresisr](https://user-images.githubusercontent.com/40075166/41413771-5a528e36-6fe4-11e8-983f-ad7fb4af8228.png)
                Figure 1: General structure of current deep-learning based SISR models.


In this project it is found that different down-sampling
methods have an deep influence on the training and performance of the deep-learning based
SISR models. Using down-sampling and interpolation methods with almost no alias for
training is not helpful for the network to recover the information in the upper half frequency.

Based on these conclusions, an autoencoder model which can learn both down-sampling and
up-sampling operations simultaneously was designed, with the hope that the autoencoder
model can learn a proper down-sampling method, such that more information in the
upper half frequency range can be recovered. Testing results showed that compared with
the VDSR [2] model this autoencoder model can achieve higher PSNR values.

The structure of the autoencoder model is shown in Figure 2.
![autoencodermodel](https://user-images.githubusercontent.com/40075166/41414770-3ca69fb4-6fe7-11e8-8751-22b3522fdfd7.png)
 Figure 2: Structure of Autoencoder based SISR model.

The testing results are given in Table 1. Comparison are made between autoencoder model and existing SISR models.
![testingresults](https://user-images.githubusercontent.com/40075166/41416215-a2d5bc0e-6fea-11e8-874a-8ff0afd46265.png)
In Table 1 the ESNR is Energy Signal-to-Noise Ratio. It is proposed in this project to evaluated the performance of SR models in frequency domain. It is defined by the raw image energy to the
error energy, expressed in dB.

## Package Usage
Usage:

step 1: Generate the training data and testing data
	
	go into resources/ , execute the generate_train_and_test_data.m matlab script. This will generate the training data from 291 dataset, and testing data from Set5;
	The training data and testing data will be stored in data/ folder by default

step 2: Training the auto encoder model
	
	first modify the train_config.cfg file. In this file there are descriptions on how to set the parameters.
	then type command:  bash train.sh

step 3: Testing the auto encoder model

	In the checkpoint there has been a epoch 38, which is the stable version so far, tests can be made based on this epochs.
	or run test to see the training performance curve.

	first modify the test_config.cfg file. There are descriptions on how to set the parameters.
	then type command:  bash test.sh

step 4: See training performance curve

    after run testing, in order to see the training curve, run command:  python3 plot.py --plot_path YOUR_PATH_OF_PSNR
    
    YOUR_PATH_OF_PSNR is the file path where the psnr information is stored. For example it could be:
    results/psnr/Set5/SHVCscale2/

## Contact
For further information, please contact  nianlonggu@gmail.com

## Bib

[1] Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution[C]//European Conference on Computer Vision. Springer, Cham, 2014: 184-199.

[2] Kim J, Kwon Lee J, Mu Lee K. Accurate image super-resolution using very deep convolutional networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 1646-1654.
