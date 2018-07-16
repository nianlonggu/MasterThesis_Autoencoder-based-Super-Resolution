import numpy as np
import tensorflow as tf



# Now the input tensor is the raw image, 
# Output are upscaled image and downsampled image.

def  sr_model(input_tensor, input_tensor_lr,  scale=2):

    tensor = None
    conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
    #conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9)))
    conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
    tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))



    # in each loop build a resNet block, and then cascade them.
    for i in range(5):
        tensor_shortcut = tensor
        conv_w = tf.get_variable("conv_%02d_w" % (2*i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
        conv_b = tf.get_variable("conv_%02d_b" % (2*i+1), [64], initializer=tf.constant_initializer(0))
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))


        conv_w = tf.get_variable("conv_%02d_w" % (2*i+2), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
        conv_b = tf.get_variable("conv_%02d_b" % (2*i+2), [64], initializer=tf.constant_initializer(0))
        tensor = tf.nn.relu( tf.add(  tf.nn.bias_add( tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME' ), conv_b  ) , tensor_shortcut  )  )



    # add a down scaling conv layer, scale = 2 by default. and should converge the chanel into 1.
    conv_w = tf.get_variable("conv_%02d_w" % (19), [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_b" % (19), [1], initializer=tf.constant_initializer(0))
    # Here we want to set the downscaled image between (0,1),so we can make further processing.

    tensor = tf.nn.relu6( tf.add( tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,scale,scale,1], padding='SAME'), conv_b), input_tensor_lr) *6 )/6


   

    # this is the downsampled image, which will be encoded and transmitted between transmitter and receiver
    # now it's quantized andnormalized
    tensor_downsampled = tensor 


    # below is the deconvolutional part.
    dconv_00_w = tf.get_variable("dconv_00_w", [3,3, 64, 1], initializer=tf.contrib.layers.xavier_initializer())
    dconv_00_b = tf.get_variable("dconv_00_b", [64], initializer=tf.constant_initializer(0))
    tensor = tf.nn.relu( tf.nn.bias_add(  tf.nn.conv2d_transpose( tensor, dconv_00_w, [tf.shape(tensor)[0], tf.shape(tensor)[1]*scale,  \
                         tf.shape(tensor)[2]*scale, 64 ], strides=[ 1, scale, scale, 1 ], padding='SAME' ) , dconv_00_b    )   )



    # in each loop build a resNet block, and then cascade them.
    for i in range(5):
        tensor_shortcut = tensor
        dconv_w = tf.get_variable("dconv_%02d_w" % (2*i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
        dconv_b = tf.get_variable("dconv_%02d_b" % (2*i+1), [64], initializer=tf.constant_initializer(0))

        tensor = tf.nn.relu( tf.nn.bias_add(  tf.nn.conv2d_transpose( tensor, dconv_w, [tf.shape(tensor)[0], tf.shape(tensor)[1],  \
                         tf.shape(tensor)[2], 64 ], strides=[ 1, 1, 1, 1 ], padding='SAME' ) , dconv_b    )   )



        dconv_w = tf.get_variable("dconv_%02d_w" % (2*i+2), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
        dconv_b = tf.get_variable("dconv_%02d_b" % (2*i+2), [64], initializer=tf.constant_initializer(0))
        tensor = tf.nn.relu( tf.add(  tf.nn.bias_add(  tf.nn.conv2d_transpose( tensor, dconv_w, [tf.shape(tensor)[0], tf.shape(tensor)[1],  \
                         tf.shape(tensor)[2], 64 ], strides=[ 1, 1, 1, 1 ], padding='SAME' ) , dconv_b    ) , tensor_shortcut)  )


    
    # linear operation: merge multiple channels into 1 channel. This layer doesn't have non-linear operation.
    dconv_w = tf.get_variable("dconv_%02d_w" % (19), [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
    dconv_b = tf.get_variable("dconv_%02d_b" % (19), [1], initializer=tf.constant_initializer(0))
    tensor =  tf.nn.relu( tf.nn.bias_add(  tf.nn.conv2d_transpose( tensor, dconv_w, [tf.shape(tensor)[0], tf.shape(tensor)[1],  \
                         tf.shape(tensor)[2], 1 ], strides=[ 1, 1, 1, 1 ], padding='SAME' ) , dconv_b    ) )



    #this is the upscaled image.
    tensor_upscale = tensor


    return tensor_upscale, tensor_downsampled



def  downscale_model(input_tensor, input_tensor_lr,  scale=2):

    tensor = None
    conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
    #conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9)))
    conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
    tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))



    # in each loop build a resNet block, and then cascade them.
    for i in range(5):
        tensor_shortcut = tensor
        conv_w = tf.get_variable("conv_%02d_w" % (2*i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
        conv_b = tf.get_variable("conv_%02d_b" % (2*i+1), [64], initializer=tf.constant_initializer(0))
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))


        conv_w = tf.get_variable("conv_%02d_w" % (2*i+2), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
        conv_b = tf.get_variable("conv_%02d_b" % (2*i+2), [64], initializer=tf.constant_initializer(0))
        tensor = tf.nn.relu( tf.add(  tf.nn.bias_add( tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME' ), conv_b  ) , tensor_shortcut  )  )



    # add a down scaling conv layer, scale = 2 by default. and should converge the chanel into 1.
    conv_w = tf.get_variable("conv_%02d_w" % (19), [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_b" % (19), [1], initializer=tf.constant_initializer(0))
    # Here we want to set the downscaled image between (0,1),so we can make further processing.
    tensor = tf.nn.relu6( tf.add( tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,scale,scale,1], padding='SAME'), conv_b), input_tensor_lr) *6 )/6


    tensor = tf.fake_quant_with_min_max_args(tensor, min =0, max =1 )

    # this is the downsampled image, which will be encoded and transmitted between transmitter and receiver
    # now it's quantized andnormalized
    tensor_downsampled = tensor 

    return tensor_downsampled



def  upscale_model(input_tensor,  scale=2):

    tensor = None
    
    # below is the deconvolutional part.
    dconv_00_w = tf.get_variable("dconv_00_w", [3,3, 64, 1], initializer=tf.contrib.layers.xavier_initializer())
    dconv_00_b = tf.get_variable("dconv_00_b", [64], initializer=tf.constant_initializer(0))
    tensor = tf.nn.relu( tf.nn.bias_add(  tf.nn.conv2d_transpose( input_tensor, dconv_00_w, [tf.shape(input_tensor)[0], tf.shape(input_tensor)[1]*scale,  \
                         tf.shape(input_tensor)[2]*scale, 64 ], strides=[ 1, scale, scale, 1 ], padding='SAME' ) , dconv_00_b    )   )



    # in each loop build a resNet block, and then cascade them.
    for i in range(5):
        tensor_shortcut = tensor
        dconv_w = tf.get_variable("dconv_%02d_w" % (2*i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
        dconv_b = tf.get_variable("dconv_%02d_b" % (2*i+1), [64], initializer=tf.constant_initializer(0))

        tensor = tf.nn.relu( tf.nn.bias_add(  tf.nn.conv2d_transpose( tensor, dconv_w, [tf.shape(tensor)[0], tf.shape(tensor)[1],  \
                         tf.shape(tensor)[2], 64 ], strides=[ 1, 1, 1, 1 ], padding='SAME' ) , dconv_b    )   )



        dconv_w = tf.get_variable("dconv_%02d_w" % (2*i+2), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
        dconv_b = tf.get_variable("dconv_%02d_b" % (2*i+2), [64], initializer=tf.constant_initializer(0))
        tensor = tf.nn.relu( tf.add(  tf.nn.bias_add(  tf.nn.conv2d_transpose( tensor, dconv_w, [tf.shape(tensor)[0], tf.shape(tensor)[1],  \
                         tf.shape(tensor)[2], 64 ], strides=[ 1, 1, 1, 1 ], padding='SAME' ) , dconv_b    ) , tensor_shortcut)  )


    
    # linear operation: merge multiple channels into 1 channel. This layer doesn't have non-linear operation.
    dconv_w = tf.get_variable("dconv_%02d_w" % (19), [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
    dconv_b = tf.get_variable("dconv_%02d_b" % (19), [1], initializer=tf.constant_initializer(0))
    tensor =  tf.nn.relu( tf.nn.bias_add(  tf.nn.conv2d_transpose( tensor, dconv_w, [tf.shape(tensor)[0], tf.shape(tensor)[1],  \
                         tf.shape(tensor)[2], 1 ], strides=[ 1, 1, 1, 1 ], padding='SAME' ) , dconv_b    ) )



    #this is the upscaled image.
    tensor_upscale = tensor


    return tensor_upscale
