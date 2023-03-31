"""
Written by,
Sanghyun Choo, schoo2@ncsu.edu

Paper - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4341618
"""

"""
Details of Keras implementation for EEGNet, ShallowConvNet, and DeepConvNet:

EEGNet (Lawhern et al. (2018)): http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
ShallowConvNet and DeepConvNet (Schirrmeister et al. (2017)): https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

## MTL for EEGNet
def MTL_EEGNet(emotion_nb_classes, context_nb_classes, Chans = 22, Samples = 256, 
             dropoutRate = 0.5, kernLength = 32, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
             n_nodes_mb = 128, n_nodes_eb = 32, n_nodes_cb = 32):
    """
        Inputs:  
        emotion_nb_classes: int, number of emotion classes to classify (sad, fear, and neutral in our case)
        context_nb_classes: int, number of context classes to classify (social and non-social in our case)      
        Chans, Samples    : number of channels and time points in the EEG data
        dropoutRate       : dropout fraction
        kernLength        : length of temporal convolution in first layer. We found
                            that setting this to be half the sampling rate worked
                            well in practice. For the SMR dataset in particular
                            since the data was high-passed at 4Hz we used a kernel
                            length of 32.     
        F1, F2            : number of temporal filters (F1) and number of pointwise
                            filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
        D                 : number of spatial filters to learn within each temporal
                            convolution. Default: D = 2
        dropoutType       : Either SpatialDropout2D or Dropout, passed as a string.
        n_nodes_mb        : int, number of hidden nodes for main branch
        n_nodes_eb        : int, number of hidden nodes for emotion branch
        n_nodes_cb        : int, number of hidden nodes for context branch
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1), name='input')

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = 3)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 3)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 3)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
    
    flatten      = Flatten(name = 'flatten')(block2)

    main_branch = Dense(n_nodes_nb)(flatten)
    main_branch = Activation('relu', name='main_branch')(main_branch)

    # Emotion task branch
    emotion_branch = Dense(n_nodes_eb)(main_branch)
    emotion_branch = Activation('relu')(emotion_branch)
    emotion_branch = Dense(emotion_nb_classes,
                           kernel_constraint = max_norm(norm_rate))(emotion_branch)
    emotion_branch = Activation('softmax', name='emotion_output')(emotion_branch)

    # Context task branch
    context_branch = Dense(n_nodes_cb)(main_branch)
    context_branch = Activation('relu')(context_branch)
    context_branch = Dense(context_nb_classes,
                           kernel_constraint = max_norm(norm_rate))(context_branch)
    context_branch = Activation('softmax', name='context_output')(context_branch)
    
    return Model(inputs=input1, outputs=[emotion_branch, context_branch])


## MTL for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))  

def MTL_ShallowConvNet(emotion_nb_classes, context_nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5,
                       n_nodes_mb = 1200, n_nodes_eb = 32, n_nodes_cb = 32):
    """
        Inputs:  
        emotion_nb_classes: int, number of emotion classes to classify (sad, fear, and neutral in our case)
        context_nb_classes: int, number of context classes to classify (social and non-social in our case)      
        Chans, Samples    : number of channels and time points in the EEG data
        dropoutRate       : dropout fraction
        n_nodes_mb        : int, number of hidden nodes for main branch
        n_nodes_eb        : int, number of hidden nodes for emotion branch
        n_nodes_cb        : int, number of hidden nodes for context branch
    """
    # start the model
    input_main   = Input((Chans, Samples, 1), name='input')
    block1       = Conv2D(40, (1, 13), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    
    flatten      = Flatten()(block1)

    main_branch = Dense(n_nodes_mb)(flatten)
    main_branch = Activation('relu', name='main_branch')(main_branch)

    # Emotion task branch
    emotion_branch = Dense(n_nodes_eb)(main_branch)
    emotion_branch = Activation('relu')(emotion_branch)
    emotion_branch = Dense(emotion_nb_classes,
                           kernel_constraint = max_norm(0.5))(emotion_branch)
    emotion_branch = Activation('softmax', name='emotion_output')(emotion_branch)

    # Context task branch
    context_branch = Dense(n_nodes_cb)(main_branch)
    context_branch = Activation('relu')(context_branch)
    context_branch = Dense(context_nb_classes,
                           kernel_constraint = max_norm(0.5))(context_branch)
    context_branch = Activation('softmax', name='context_output')(context_branch)
    
    return Model(inputs=input_main, outputs=[emotion_branch, context_branch])


## MTL for DeepConvNet
def MTL_DeepConvNet(emotion_nb_classes, context_nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5, n_nodes_mb = 2400, n_nodes_eb = 32, n_nodes_cb = 32):
    """
        Inputs:  
        emotion_nb_classes: int, number of emotion classes to classify (sad, fear, and neutral in our case)
        context_nb_classes: int, number of context classes to classify (social and non-social in our case)      
        Chans, Samples    : number of channels and time points in the EEG data
        dropoutRate       : dropout fraction
        n_nodes_mb        : int, number of hidden nodes for main branch
        n_nodes_eb        : int, number of hidden nodes for emotion branch
        n_nodes_cb        : int, number of hidden nodes for context branch
    """

    # start the model
    input_main   = Input((Chans, Samples, 1), name='input')
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)

    main_branch = Dense(n_nodes_mb)(flatten)
    main_branch = Activation('relu', name='main_branch')(main_branch)

    # Emotion task branch
    emotion_branch = Dense(n_nodes_eb)(main_branch)
    emotion_branch = Activation('relu')(emotion_branch)
    emotion_branch = Dense(emotion_nb_classes,
                           kernel_constraint = max_norm(0.5))(emotion_branch)
    emotion_branch = Activation('softmax', name='emotion_output')(emotion_branch)

    # Context task branch
    context_branch = Dense(n_nodes_cb)(main_branch)
    context_branch = Activation('relu')(context_branch)
    context_branch = Dense(context_nb_classes,
                           kernel_constraint = max_norm(0.5))(context_branch)
    context_branch = Activation('softmax', name='context_output')(context_branch)
    
    return Model(inputs=input_main, outputs=[emotion_branch, context_branch])