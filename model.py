from keras.models import *
from keras.layers import *
from keras.utils import plot_model

from config import Config

conf = Config()
bn_axis = 3 #channel last
eps = 1.1e-5

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_name = name + '_BatchNorm' if name else None
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation:
        ac_name = name + '_Activation' if name else None
        x = Activation(activation, name=ac_name)(x)

    return x

def CNN_block(img_input):
    # convolutional layer 0 - 16 kernals
    x = conv2d_bn(img_input,filters=16, kernel_size=(5,5), name='conv0')
    # pooling 0
    x = MaxPooling2D(pool_size = (2, 2), strides=2, name='pool0')(x)
    # convolutional layer 1 - 32 kernals
    x = conv2d_bn(x, filters=32, kernel_size=(5, 5), name='conv1')
    # pooling 1
    x = MaxPooling2D(pool_size = (2, 2), strides=2, name='pool1')(x)
    # convolutional layer 2 - 64 kernals
    x = conv2d_bn(x, filters=64, kernel_size=(5, 5), name='conv2')
    # pooling 2
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2')(x)
    # flattern[batch_size, k]
    x_fc = Flatten()(x)
    # fully connected layer 0
    x_fc = Dense(1024, activation='relu', name='fc0')(x_fc)
    x_fc = Dropout(0.5)(x_fc)
    # fully connected layer 1
    x_fc = Dense(128, activation='relu', name='fc1')(x_fc)
    x_fc = Dropout(0.5)(x_fc)

    return x_fc

def baselineModel(train_shape):
    x = Input(train_shape)
    out = CNN_block(x)
    out = Dense(1, activation='sigmoid', name='fc_concat')(out)

    model = Model(inputs=x, outputs=out)
    model.summary()
    return model
