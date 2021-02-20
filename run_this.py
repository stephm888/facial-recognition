import h5py
import gc
from keras import optimizers
from keras.callbacks import *
from sklearn.model_selection import train_test_split

from config import Config

import numpy as np
from sklearn.utils import shuffle

from model import baselineModel

rootdir = 'data/h5/'

if __name__ == '__main__':
    conf = Config()

    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    KTF.set_session(sess)

    with h5py.File(rootdir + 'train.h5', 'r') as h:
        X_train_1 = np.array(h['train_1'])
        X_train_2 = np.array(h['train_2'])
        y_train = np.array(h['label'])

    with h5py.File(rootdir + 'test.h5', 'r') as h:
        X_test_1 = np.array(h['test_1'])
        X_test_2 = np.array(h['test_2'])
        y_test = np.array(h['label'])

    print('xtrain1 shape:', X_train_1.shape,'xtrain2 shape:', X_train_2.shape,'ytrain shape:', y_train.shape)
    print('xtest1 shape:', X_test_1.shape,'xtest2 shape:', X_test_2.shape,'ytest shape:', y_test.shape)


    X_train = np.concatenate([X_train_1,X_train_2],axis=-1)
    del X_train_1,X_train_2
    gc.collect()
    print('new train shape:',X_train.shape)

    X_train,X_valid,y_train,y_valid = train_test_split(X_train, y_train,random_state=conf.random_seed)
    model = baselineModel(train_shape=X_train.shape[1:])
    print(X_train.shape)

    optimizer = optimizers.Adam(0.0001, decay=0.00000001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X_train, y_train,
              batch_size=conf.batch_size,
              epochs=50,
              validation_data=(X_valid, y_valid),
              callbacks=[early_stopping],
              verbose=2)
