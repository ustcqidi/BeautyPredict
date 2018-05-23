from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import pickle
import numpy as np
from keras.layers import Dropout
import keras.backend as K
import matplotlib.pyplot as plt
from keras import regularizers


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

resnet = ResNet50(include_top=False, pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.layers[0].trainable = False
print (model.summary())

model.load_weights('model-dropout/model-ldl-resnet-base.h5')

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='kld', optimizer=sgd, metrics=['accuracy'])
# model.compile(loss=euclidean_distance_loss, optimizer=sgd, metrics=['accuracy'])

# image_data = pickle.load(open('train_image_data.dat','rb'))
lable_distribution = pickle.load(open('train_lable_distribution.dat','rb'))

# train_X = np.array(image_data[0:len(image_data)])
train_X = np.array([x[1] for x in lable_distribution[0:len(lable_distribution)]])
train_Y = np.array([x[2] for x in lable_distribution[0:len(lable_distribution)]])

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

history = model.fit(x=train_X, y=train_Y, batch_size=32, callbacks=[earlyStopping], epochs=100, verbose=1, validation_split=0.1)

# fig = plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='lower left')
# #
# fig.savefig('performance.png')

model.save_weights('model-dropout/model-ldl-resnet-base.h5')
