from keras.layers import Conv2D, Input, MaxPool2D,Flatten, Dense, Permute, Dropout
from keras.models import Model
from keras.optimizers import adam
import numpy as np
import pickle
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

def lossfun(y_true, y_pred):
    val = K.mean(K.square(y_pred-y_true),axis = -1)
    val.set_shape(K.mean(K.square(y_true),axis = -1).shape)
    return val

input = Input(shape=[128,128,3])
x = Conv2D(50,(5,5), strides=1, padding='valid',name='conv1',activation='relu')(input)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(100,(5,5), strides=1, padding='valid',name='conv2',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(150,(4,4), strides=1, padding='valid',name='conv3',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(200,(4,4), strides=1, padding='valid',name='conv4',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(250,(4,4), strides=1, padding='valid',name='conv5',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Conv2D(300,(1,1), strides=1, padding='valid',name='conv6',activation='relu')(x)
x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)
x = Flatten()(x)
x = Dropout(0.5)(x);
x = Dense(128, name='dense1')(x)
score = Dense(1,name='score')(x)

my_adam = adam(lr=0.001)

model = Model([input], [score])
print (model.summary())
# model.load_weights('model-dropout/model-52.h5', by_name=True)
model.compile(loss=[lossfun], optimizer=my_adam)

image_data = pickle.load(open('image_data.dat','rb'))
attractive_data = pickle.load(open('attractive_data.dat','rb'))

image_data = np.array(image_data[0:len(image_data)])
output_y = [x[1] for x in attractive_data[0:len(image_data)]]
#print output_y
output_y = np.array(output_y)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(image_data)

model.fit_generator(datagen.flow(image_data, output_y, batch_size=32),
                    steps_per_epoch=len(image_data) / 32, epochs=10)



for e in range(10):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(image_data, output_y, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(image_data) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

model.save_weights('model-dropout/model-aug.h5')

# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
# for i in range(55,56):
#     model.fit([image_data],[output_y],batch_size=16,epochs=10, callbacks=[earlyStopping],validation_split=0.1)
#     model.save_weights('model-dropout/model-'+str(i)+'.h5')
