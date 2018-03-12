from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Activation, Flatten, MaxPooling2D , Input
from keras.layers import BatchNormalization, AveragePooling2D
import os
import datetime

batch_size = 32
num_classes= 10
epochs = 300
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(),'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0]," train samples")
print(y_test.shape[0]," test sampels")

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

input_img = Input(shape=x_train.shape[1:], name='input_img') #(32, 32, 3)

k = 12 
theta=0.5
learning_rate = 0.1
step1 = int(0.5*epochs)
step2 = int(0.75*epochs)
num_blobs=[6, 18, 24]

def dense_block(input_tensor,channels,name=None):
    bn1 = BatchNormalization()(input_tensor)
    relu = Activation('relu')(bn1)
    conv = Conv2D(channels, (3,3),padding='same')(relu)
    return conv

def dense_b_block(input_tensor,channels,name=None):
    bn1 = BatchNormalization()(input_tensor)
    relu = Activation('relu')(bn1)
    conv = Conv2D(4*channels, (1,1),padding='same')(relu)
    bn2 = BatchNormalization()(conv)
    relu2 = Activation('relu')(bn2)
    conv2 = Conv2D(channels,(3,3),padding='same')(relu2)
    return conv2

def transition_layer(input_tensor, k,name=None):
    conv = Conv2D(k,(1,1),padding='same')(input_tensor)
    pool = AveragePooling2D(pool_size=(2,2),strides=(2,2))(conv)
    return pool

# first_layer before input Dense Block
first_layer = Conv2D(2*k, (3,3),padding='same')(input_img)
x = MaxPooling2D(pool_size=(2,2))(first_layer)

b1_1 = dense_b_block(x,k)
b1_1_conc = keras.layers.concatenate([x,b1_1],axis=3)
b1_2 = dense_b_block(b1_1_conc,k)
b1_2_conc = keras.layers.concatenate([x,b1_1,b1_2],axis=3)
b1_3 = dense_b_block(b1_2_conc,k)
b1_3_conc = keras.layers.concatenate([x,b1_1,b1_2,b1_3],axis=3)
b1_4 = dense_b_block(b1_3_conc,k)
b1_4_conc = keras.layers.concatenate([x,b1_1,b1_2,b1_3,b1_4],axis=3)
b1_5 = dense_b_block(b1_4_conc,k)
b1_5_conc = keras.layers.concatenate([x,b1_1,b1_2,b1_3,b1_4,b1_5],axis=3)
b1_6 =dense_b_block(b1_5_conc,k)

pool1 = transition_layer(b1_6,k) # pool2 shape( 4,4,3)

#pool1 = keras.layers.concatenate([x,pool1],axis=3)

b2_1 = dense_b_block(pool1,k)
b2_1_conc = keras.layers.concatenate([pool1, b2_1],axis=3)
b2_2 = dense_b_block(b2_1_conc,k)
b2_2_conc = keras.layers.concatenate([pool1, b2_1, b2_2],axis=3)
b2_3 = dense_b_block(b2_2_conc,k)
b2_3_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3],axis=3)
b2_4 = dense_b_block(b2_3_conc,k)
b2_4_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4],axis=3)
b2_5 = dense_b_block(b2_4_conc,k)
b2_5_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5],axis=3)
b2_6 = dense_b_block(b2_5_conc,k)
b2_6_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6],axis=3)
b2_7 = dense_b_block(b2_6_conc,k)
b2_7_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7],axis=3)
b2_8 = dense_b_block(b2_7_conc,k)
b2_8_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8],axis=3)
b2_9 = dense_b_block(b2_8_conc,k)
b2_9_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9],axis=3)
b2_10 = dense_b_block(b2_9_conc,k)
b2_10_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10],axis=3)
b2_11 = dense_b_block(b2_10_conc,k)
b2_11_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10,b2_11],axis=3)
b2_12 = dense_b_block(b2_11_conc,k)
b2_12_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10,b2_11,b2_12],axis=3)
b2_13 = dense_b_block(b2_12_conc,k)
b2_13_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10,b2_11,b2_12,b2_13],axis=3)
b2_14 = dense_b_block(b2_13_conc,k)
b2_14_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10,b2_11,b2_12,b2_13,b2_14],axis=3)
b2_15 = dense_b_block(b2_14_conc,k)
b2_15_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10,b2_11,b2_12,b2_13,b2_14,b2_15],axis=3)
b2_16 = dense_b_block(b2_15_conc,k)
b2_16_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10,b2_11,b2_12,b2_13,b2_14,b2_15,b2_16],axis=3)
b2_17 = dense_b_block(b2_16_conc,k)
b2_17_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10,b2_11,b2_12,b2_13,b2_14,b2_15,b2_16,b2_17],axis=3)
b2_18 = dense_b_block(b2_17_conc,k)
b2_18_conc = keras.layers.concatenate([pool1, b2_1,b2_2,b2_3, b2_4,b2_5, b2_6, b2_7,b2_8,b2_9,b2_10,b2_11,b2_12,b2_13,b2_14,b2_15,b2_16,b2_17,b2_18],axis=3)

pool2 = transition_layer(b2_18_conc,k) # pool2 shape( 4,4,3)

b3_1 = dense_b_block(pool2,k)
b3_1_conc = keras.layers.concatenate([pool2,b3_1],axis=3)
b3_2 = dense_b_block(b3_1_conc,k)
b3_2_conc = keras.layers.concatenate([pool2,b3_1,b3_2],axis=3)
b3_3 = dense_b_block(b3_2_conc,k)
b3_3_conc = keras.layers.concatenate([pool2,b3_1,b3_2,b3_3],axis=3)
b3_4 = dense_b_block(b3_3_conc,k)
b3_4_conc = keras.layers.concatenate([pool2,b3_1,b3_2,b3_3,b3_4],axis=3)
b3_5 = dense_b_block(b3_4_conc,k)
b3_5_conc = keras.layers.concatenate([pool2,b3_1,b3_2,b3_3,b3_4,b3_5],axis=3)
b3_6 =dense_b_block(b3_5_conc,k)

pool_f = AveragePooling2D(pool_size=(4,4))(b3_6)
flattenL = Flatten()(pool_f)
logits = Dense(10, activation='softmax')(flattenL)

model = Model(inputs=input_img, outputs=logits)
opt = keras.optimizers.RMSprop(lr=0.01,decay=1e-6)
model.compile(optimizer=opt, 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)

t_start = datetime.datetime.now()
scores = model.evaluate(x_test,y_test, verbose=1)
t_end = datetime.datetime.now()
print('Test loss :',scores[0])
print('Test accuracy :',scores[1])
print("test time is: ",(t_end-t_start).seconds)

    
        





