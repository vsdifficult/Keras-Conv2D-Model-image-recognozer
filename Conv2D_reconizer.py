import numpy as np
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation 
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.utils import np_utils 
from keras.datasets import cifar10 
from keras.constraints import maxnorm

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32') 
X_train = X_train / 255.0 
X_test = X_test / 255.0 

y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test) 

class_num = y_test.shape[1] 

model = Sequential() 
model.add(Conv2D(32, (3,3), input_shape = X_train.shape[:1], padding='same')) 
model.add(Activation('relu')) 
model.add(Conv2D(32, (3,3), input_shape = (3, 32, 32), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(BatchNormalization()) 
model.add(Conv2D(64, (3,3), padding="same")) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.2)) 
model.add(BatchNormalization()) 
model.add(Conv2D(64, (3,3), padding="same")) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.2)) 
model.add(BatchNormalization()) 

model.add(Flatten()) 
model.add(Dropout(0.2)) 

model.add(Dense(256, kernel_constraint=maxnorm(3))) 
model.add(Activation('relu')) 
model.add(Dropout(0.2))

model.add(Dense(128, kernel_constraint=maxnorm(3))) 
model.add(Activation('relu')) 
model.add(Dropout(0.2))

model.add(Dense(class_num)) 
model.add(Activation('relu')) 

epochs = 25
opt = Adam(0.001) 

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy']) 
print(model.summary())

model.save("KerasConv2DIMAGE.h5")

np.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
