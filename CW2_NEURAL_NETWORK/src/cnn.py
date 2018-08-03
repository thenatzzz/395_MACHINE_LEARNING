import numpy as np
import keras
import h5py
import matplotlib.pyplot as plt
import src.utils.analysis as an
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.models import load_model
from keras import optimizers
from keras.models import model_from_json
from src.data_utils_fer2013 import get_FER2013_data


np.random.seed(123)


model = Sequential()
model.add(Convolution2D(36, (5, 5), padding="same",
			input_shape=(48, 48, 1)))
model.add(Activation("relu"))
model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(Convolution2D(48, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Convolution2D(48, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(Convolution2D(48, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Convolution2D(48, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
		# softmax classifier
model.add(Dense(7))
model.add(Activation("softmax"))
opt = optimizers.SGD(lr = 0.001, decay = 1e-5, momentum = 0.9)
model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])



# #load data
#
#
data = get_FER2013_data(28709, 0 , 3589)
#
#
# #normalize data
#
# #fit model
X_train = data['X_train'].reshape(data['X_train'].shape[0], 48, 48, 1)
X_val = data['X_test'].reshape(data['X_test'].shape[0], 48, 48, 1)
Y_train = np_utils.to_categorical(data['y_train'], 7)
Y_val = np_utils.to_categorical(data['y_test'], 7)
#
#
#
#
#
history = model.fit(X_train,
          Y_train,
          batch_size=32,
          nb_epoch=50,
          validation_split=0.1,
          verbose=1)

#test model
#print plot
# plt.subplot(211)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
#
#  # summarize history for loss
#
# plt.subplot(212)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

targets = data['y_test']
scores = model.predict(X_val,
                        verbose=0)
print(scores)
classifications = an.getClassifications(scores)
print(classifications)
cm = an.confusionMatrix(classifications, targets, 7)
print(cm)
for i in range(1, 8):
    recall = an.averageRecall(cm, i)
    precision = an.precisionRate(cm, i)
    f1measure = an.f1(precision, recall)
    print("f1 class %d: " %i, f1measure)
classification = an.classificationRate(cm, 7)

print("classification rate: ", classification)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
