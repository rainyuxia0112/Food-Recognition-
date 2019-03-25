# import
from keras import layers
from keras import models

# 宇宙洪荒，继续导包
from cv2 import imread, resize, imwrite
from my_utils import utils_paths
import os
import random
import numpy as np

# import food image
imagePaths = sorted(list(utils_paths.list_images('./dataset/main_food')))
random.shuffle(imagePaths)

print(len(imagePaths))
print(imagePaths[109])

# import data
data = []
labels = []
height = 200
width = 200

for imagePath in imagePaths:
    image = imread(imagePath)
    image = resize(image, (height, width))
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Scale
my_data = np.array(data, dtype="float") / 255.0
my_labels = np.array(labels)

# network
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from keras import layers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pickle

(trainX, testX, trainY, testY) = train_test_split(my_data,
    my_labels, test_size=0.25, random_state=1234)

print(trainY[0:10])
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
if len(trainY[0])==1:
    trainY = np.hstack((1-trainY, trainY))
    testY = np.hstack((1-testY, testY))

import warnings
warnings.filterwarnings('ignore')

model = Sequential()
model.add(Conv2D(100, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(200, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(Dense(200, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(len(lb.classes_), activation='sigmoid'))

model.summary()

from keras import optimizers

model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])

history = model.fit(trainX, trainY,
                   epochs = 15,
                   batch_size=32,
                   validation_data=(testX, testY))

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'b', label='Training Loss', c='r')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title("Training and Validation of binary coding")
plt.xlabel("Epochs")
plt.legend()

plt.show()

train_acc = history_dict['acc']
test_acc = history_dict['val_acc']

epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, 'b', label='Training Acc', c='r')
plt.plot(epochs, test_acc, 'b', label='Validation Acc')
plt.title("Training and Validation of binary coding")
plt.xlabel("Epochs")
plt.legend()

plt.show()

model.save('./output/model')
f = open('./output/label', "wb")
f.write(pickle.dumps(lb))
f.close()