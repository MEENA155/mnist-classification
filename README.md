# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
![230726342-18ce57e0-6048-4702-b8db-b7159c95fe59](https://user-images.githubusercontent.com/94677128/230782266-813ad7d6-fbb3-40d1-ba50-5bc0871a3419.png)

## Neural Network Model
![230725435-9d05144a-0bde-48c3-90b2-c8f909c32ff9](https://user-images.githubusercontent.com/94677128/230782247-ef75ce9e-fe59-425d-83c0-eb2cebae58d9.png)


## DESIGN STEPS
STEP 1:
Import tensorflow and preprocessing libraries

STEP 2:
Build a CNN model

STEP 3:
Compile and fit the model and then predict

## PROGRAM
```
Developed By:Meena S
Reg No:212221240028
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image=X_train[100]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

y_train[100]

X_train.min()

X_train.max()

X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.00

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model= keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=8,batch_size=128, validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('t.jpeg')
type(img)
img = image.load_img('t.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)), axis=1)

print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/94677128/230781184-1502ffa5-495e-4319-b38c-68a4b63fe62b.png)
![image](https://user-images.githubusercontent.com/94677128/230781215-060713c7-d079-4d92-8bbb-ba13726dcb3b.png)


### Classification Report
![image](https://user-images.githubusercontent.com/94677128/230781260-6ce3e80a-8228-44a7-b1e8-a5cdc29f4eef.png)


### Confusion Matrix
![image](https://user-images.githubusercontent.com/94677128/230781314-641c9061-466a-497a-abc7-74b8016a595e.png)


### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/94677128/230781361-34f7b188-1773-4123-9842-cb8527a43d1c.png)
![image](https://user-images.githubusercontent.com/94677128/230781375-563e9c8f-f453-42d3-88d9-5181dcb37dc0.png)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
