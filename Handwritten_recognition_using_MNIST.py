#Handwritten recognition using MNSIT – CNN
#import required library packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# import library packages for plotting the dataset
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
import random

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display some random examples from the training data
num_examples = 1
rand_indices = np.random.randint(len(x_train), size=num_examples)
tem= random.randint(1,1000)
images = x_train[tem]
labels = y_train[tem]
fig, axs = plt.subplots(1, num_examples, figsize=(20, 2))
plt.imshow(images)
plt.show()
 
# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Define the model
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(x_test, y_test))


# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Make predictions on test data
predictions = model.predict(x_test)
x=model.predict(images.reshape(1,28,28,1))

np.argmax(x,axis=1)