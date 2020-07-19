# import libraries
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm

# load MNIST dataset
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# add noise to data
noise_factor = 0.25
train_images_noisy = train_images + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=train_images.shape)
test_images_noisy = test_images + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=test_images.shape)
train_images_noisy = np.clip(train_images_noisy, 0., 1.)
test_images_noisy = np.clip(test_images_noisy, 0., 1.)

# reshape data to feed into CNN
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images_noisy = train_images_noisy.reshape(60000, 28, 28, 1)
test_images_noisy = test_images_noisy.reshape(10000, 28, 28, 1)

# Autoencoder to reduce noise
auto_encoder = Sequential()
auto_encoder.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(
    2), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
auto_encoder.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(
    2), activation='relu', kernel_initializer='he_uniform'))
auto_encoder.add(Conv2DTranspose(32, kernel_size=(3, 3), kernel_constraint=max_norm(
    2), activation='relu', kernel_initializer='he_uniform'))
auto_encoder.add(Conv2DTranspose(64, kernel_size=(3, 3), kernel_constraint=max_norm(
    2), activation='relu', kernel_initializer='he_uniform'))
auto_encoder.add(Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(
    2), activation='sigmoid', padding='same'))

# normalize data
train_images_noisy = train_images_noisy / 255.0
test_images_noisy = test_images_noisy / 255.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# compile autoencoder
auto_encoder.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# use 50% of training data to train the autoencoder to reduce noise
X_noise_train, X_noise_test, y_noise_train, y_noise_test = train_test_split(
    train_images_noisy, train_images, test_size=0.75, random_state=42, shuffle=True)
auto_encoder.fit(train_images_noisy, train_images, epochs=3)

denoised_train_images = auto_encoder.predict(train_images_noisy)
denoised_test_images = auto_encoder.predict(test_images_noisy)

# CNN model
model = models.Sequential()
model.add(layers.Conv2D(15, (7, 7), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(30, (7, 7), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# train model
history = model.fit(denoised_train_images, train_labels, epochs=10)

# plot model performance
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model Performance')
plt.ylabel('Score')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()

# test accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('\nTest accuracy:', test_acc)
