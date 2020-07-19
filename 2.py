# import libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from matplotlib import pyplot as plt

# load MNIST dataset
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# add noise to train and test images
noise_factor = 0.25
train_images_noisy = train_images + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=train_images.shape)
test_images_noisy = test_images + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=test_images.shape)
train_images_noisy = np.clip(train_images_noisy, 0., 1.)
test_images_noisy = np.clip(test_images_noisy, 0., 1.)

# normalize data
train_images = train_images_noisy / 255.0
test_images = test_images_noisy / 255.0

# reshape data to feed into CNN
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# CNN model
model = models.Sequential()
model.add(layers.Conv2D(15, (7, 7), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10))

# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# train model
history = model.fit(train_images, train_labels, epochs=10)
print(history.history.keys())

# plot model performance
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model Performance')
plt.ylabel('Score')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()

# accuracy on test data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('\nTest accuracy:', test_acc)
