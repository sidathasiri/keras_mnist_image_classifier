# import libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt

# load MNIST dataset from keras
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape data to feed into CNN
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
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

# plot model performance
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('Model Performance')
plt.ylabel('Score')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper right')
plt.show()

# test set accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('\nTest accuracy:', test_acc)
