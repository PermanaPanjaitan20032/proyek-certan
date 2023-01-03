from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Map the "dog" and "cat" classes to the "dog" and "cat" classes in the CIFAR-10 dataset
y_train[y_train == 5] = 3
y_train[y_train == 7] = 5
y_test[y_test == 5] = 3
y_test[y_test == 7] = 5

# Get the training and test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Select all elements of y_train that are equal to 3 or 5
y_train = y_train[(y_train == 3) | (y_train == 5)]

# Select all elements of y_test that are equal to 3 or 5
y_test = y_test[(y_test == 3) | (y_test == 5)]

# Select all elements of x_train that correspond to 3 or 5 in y_train
x_train = x_train[(y_train == 3) | (y_train == 5)]

# Select all elements of x_test that correspond to 3 or 5 in y_test
x_test = x_test[(y_test == 3) | (y_test == 5)]

# Replace all elements of y_train that are equal to 3 with the value 0
y_train[y_train == 3] = 0

# Replace all elements of y_train that are equal to 5 with the value 1
y_train[y_train == 5] = 1

# Replace all elements of y_test that are equal to 3 with the value 0
y_test[y_test == 3] = 0

# Replace all elements of y_test that are equal to 5 with the value 1
y_test[y_test == 5] = 1


# Import the necessary modules
from keras.utils import to_categorical
from keras.models import Sequential

# Preprocess the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert the labels to categorical
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)


# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))


# Evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Plot the training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

# Import the necessary modules
import matplotlib.pyplot as plt

# Plot the training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save the model
model.save('animal_classifier.h5')


