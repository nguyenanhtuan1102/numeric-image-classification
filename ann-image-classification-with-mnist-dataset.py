# Import libraries
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Import dataset mnist
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(X_train.shape)
plt.matshow(X_train[0])

# Set labels
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
class_nums = len(class_names)

# Feature scaling
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# Create chanel
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print(X_train.shape)

# Encoding target data
y_train_label = keras.utils.to_categorical(y_train, class_nums)
y_test_label = keras.utils.to_categorical(y_test, class_nums)

# create neural network
model = keras.models.Sequential([
  layers.Flatten(input_shape=(28, 28, 1)),
  layers.Dense(512, activation="relu"),
  layers.Dense(256, activation="relu"),
  layers.Dense(10, activation="softmax")
])

model.summary()

# Compile model
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"]
              )

# Train model
batch_size = 256
epochs = 10
history = model.fit(X_train, y_train_label,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_label)
print("Loss = {0}".format(test_loss))
print("Accuracy = {0}".format(test_acc))

# Making a prediction
y_predicted = model.predict(X_test)
plt.matshow(X_test[89])
np.argmax(y_predicted[89])

# Visualize result
# Graph representing the model's accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("epoch")
plt.ylabel("accuracy/loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Graph representing the model's accuracy and model's loss
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training loss and accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy/loss")
plt.legend(["accuracy", "val_accuracy", "loss", "val_loss"])
plt.show()