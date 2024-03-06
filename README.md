# OVERVIEW

![6](https://github.com/tuanng1102/ann-image-classification-with-mnist-dataset/assets/147653892/ef6d64e4-b12c-4293-81e8-655962fbab90)


Artificial Neural Network for MNIST Image Classification (Fashion MNIST)
This code implements an artificial neural network (ANN) to classify images from the Fashion MNIST dataset, which consists of labeled images of various clothing items. The code covers the following steps:

### 1. Import Libraries:

numpy: Provides numerical operations.
tensorflow.keras: Provides deep learning functionalities.
matplotlib.pyplot: Used for visualization.

``` bash
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
```

### 2. Load Dataset:

The keras.datasets.mnist.load_data function loads the Fashion MNIST dataset, splitting it into training and testing sets.
X_train and X_test represent the image data.
y_train and y_test represent the corresponding labels (categories) for each image.

``` bash
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
```

### 3. Preprocessing:

Feature Scaling:
Converts pixel values from the range [0, 255] to [0, 1] for better training performance.
Achieved by dividing each pixel value by 255 and converting to float32 data type.
Adding Channel Dimension:
Reshapes the data to add a channel dimension, which is necessary for convolutional neural networks (CNNs).

``` bash
# Feature scaling
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

# Adding Channel Dimension
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
```

### 4. Label Encoding:

The keras.utils.to_categorical function converts the integer labels (e.g., 0, 1, 2) into one-hot encoded vectors.
This allows the model to handle multi-class classification problems.

``` bash
y_train_label = keras.utils.to_categorical(y_train, class_nums)
y_test_label = keras.utils.to_categorical(y_test, class_nums)
```

### 5. Create the Neural Network:

A sequential model is created using keras.models.Sequential.
The model consists of the following layers:
Flatten: Flattens the 2D image data into a 1D vector.
Dense (2 layers): Fully connected layers with 512 and 256 neurons, respectively.
Dense (output layer): Fully connected layer with 10 neurons (one for each class) and a softmax activation function for probability distribution.

``` bash
model = keras.models.Sequential([
  layers.Flatten(input_shape=(28,28,1)), #(28,28,1) => 1D (784)
  layers.Dense(512, activation="relu"),
  layers.Dense(256, activation="relu"),
  layers.Dense(10, activation="softmax")
])

model.summary()
```

### 6. Compile the Model:

Configures the model for training by specifying the optimizer, loss function, and metrics.
Optimizer: "rmsprop"
Loss function: "categorical_crossentropy" (suitable for multi-class classification)
Metrics: "accuracy" to track the model's performance

``` bash
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"]
              )
```

### 7. Train the Model:

Trains the model on the training data (X_train, y_train_label) for a specified number of epochs (epochs) and batch size (batch_size).
validation_split is used to split a portion of the training data for validation during training.

``` bash
batch_size = 256
epochs = 10
history = model.fit(X_train, y_train_label,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1)
```
The training process is stored in the history object.

![4](https://github.com/tuanng1102/ann-image-classification-with-mnist-dataset/assets/147653892/e8e7bf60-c223-42b2-8723-7806c55fa992)


### 8. Evaluate the Model:

Evaluates the model's performance on the testing data (X_test, y_test_label) and prints the loss and accuracy.

``` bash
test_loss, test_acc = model.evaluate(X_test, y_test_label)
print("Loss = {0}".format(test_loss))
print("Accuracy = {0}".format(test_acc))
```

![5](https://github.com/tuanng1102/ann-image-classification-with-mnist-dataset/assets/147653892/7edcd296-3466-494d-86f9-f6674a7f430d)

### 9. Make Predictions:

Uses the trained model to predict the class labels for individual images from the testing set (X_test).
The predicted class for the 89th image in the testing set is displayed.

``` bash
y_predicted = model.predict(X_test)
plt.matshow(X_test[89])
print("Hình ảnh trên biểu thị số ", np.argmax(y_predicted[89]))
```

![3](https://github.com/tuanng1102/ann-image-classification-with-mnist-dataset/assets/147653892/bff95277-3a3c-4dd7-9f45-437e00337416)

``` bash
# result
Hình ảnh trên hiển thị số 1
```
### 10. Visualize Results:

Plots two graphs to visualize the model's training process:
Accuracy: Shows the training and validation accuracy curves.
Loss: Shows the training and validation loss curves.

``` bash
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
```
![1](https://github.com/tuanng1102/ann-image-classification-with-mnist-dataset/assets/147653892/805241cb-38ab-4ae7-8e2a-268997b668e9)
![2](https://github.com/tuanng1102/ann-image-classification-with-mnist-dataset/assets/147653892/411bda39-fb5e-4fa0-92f6-7cf08fc02d7a)
