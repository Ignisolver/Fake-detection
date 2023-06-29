import keras
from keras.utils import plot_model

from PIL import Image
import tensorflow as tf

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class ModelTemplate:
    def __init__(self):
        pass


class SVM:
    def __init__(self):
        self.input_size = int(32 * 32 * 3)
        self.model = None
        self.model_init_restart()

    def model_init_restart(self):
        self.model = svm.SVC()

    def model_train(self, X, y):
        # Split data into 50% train and 50% test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.50, shuffle=False
        )

        # Learn the digits on the train subset
        self.model.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = self.model.predict(X_test)

        print(
            f"Classification report for classifier {self.model}:\n"
            f"{classification_report(y_test, predicted)}\n"
        )

    def model_train_flatten(self, X, y):
        n_samples = len(X)
        X = X.reshape((n_samples, -1))
        self.model_train(X, y)

    def model_eval(self, X, y):
        # Predict the value of the digit on the test subset
        predicted = self.model.predict(X)

        print(accuracy_score(y, predicted))

class CNN:

    def __init__(self):
        self.input_size = int(32 * 32 * 3)
        self.model = None
        self.model_init_restart()

    def model_init_restart(self):
        print("Initializing CNN model")

        inputs = keras.Input(shape=(None, None, 3))
        processed = keras.layers.RandomCrop(width=32, height=32)(inputs)
        conv = keras.layers.Conv2D(filters=6, kernel_size=3)(processed)
        pooling = keras.layers.MaxPooling2D()(conv)
        dense = keras.layers.Dense(2)(pooling)
        feature = keras.layers.Dense(1)(dense)

        full_model = keras.Model(inputs, feature)
        backbone = keras.Model(processed, conv)
        activations = keras.Model(conv, feature)

        self.model = full_model
        self.model.compile(loss='mean_squared_error',
                           optimizer='adam',
                           metrics=['categorical_accuracy'])

    def show_model(self):
        plot_model(self.model, show_shapes=True)

    def model_train(self, X, y, verbose=1):
        self.model.fit(X, y, epochs=150, batch_size=10, verbose=verbose)
        print("Training model finished")

    def model_evaluate(self, x, y, verbose=True):
        self.model.evaluate(x, y, epochs=150, batch_size=10, verbose=verbose)
        print("Evaluating model finished")

    def model_predict(self, x):
        return self.model.predict_classes(x)
