import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models


class CNNClass:
    def __init__(self, train_images, train_labels, test_images, test_labels, epochs, ):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.epochs = epochs
        self.model = models.Sequential()

    def load_saved_model(self, save_path):
        try:
            self.model = tf.keras.models.load_model(save_path)
            return self.model
        except:
            print("no saved model")
            return None

    def create_model(self):
        self.model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

    def fit_model(self):
        fit_result = self.model.fit(self.train_images, self.train_labels, epochs=self.epochs,
                                    validation_data=(self.test_images, self.test_labels))
        return fit_result

    def save_model(self, save_path):
        self.model.save(save_path)

    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print("EVALUATIONS ==> Accuracy: " + str(test_acc) + "  -   loss: " + str(test_loss))

    def predict_class(self, test_images, test_labels, class_names):
        all_sample = len(test_images)
        predict_x = self.model.predict(test_images[:all_sample])
        classes_x = np.argmax(predict_x, axis=1)

        correct_detected = 0
        for num, data in enumerate(test_images[:all_sample]):
            detected_label = class_names[classes_x[num]]
            real_label = class_names[test_labels[num][0]]
            print(str(num) + "==> real label : " + real_label + "  , detected label: " + detected_label)
            if detected_label == real_label:
                correct_detected += 1

        print("correct_detected : " + str(correct_detected) + "/" + str(all_sample))
