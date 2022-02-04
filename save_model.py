
import tensorflow as tf
from tensorflow import keras

print("Get an example dataset")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


def create_model():
    print("Define a simple sequential model")

    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
model.save('saved_model/my_model')
