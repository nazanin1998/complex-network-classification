from cnn_model.cnn import CNNClass
from cnn_model.fake_data_fetch import InputClass
from cnn_model.ploting import Plotting


def run_cnn_phase():
    # read input data todo replace sarvin's data
    input_class = InputClass()

    # plot some data
    # Plotting.show_plot(fig_size=(10, 10), imgs=input_class.train_images, vertical_axis=5, horizontal_axis=5, range_num=25)

    # create cnn model and train it
    save_model_path = 'saved_model/my_model'

    cnn = CNNClass(train_images=input_class.train_images, train_labels=input_class.train_labels,
                   test_images=input_class.test_images, test_labels=input_class.test_labels, epochs=20)
    if cnn.load_saved_model(save_model_path) is None:
        cnn.create_model()
        fit_result = cnn.fit_model()
        cnn.save_model(save_path=save_model_path)
        Plotting.show_accuracy_plot(accuracy=fit_result.history['accuracy'], label1='accuracy',
                                    val_accuracy=fit_result.history['val_accuracy'], label2='val_accuracy',
                                    x_label='Epoch',
                                    y_label='Accuracy')

    cnn.evaluate_model()

    cnn.predict_class(test_images=input_class.test_images[:20], test_labels=input_class.test_labels[:20],
                      class_names=input_class.class_names)
