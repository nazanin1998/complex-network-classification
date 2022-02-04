from tensorflow.keras import datasets


class InputClass:
    def __init__(self):
        self._read_data()
        self._summery()

    def _read_data(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0

        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

    def _summery(self):
        print("READ FAKE DATA")
        print("train_images.shape ==> " + str(self.train_images.shape))
        print("train_labels.shape ==> " + str(self.train_labels.shape))
        print("test_images.shape  ==> " + str(self.test_images.shape))
        print("test_labels.shape  ==> " + str(self.test_labels.shape))
        print("num_class          ==> " + str(len(self.class_names)))
