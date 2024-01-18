import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers

class Model():
    model = None
    model_name: str = None
    is_model_trained: bool = False
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None

    def __init__(self, name: str):
        self.model_name = name
        self.load_model()
        # try:
        #     temp = tf.keras.models.load_model(name+".keras")
        #     self.model = temp
        #     print(f"Model {self.model} loaded")
        # except():
        #     print('Model not found, empty model created')


    def load_data(self, dataset=mnist):
        # Loading the dataset
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        # Normalizing the images
        self.train_images = self.train_images.reshape((60000, 28, 28, 1))
        self.train_images = self.train_images/255.0 #float32

        self.test_images = self.test_images.reshape((10000, 28, 28, 1))
        self.test_images = self.test_images/255.0
        # One-hot encoding the labels
        self.train_labels = tf.keras.utils.to_categorical(self.train_labels)
        self.test_labels = tf.keras.utils.to_categorical(self.test_labels)


    def create_model(self):
        if (self.test_images is None or self.test_labels is None or self.train_images is None or self.train_labels is None):
            raise ValueError("Please load data first")
        if (self.is_model_trained):
            print("Trained model already created")
            return
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu',))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu',))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()
        self.model = model


    def compile_model(self, optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']):
        if (self.model is None):
            raise ValueError("Please create model first")
        if (self.is_model_trained):
            print("Trained model already created")
            return
        self.model.compile(optimizer, loss, metrics)

        
    def train_model(self, epochs=5, batch_size=64):
        if (self.model is None):
            raise ValueError("Please create model first")
        if (self.is_model_trained):
            print("Trained model already created")
            return
        self.model.fit(self.train_images, self.train_labels, epochs=epochs, batch_size=batch_size, verbose=2)
        self.is_model_trained = True


    def evaluate_model(self):
        if (self.model is None):
            raise ValueError("Please create model first")
        if (not self.is_model_trained):
            raise ValueError("Please train model first")
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print("Test accuracy: ", test_acc)


    def save_model(self):
        if (self.model is None):
            raise ValueError("Please create model first")
        if (not self.is_model_trained):
            raise ValueError("Please train model first")
        try:
            self.model.save(self.model_name+".keras")
            print(f"Model {self.model_name} saved")
        except():   
            print("Model not saved. Error saving model")

    
    def load_model(self):
        if (self.model_name is None):
            raise ValueError("Please provide model name")
        try:
            temp_model = tf.keras.models.load_model(self.model_name+".keras")
            self.model = temp_model
            self.is_model_trained = True
            print(f"Model {self.model_name} loaded")
            self.model.summary()
        except:
            print("Model not found, empty model class")
            return