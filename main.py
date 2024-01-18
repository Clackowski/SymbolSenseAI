import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import Model
import cv2

def run_model(model):
    model.load_data()
    model.create_model()
    model.compile_model()
    model.train_model()
    model.evaluate_model()
    model.save_model()

def make_model():
    model = Model("first_mnist_model")
    run_model(model)

def load_model():
    test_load_model = Model("first_mnist_model")
    test_load_model.load_data()
    test_load_model.evaluate_model()

model_emnist = tf.keras.models.load_model("emnist_v7.h5")
model_mnist = tf.keras.models.load_model('first_mnist_model.keras')

def predict_num():
    for i in range(10):
        img = cv2.imread(f'num_images/{i}.png')[:,:,0]
        img=np.array([img])
        output = model_mnist.predict(img)
        print(output)
        print("Prediction: ", np.argmax(output))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

def predict_char():
    for i in ["a", "b", "c", "x", "y", "z"]:
        img = cv2.imread(f'{i}.png')[:,:,0]
        img=np.array([img])
        output = model_emnist.predict(img)
        print(output)
        print("Prediction: ", chr(ord('`') + np.argmax(output)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

predict_num()