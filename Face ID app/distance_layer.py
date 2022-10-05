# import all libraries
import tensorflow as tf
from keras.layers import Layer

class DistanceLayer(Layer):
    """
    Calculate the distance between between anchor image, Positive / Negative
    Somehow compare the input image and the VERIFICATION image 
    """
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, verification_embedding):
        distance = tf.math.abs(input_embedding - verification_embedding)

        return distance