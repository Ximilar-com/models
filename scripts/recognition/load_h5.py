import tensorflow as tf
import cv2
import numpy as np
import argparse

from tensorflow.python.keras.backend import set_session


@tf.function
def preprocess_tf(x):
    """
    Preprocessing for Keras (MobileNetV2, ResNetV2).
    :param x: np.asarray([image, image, ...], dtype="float32") in RGB
    :return: normalized image tf style (RGB)
    """
    batch, height, width, channels = x.shape
    x = tf.cast(x, tf.float32)

    # ! do not use tf.constant as they are not right now serializable when saving model for .h5
    # ! https://stackoverflow.com/questions/47066635/checkpointing-keras-model-typeerror-cant-pickle-thread-lock-objects
    # mean_tensor = tf.constant([127.5, 127.5, 127.5], dtype=tf.float32, shape=[1, 1, 1, 3], name="mean")
    # one_tensor = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 1, 3], name="one")

    mean_tensor = np.asarray([[[[127.5, 127.5, 127.5]]]], dtype=np.float32)
    one_tensor = np.asarray([[[[1.0, 1.0, 1.0]]]], dtype=np.float32)

    x = tf.keras.backend.reshape(x, (-1, 3))
    result = (x / mean_tensor) - one_tensor
    return tf.keras.backend.reshape(result, (-1, height, width, channels))


@tf.function
def preprocess_caffe(x):
    """
    Preprocessing for Keras (VGG, ResnetV1).
    ! This works only for channels_last
    :param x: np.asarray([image, image, ...], dtype="float32") in RGB
    :return: normalized image vgg style (BGR)
    """
    batch, height, width, channels = x.shape
    x = tf.cast(x, tf.float32)
    r, g, b = tf.split(x, 3, axis=3)
    x = tf.concat([b, g, r], 3)
    mean_tensor = np.asarray([[[[103.939, 116.779, 123.68]]]], dtype=np.float32)
    result = x - mean_tensor
    return tf.keras.backend.reshape(result, (-1, height, width, channels))


class PreprocessTFLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_tf", **kwargs):
        super(PreprocessTFLayer, self).__init__(name=name, **kwargs)
        self.preprocess = preprocess_tf

    def call(self, input):
        return self.preprocess(input)

    def get_config(self):
        config = super(PreprocessTFLayer, self).get_config()
        return config


class PreprocessCaffeLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_cafe", **kwargs):
        super(PreprocessCaffeLayer, self).__init__(name=name, **kwargs)
        self.preprocess = preprocess_caffe

    def call(self, input):
        return self.preprocess(input)

    def get_config(self):
        config = super(PreprocessCaffeLayer, self).get_config()
        return config


class KerasModel(object):
    def __init__(self, model, eager_execution=True, processing_unit="cpu:0"):
        self.processing_unit = processing_unit
        self.eager = eager_execution
        
        if not self.eager:
            tf.compat.v1.disable_eager_execution()
            self.graph = tf.Graph()
            self.session = tf.compat.v1.Session(graph=self.graph)
            with self.graph.as_default() as graph:
                with graph.device("/" + self.processing_unit):
                    set_session(self.session)
                    self.model = self.load_model(model, ["probs"])
        else:
            with tf.device("/" + self.processing_unit):
                self.model = self.load_model(model, ["probs"])

    def __str__(self):
        return "KerasModel"

    def load_model(self, path, outputs):
        """
        Load keras model.
        :param path: local path
        :return: object of type tf.keras.Model
        """
        model = tf.keras.models.load_model(path, custom_objects=self.preprocess_layer())
        model = tf.keras.Model(
            inputs=model.input, outputs=[model.get_layer(layer_name).output for layer_name in outputs]
        )
        model.trainable = False
        return model

    def preprocess_layer(self):
        return {"PreprocessTFLayer": PreprocessTFLayer, "PreprocessCaffeLayer": PreprocessCaffeLayer}

    def get_outputs(self, images):
        """
        Get output from network.
        :param data: dictionary of { 'output_name' : str, 'input' : [np.array] }
        :return: outputs
        """
        if not self.eager:
            with self.graph.as_default() as graph:
                with graph.device("/" + self.processing_unit):
                    set_session(self.session)
                    outputs = self.model.predict_on_batch(np.asarray(images))
        else:
            with tf.device("/" + self.processing_unit):
                outputs = self.model.predict_on_batch(np.asarray(images))

        if self.eager:
            if isinstance(outputs, list):
                return [output for output in outputs]
            return [np.asarray(outputs, dtype=np.float32)]

        if isinstance(outputs, list):
            return outputs
        return [np.asarray(outputs, dtype=np.float32)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="path to the tflite model")
    parser.add_argument('--labels', type=str, help="path to the labels file")
    parser.add_argument('--image', type=str, default="image.png", help="path to the image file")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    model = KerasModel(args.model)
    outputs = model.get_outputs([image])

    with open(args.labels) as f:
        labels = [line.rstrip() for line in f]
        for i, label in enumerate(labels):
            print("Label: '" + label + "' with probability: " + str(float(outputs[0][0][i])))
