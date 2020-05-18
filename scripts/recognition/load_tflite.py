import tensorflow as tf
import cv2
import numpy as np
import argparse


class TFLiteModel(object):
    def __init__(self, model):
        # create tf lite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __str__(self):
        return "TFLiteModel"

    def get_outputs(self, images):
        """
        Get output from network for images.
        :param images: array of images
        :return: results from neural network
        """
        outputs = []

        for input_img in images:
            self.interpreter.set_tensor(self.input_details[0]["index"], np.asarray([input_img], dtype=np.float32))
            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
            outputs.append(output_data[0])

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

    model = TFLiteModel(args.model)
    outputs = model.get_outputs([image])

    with open(args.labels) as f:
        labels = [line.rstrip() for line in f]
        for i, label in enumerate(labels):
            print("Label: '" + label + "' with probability: " + str(float(outputs[0][0][i])))
