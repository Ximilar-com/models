## Example Models

### Ximilar Custom Recognition

* TFLite model [download](https://github.com/Ximilar-com/models/releases/download/1.0/model.zip) - Trained simple tagger with (Animal, Cat, Dog)

Download the model to the scripts/recognition folder, unzip it. Test tflite model on the specific image:

    python load_tflite.py --model model/tflite/model.tflite --labels model/labels.txt --image image.png

Test keras saved model:

    python load_tflite.py --model model/tflite/model.tflite --labels model/labels.txt --image image.png