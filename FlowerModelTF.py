import os
import cv2
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import json
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

BATCH_SIZE = 32
IMAGE_RES = 224

NUM_TRAIN = 1020
NUM_VALID = 1020
NUM_TEST = 6149
NUM_CLASSES = 102

EPOCHS = 20

MODEL_NAME = './1634414054.h5'


def get_train_images(folder_path: str, resize: tuple):
    image_paths = []
    train_labels = []
    train_images = []

    for folder in folder_path:
        for file in os.listdir(os.path.join(folder_path, folder)):
            if file.endswith('jpg'):
                img_path = os.path.join(folder_path, folder, file)
                image_paths.append(img_path)
                train_labels.append(folder)
                img = cv2.imread(img_path)
                resize_img = cv2.resize(img, resize)
                train_images.append(resize_img)

    return image_paths, train_labels, train_images


def get_oxford_dataset():
    dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
    return dataset['test'], dataset['train'], dataset['validation']


def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label


def prepare_data(train_set, test_set, valid_set):
    return (
        train_set.cache().shuffle(NUM_TRAIN // 4).map(format_image).batch(BATCH_SIZE).prefetch(1),
        valid_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1),
        test_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)
    )


def create_model():
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL,
                                       input_shape=(IMAGE_RES, IMAGE_RES, 3))
    # Freeze the Pre-Trained Model
    feature_extractor.trainable = False
    # Attach a classification head
    model = tf.keras.Sequential([
        feature_extractor,
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model



def do_rest_model(model, train_set, valid_set):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(train_set,
                        epochs=EPOCHS,
                        validation_data=valid_set,
                        callbacks=[early_stopping])


def test_model(model, test_set):
    print("test_loss, test acc", model.evaluate(test_set))


def save_model(model):
    t = time.time()

    export_path_keras = "./{}.h5".format(int(t))
    print(export_path_keras)

    model.save(export_path_keras)


def load_model(export_path):
    return tf.keras.models.load_model(export_path, custom_objects={'KerasLayer': hub.KerasLayer})


def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image


def predict_image(model, image_path, top_k, class_names):
    img = np.asarray(Image.open(image_path))
    proc_test_img = process_image(img)
    prediction = model.predict(np.expand_dims(proc_test_img, axis=0))
    top_vals, top_inds = tf.math.top_k(prediction, top_k)
    print("These are the top propabilities", top_vals.numpy()[0])
    print(top_inds.cpu().numpy()[0])
    top_classes = [class_names[str(value + 1)] for value in top_inds.cpu().numpy()[0]]
    print('Of these top classes', top_classes)
    return top_vals.numpy()[0], top_classes


def vis_predictions(model):
    files = glob.glob(r"C:\Users\Victor\Desktop\trial-images\*.jpg")
    class_names = load_json_data(r"C:\Users\Victor\Desktop\MachineLearning\Datasets\flowers\label_map.json")
    for image_path in files:
        im = Image.open(image_path)
        test_image = np.asarray(im)
        processed_test_image = process_image(test_image)
        probs, classes = predict_image(model, image_path, 5, class_names)
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2)
        ax1.imshow(processed_test_image)
        ax2 = plt.barh(classes[::-1], probs[::-1])
        plt.tight_layout()
        plt.show()


def train_model():
    train_set, valid_set, test_set = prepare_data(*get_oxford_dataset())
    model = create_model()
    do_rest_model(model, train_set, valid_set)
    test_model(model, test_set)
    save_model(model)


def convert_model_tflite(model_path, model, optimize='speed'):
    export_dir = 'saved_model'
    tf.saved_model.save(model, export_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    open("model_tl.tflite", "wb").write(tflite_model)

    if optimize.lower() == 'speed':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    elif optimize.lower() == 'storage':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # reduce the size of a floating point model by quantizing the weights to float16
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()
    # save the quanitized model toa binary file
    open("model_quant_tl.tflite", "wb").write(tflite_quant_model)


def test_model():
    model = load_model(MODEL_NAME)
    vis_predictions(model)

interpreter = tf.lite.Interpreter(model_path='model_quant_tl.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
image = Image.open(r"C:\Users\Victor\Desktop\trial-images\Oxeye-daisy-flower_3859338724_o.jpg")
image = process_image(image)
input_shape = input_details[0]['shape']
input_tensor = np.array(np.expand_dims(image,0), dtype=np.float32)
#set the tensor to point to the input data to be inferred
input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)
#Run the inference
interpreter.invoke()
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
probabilities = np.array(output_data[0])

#results = np.squeeze(output_data)
#top_k = results.argsort()

print(output_data)
label_probs = []
class_names = load_json_data(r"C:\Users\Victor\Desktop\MachineLearning\Datasets\flowers\label_map.json")
for i, probability in enumerate(probabilities):
    label_probs.append([class_names[str(i + 1)], float(probability)])
print(sorted(label_probs, key=lambda element: element[1])[-1])