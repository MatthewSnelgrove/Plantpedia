
from tflite_model_maker import image_classifier, model_spec
from tflite_model_maker.image_classifier import DataLoader




def get_data(split, folder_path):
    data = DataLoader.from_folder(folder_path)
    return data.split(split)


def create_simple_model(train_data, model_wanted, epochs=5, split=0.9, momentum=None):
    model = None
    if model_wanted is None:
        model = image_classifier.create(train_data, epochs=epochs)
    else:
        model = image_classifier.create(train_data, model_spec=model_spec.get(model_wanted), epochs=epochs,
                                        momentum=momentum)

    return model


def evaluate_model(model, test_data):
    loss, accuracy = model.evaluate(test_data)
    print("loss:", loss, "accuracy:", accuracy)
    return accuracy


def export_model(model, export_path, acc):
    if acc >= 0.91:
        model.export(export_dir=export_path)

def tf_lite_run():
    train_data, test_data = get_data(0.8, r"C:\Users\Victor\Desktop\MachineLearning\Datasets\flowers")
    model, test_data = create_simple_model(train_data, model_wanted=None,
                                           split=0.9, epochs=8, momentum=True)
    accuracy = evaluate_model(model, test_data)
    export_model(model, r"C:\Users\Victor\Desktop\MachineLearning\Trained Files", accuracy)
