from kivy.app import App
from kivy.core import text
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from datetime import datetime

import json
import numpy as np
from PIL import Image
import tensorflow as tf

import pandas as pd

df = pd.read_csv('plant_info.csv')

my_coll = []
my_collection = None
curr_plant = "daisy"

IMAGE_RES = 224

class MyCollection(BoxLayout):

    def __init__(self, **kwargs):
        super(MyCollection, self).__init__(**kwargs)


class MainWindow(Screen):

    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
       

class CollectionWindow(Screen):
    def __init__(self, **kwargs):
        super(CollectionWindow, self).__init__(**kwargs)

    

    def update_plants(self):
        my_collec = MyCollection()
        for plant in my_coll:
            plt = BoxLayout(orientation = "horizontal", padding = [self.width * 0.02, self.height * 0.02, self.width * 0.02, self.height * 0.02])
            img = Image(source = "pics/" + plant.common_name + ".jpg", size_hint = (0.15, 0.9), pos_hint = {"x": 0.05, "top": 0.9})
            plt.add_widget(img)
            com_name = Label(text = plant.common_name, size_hint = (0.10, 0.9), pos_hint = {"top": 0.95}, font_size = 12)
            plt.add_widget(com_name)
            sci_name = Label(text = plant.sci_name, size_hint = (0.15, 0.9), pos_hint = {"top": 0.95}, font_size = 12)
            plt.add_widget(sci_name)
            info = Label(text = plant.info, size_hint = (0.4, 0.9), pos_hint = {"top": 0.95}, text_size = (250, 75), font_size = 8)
            plt.add_widget(info)
            water = Label(text = plant.water, size_hint = (0.20, 0.9), pos_hint = {"top": 0.95}, font_size = 12)
            plt.add_widget(water)
            my_collec.add_widget(plt)
            
        self.add_widget(my_collec)

    def update_plant_info(self):
        for plant in my_coll:
            plant.update_info(plant)


class NewPlantWindow(Screen):
    def __init__(self, **kwargs):
        super(NewPlantWindow, self).__init__(**kwargs)
    


class WindowManager(ScreenManager):
    pass
                

class CamWin(Screen):
    def __init__(self, **kwargs):
        super(CamWin, self).__init__(**kwargs)

    def activate_camera(self):
        self.clear_widgets()
        self.add_widget(CamClick())
    

class CamClick(BoxLayout):
    def __init__(self, **kwargs):
        super(CamClick, self).__init__(**kwargs)
    def capture(self):
        #replace with correct plant
        curr_plant = "daisy"
        
        camera = self.ids['camera']
        camera.export_to_png("user_image.png")
        camera.play = False
        print("Captured")

        #Call model with user_image.png
        identify_plant("user_image.png")

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image
        
def identify_plant(image_path):
    interpreter = tf.lite.Interpreter(model_path='model_quant_tl.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = Image.open(image_path)
    image = process_image(image)
    
    input_shape = input_details[0]['shape']
    input_tensor = np.array(np.expand_dims(image,0), dtype=np.float32)

    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)

    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])
    
    label_probs = []
    for i, probability in enumerate(probabilities):
        label_probs.append([class_names[str(i + 1)], float(probability)])
    
    curr_plant = sorted(label_probs, key=lambda element: element[1])[-1][0]
    print(curr_plant)


class Plant(BoxLayout):
    img_src = ObjectProperty(None)
    common_name = ObjectProperty(None)
    sci_name = ObjectProperty(None)
    info = ObjectProperty(None)
    water = ObjectProperty(None)
    def __init__(self, **kwargs):
        super(Plant, self).__init__(**kwargs)
        # self.img_src.source = "daisy.jfif"
        # self.common_name.text = "daisy"
        # print(kwargs.get("common_name"))
        # print(kwargs)
        # self.img_src = kwargs.get("common_name") + ".jfif"
        # self.common_name = kwargs.get("common_name")
        # self.sci_name = kwargs.get("sci_name")
        # self.info = kwargs.get("infO")
        # self.water = kwargs.get("water")




    

# def __init__(self, common_name, sci_name, info, water):
#         self.common_name = common_name
#         self.sci_name = sci_name
#         self.info = info
#         self.water = water

#     def on_remove(self):
#         my_collection.pop(self.common_name)


kv = Builder.load_file("my.kv")





class MyApp(App):
    def build(self):
        # self.add_to_collection()
        return kv
    def add_to_collection(self):
        if not df.index.name == "common name":
            df.set_index("common name", inplace=True)
        row = df.loc[curr_plant]
        kwargs = {"common_name":curr_plant, "sci_name":row[0], "info":row[1], "water":row[2]}
        new_plant = Plant(**kwargs)
        my_coll.append(new_plant)


if __name__ == "__main__":
    MyApp().run()
