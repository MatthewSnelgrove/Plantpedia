from kivy.app import App
from kivy.core import text
from kivy.lang import Builder
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
import pandas as pd

df = pd.read_csv('plant_info.csv')

my_coll = []
my_collection = None


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
            img = Image(source = "pics/" + plant.common_name + ".png", size_hint = (0.15, 0.9), pos_hint = {"x": 0.05, "top": 0.9})
            plt.add_widget(img)
            com_name = Label(text = plant.common_name, size_hint = (0.10, 0.9), pos_hint = {"top": 0.95})
            plt.add_widget(com_name)
            sci_name = Label(text = plant.sci_name, size_hint = (0.15, 0.9), pos_hint = {"top": 0.95})
            plt.add_widget(sci_name)
            info = Label(text = plant.info, size_hint = (0.4, 0.9), pos_hint = {"top": 0.95}, text_size = (250, 75), font_size = 10)
            plt.add_widget(info)
            water = Label(text = plant.water, size_hint = (0.20, 0.9), pos_hint = {"top": 0.95})
            plt.add_widget(water)
            my_collec.add_widget(plt)
            
        self.add_widget(my_collec)

    def update_plant_info(self):
        for plant in my_coll:
            plant.update_info(plant)


class NewPlantWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass
                




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
        return kv
    def add_to_collection(self, plant: str):
        if not df.index.name == "common name":
            df.set_index("common name", inplace=True)
        row = df.loc[plant]
        kwargs = {"common_name":plant, "sci_name":row[0], "info":row[1], "water":row[2]}
        new_plant = Plant(**kwargs)
        my_coll.append(new_plant)

        



if __name__ == "__main__":
    MyApp().run()