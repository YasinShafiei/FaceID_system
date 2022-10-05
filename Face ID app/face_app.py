"""
Face ID app GUI
Using Kivy
"""
# import all libraries
from kivy.app import *
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
import cv2
import tensorflow as tf
from keras.models import load_model
from distance_layer import DistanceLayer
import keras
import os
import numpy as np
import keras

class FaceApp(App):
    title="Face ID"
    def build(self):
        # define the main screen
        root = BoxLayout(orientation='vertical')

        # define widgets 
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify person", on_press = self.verify, size_hint=(0.4, .1), pos_hint={'center_x':.5}, background_color=(0, 1, 0, 1))
        self.verification_label = Label(text="Waiting for verification", size_hint=(1, .1), bold=True, font_size=15)

        # add widgets to layout
        root.add_widget(self.web_cam)
        root.add_widget(self.button)
        root.add_widget(self.verification_label)

        # load nmodel
        loss = tf.losses.BinaryCrossentropy()
        self.model = load_model("face_id_model.h5", custom_objects={"DistanceLayer":DistanceLayer, "BinaryCrossentropy":loss})

        # Setup video capture device
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return root

    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_images', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_images', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        # Set verification text 
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'
        self.verification_label.color = (0, 1, 0, 1) if self.verification_label.text == "Verified" else (1, 0, 0, 1)

        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        
        return results, verified


if __name__ == "__main__":
    FaceApp().run()