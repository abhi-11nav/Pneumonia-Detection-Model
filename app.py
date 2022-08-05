#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:15:11 2022

@author: abhinav
"""

from __future__ import division, print_function

import sys 
import os
import numpy as np 

# Keras libararies
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 

# Flask utils 
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# defining a flask app 

app = Flask(__name__,template_folder="template")

# Loading the model saved with keras 
model = load_model('pneumonia_model.h5')


def model_predict(path, model):
    img = image.load_img(path, target_size=(224,224))
    
    # Preprocessing the image
    img = image.img_to_array(img)
    
    img /= 255
        
    img = np.expand_dims(img, axis=0)
    
    img = preprocess_input(img)
    
    prediction = model.predict(img)
    
    prediction = np.argmax(prediction, axis=1)
    
    if prediction==0:
        prediction = "The person is infected with Pneumonia"
    else:
        prediction = "The person is not infected with Pneumonia"
    
    return prediction

@app.route("/",methods=["GET","POST"])
def index():
    return render_template("/index.html")

@app.route("/predict",methods=["GET","POST"])
def upload():
    if request.method == "POST":
        target = request.files["file"]
        
        basepath = os.path.dirname(__file__)
        
        file_path = os.path.join(basepath,'uploads',secure_filename(target.filename))
        target.save(file_path)

        prediction = model_predict(file_path, model)       
    
    return prediction



if __name__ == "__main__":
    app.run(debug=True)
        


        
    
    
    
    
    
    
    
    
    