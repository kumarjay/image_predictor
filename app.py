# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import os
import glob
import re

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image



from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np

app= Flask(__name__)

model_path= 'model/model_resnet.h5'

#model= load_model(model_path)
#model._make_predict_function()

from keras.applications.resnet50 import ResNet50
model= ResNet50(weights='imagenet')
model.save('model/model_resnet.h5')

def model_predict(img_path, model):
    img= image.load_img(img_path, target_size=(224, 224))
    
    x= image.img_to_array(img)
    x= np.expand_dims(x, axis=0)
    
    x= preprocess_input(x, mode='caffe')
    
    preds= model.predict(x)
    
    return preds


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f= request.files['file']
        
        basepath= os.path.dirname(__file__)
        file_path= os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        preds= model_predict(file_path, model)
        
        pred_class= decode_predictions(preds, top=1)
        result= str(pred_class[0][0][1])
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)


