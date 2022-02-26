# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:51:45 2021

@author: Nadian
"""

import tensorflow as tf


saved_model = tf.keras.models.load_model('D:/TextClassification/mymodel')


string = 'hi'

string = string.replace(' ','')




saved_model.predict(string)