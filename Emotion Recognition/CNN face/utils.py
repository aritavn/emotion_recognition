# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cv2
import os
from keras.utils import np_utils


"""
Read csv file and convert pixels into array
Returns: faces and emotions
"""
def load_FER2013():
    
    data = pd.read_csv('fer2013/fer2013.csv')
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))
        
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    
    return faces, emotions



"""
Read FABO face images
Returns: faces and emotions
"""
def load_FABO():
    
    data_path = os.getcwd() + '/FABO_apex_phase_faces'
    data_dir_list = os.listdir(data_path)
    labels_name = {'Anger' : 0, 'Disgust' : 1, 'Fear' : 2, 'Happiness' : 3,
                   'Sadness' : 4, 'Surprise' : 5, 'Neutral' : 6}
    num_classes = 7
    images_data_list = []
    labels_list = []
    
    for dataset in data_dir_list:
    	images_list = os.listdir(data_path + '/' + dataset)
    	#print('Loading the images of dataset - ' + '{}\n'.format(dataset))
    	label = labels_name[dataset]
        
    	for image in images_list:
    		input_image = cv2.imread(data_path + '/' + dataset + '/'+ image, 0)
    		images_data_list.append(input_image)
    		labels_list.append(label)

    images_data = np.array(images_data_list)
    images_data = np.expand_dims(images_data, axis = 4) 
    images_data = images_data.astype('float32') 

    labels = np.array(labels_list)
    labels = np_utils.to_categorical(labels, num_classes)
    labels = labels.astype('uint8')
    
    return images_data, labels



"""
Pre-process images by scaling them between [-1, 1] which is a better range for
neural network models.
Returns: faces
"""
def preprocess_input(faces):
    
   faces = faces / 255.
   faces = faces - 0.5
   faces = faces * 2.
   
   return faces




