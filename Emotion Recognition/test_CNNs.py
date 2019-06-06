# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from face_detection import face_detection_opencv_dnn
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



"""
Read FABO images
Returns: images and emotions
"""
def load_FABO():
    
    # Face Detection
    modelFile = "face_detection/models/opencv_face_detector_uint8.pb"
    configFile = "face_detection/models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    
    data_path = os.getcwd() + '\\CNN body\\FABO_apex_test\\'
    data_dir_list = os.listdir(data_path)
    labels_name = {'Anger' : 0, 'Disgust' : 1, 'Fear' : 2, 'Happiness' : 3,
                   'Sadness' : 4, 'Surprise' : 5, 'Neutral' : 6}
    
    num_classes = 7
    images_data_list_face = []
    images_data_list_body = []
    labels_list = []
    
    for dataset in data_dir_list:
        images_list = os.listdir(data_path + '/' + dataset)
        print('Loading the images of dataset - ' + '{}\n'.format(dataset))
        label = labels_name[dataset]
    
        for image in images_list:
            input_image = cv2.imread(data_path + '/' + dataset + '/'+ image)
    
            # Face Detection
            frame, face_box = face_detection_opencv_dnn.detect_face_OpenCV_DNN(net, input_image)
            
            if face_box != []:
                # Face
                x1, y1, x2, y2 = face_box[0]                
                face = input_image[y1:y2, x1:x2, :]                
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                img_resize = cv2.resize(face_gray, (48, 48))
                img_preprocess = img_resize.astype("float") / 255.0
                img_preprocess = img_preprocess - 0.5
                img_preprocess = img_preprocess * 2.
                images_data_list_face.append(img_preprocess)
                
                # Body
                image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                img_resize = cv2.resize(image_gray, (128, 96))
                img_preprocess = img_resize.astype("float") / 255.0
                img_preprocess = img_preprocess - 0.5
                img_preprocess = img_preprocess * 2.
                images_data_list_body.append(img_preprocess)
                
                # Labels
                labels_list.append(label)
                
    images_data_face = np.array(images_data_list_face)
    images_data_face = np.expand_dims(images_data_face, axis = 4) 
    images_data_face = images_data_face.astype('float32') 
    
    images_data_body = np.array(images_data_list_body)
    images_data_body = np.expand_dims(images_data_body, axis = 4) 
    images_data_body = images_data_body.astype('float32') 
    
    labels = np.array(labels_list)
    labels = np_utils.to_categorical(labels, num_classes)
    labels = labels.astype('uint8')
        
    return images_data_face, images_data_body, labels



"""
Plot confusion matrix.
"""
def plot_confusion_matrix(true_classes, predicted_classes, classes, 
                          normalize = True, cmap = plt.cm.Blues):

    # Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation = 'nearest', cmap = cmap)
    ax.figure.colorbar(im, ax = ax)
    
    ax.set(xticks = np.arange(cm.shape[1]), yticks = np.arange(cm.shape[0]),
           xticklabels = classes, yticklabels = classes, title = 'Confusion matrix',
           ylabel = 'True label', xlabel = 'Predicted label')
           
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")

    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha = "center", va = "center",
                    color = "white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax





images_face, images_body, labels = load_FABO()

# Face
emotion_model_path = "CNN face/Model/mini_XCEPTION.65-0.65.hdf5"
model_face = load_model(emotion_model_path, compile = False)

predictions_face = model_face.predict(images_face)


# Body
emotion_model_path = "CNN body/Model/mini_XCEPTION.35-0.96.hdf5"
model_body = load_model(emotion_model_path, compile = False)

predictions_body = model_body.predict(images_body)


predictions_avg = np.zeros(predictions_face.shape)
predictions_avg = predictions_avg.astype('float32') 

predictions_prod = np.zeros(predictions_face.shape)
predictions_prod = predictions_prod.astype('float32')

for i in range(len(predictions_face)):
   for j in range(len(predictions_face[0])):
       predictions_avg[i][j] = (predictions_face[i][j] + predictions_body[i][j]) / 2.

for i in range(len(predictions_face)):
   for j in range(len(predictions_face[0])):
       predictions_prod[i][j] = predictions_face[i][j] * predictions_body[i][j]
       

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

predictions_avg = np.argmax(predictions_avg, axis = 1)
predictions_prod = np.argmax(predictions_prod, axis = 1)

labels = np.argmax(labels, axis = 1)


# Prediction average fusion
print(classification_report(labels, predictions_avg, digits=3))
print(confusion_matrix(labels, predictions_avg))
print('Accuracy: ' + str(accuracy_score(labels, predictions_avg)))

np.set_printoptions(precision = 2)

labels = labels.astype(int)
predictions_avg = predictions_avg.astype(int)

# Plot normalized confusion matrix
plot_confusion_matrix(labels, predictions_avg, classes = EMOTIONS, normalize = True)

plt.show()


# Prediction product fusion
print(classification_report(labels, predictions_prod, digits=3))
print(confusion_matrix(labels, predictions_prod))
print('Accuracy: ' + str(accuracy_score(labels, predictions_prod)))

np.set_printoptions(precision = 2)

labels = labels.astype(int)
predictions_prod = predictions_prod.astype(int)

# Plot normalized confusion matrix
plot_confusion_matrix(labels, predictions_prod, classes = EMOTIONS, normalize = True)

plt.show()



