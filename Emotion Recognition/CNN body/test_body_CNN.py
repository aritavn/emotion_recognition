# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



"""
Read FABO images
Returns: images and emotions
"""
def load_FABO():
    
    data_path = os.getcwd() + '\\FABO_apex_test\\'
    data_dir_list = os.listdir(data_path)
    labels_name = {'Anger' : 0, 'Disgust' : 1, 'Fear' : 2, 'Happiness' : 3,
                   'Sadness' : 4, 'Surprise' : 5, 'Neutral' : 6}
    
    num_classes = 7
    images_data_list = []
    labels_list = []
    
    for dataset in data_dir_list:
        images_list = os.listdir(data_path + '/' + dataset)
        print('Loading the images of dataset - ' + '{}\n'.format(dataset))
        label = labels_name[dataset]
    
        for image in images_list:
            input_image = cv2.imread(data_path + '/' + dataset + '/'+ image)
            image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            img_resize = cv2.resize(image_gray, (128, 96))
            img_preprocess = img_resize.astype("float") / 255.0
            img_preprocess = img_preprocess - 0.5
            img_preprocess = img_preprocess * 2.

            images_data_list.append(img_preprocess)
            labels_list.append(label)
                
    images_data = np.array(images_data_list)
    images_data = np.expand_dims(images_data, axis = 4) 
    images_data = images_data.astype('float32') 
    
    labels = np.array(labels_list)
    labels = np_utils.to_categorical(labels, num_classes)
    labels = labels.astype('uint8')
        
    return images_data, labels



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





images, labels = load_FABO()

# Trained emotion model
emotion_model_path = "Model/mini_XCEPTION.35-0.96.hdf5"
model = load_model(emotion_model_path, compile = False)
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

predictions = model.predict(images)
predictions = np.argmax(predictions, axis = 1)

labels = np.argmax(labels, axis = 1)

print(classification_report(labels, predictions, digits=3))
print(confusion_matrix(labels, predictions))
print('Accuracy: ' + str(accuracy_score(labels, predictions)))

np.set_printoptions(precision = 2)

labels = labels.astype(int)
predictions = predictions.astype(int)

# Plot normalized confusion matrix
plot_confusion_matrix(labels, predictions, classes = EMOTIONS, normalize = True)

plt.show()

