# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:48:21 2022

@author: Padmesh

User Interface for Eye and Scan classification using Deep Learning
"""
#Importing libraries
import streamlit as st
import tensorflow
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

#Function to convert jpg/jpeg format file into array format compatible with model
def model_image_gen_novel(img_1):
    data_n = np.ndarray(shape=(1, 28, 28, 3), dtype=np.float32)

    image = img_1

    #image sizing
    size = (28, 28)

    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array

    image_a = np.asarray(image)
    # Normalize the image
    if image_a.shape == (28,28,4):
        image_a = image_a[:,:,:3]

    normalized_img_arr = (image_a.astype(np.float32) / 255.0)
    
    # Load the image into the array

    data_n[0] = normalized_img_arr

    return data_n
    


#Function to convert jpg format file into array format compatible with model
def model_image_gen(img):
    
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 32, 32, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (32, 32)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_a = np.asarray(image)
    if image_a.shape == (32,32,4):
        image_a = image_a[:,:,:3]
    
    # Normalize the image
    normalized_img_arr = (image_a.astype(np.float32) / 255.0)

    # Load the image into the array
    data[0] = normalized_img_arr
    return data

#Function to predict retina scan on different model

def multi_conv_retina_model(img):
    #Loading the model
    model = keras.models.load_model('multi_conv_model_retina (1).h5')
    
    # predicting the image
    prediction = model.predict(img)
    acc = np.amax(prediction)
    retina_class = np.argmax(prediction)
    return acc, retina_class, prediction

def simplistic_retina_model(img):
    #Loading the model
    model = keras.models.load_model('simplistic_model_retina_dropout (1).h5')
    
    # predicting the image
    prediction = model.predict(img)
    acc = np.amax(prediction)
    retina_class = np.argmax(prediction)
    return acc, retina_class, prediction   
    
def resnet_retina_model(img):
    #Loading the model
    model = keras.models.load_model('resnet_retina_model.h5')
    
    #predicting the image 
    prediction = model.predict(img)
    acc = np.amax(prediction)
    retina_class = np.argmax(prediction)
    return acc,retina_class, prediction

def alexnet_retina_model(img):
    #load the model
    model = keras.models.load_model('alexnet_retina_model.h5')
    
    #predicting the image
    prediction = model.predict(img)
    acc = np.amax(prediction)
    retina_class = np.argmax(prediction)
    return acc, retina_class, prediction

#Function to predict Skin scan on different model

def multi_conv_derma_model(img):
    #load the model
    model = keras.models.load_model('multi_conv_model_derma (1).h5')
    
    #predicting the image 
    prediction = model.predict(img)
    acc = np.amax(prediction)
    skin_class = np.argmax(prediction)
    return acc,skin_class, prediction

def simplistic_derma_model(img):
    #load the model
    model = keras.models.load_model('simplistic_model_derma_dropout (1).h5')
    
    #predicting the image
    prediction = model.predict(img)
    acc = np.amax(prediction)
    skin_class = np.argmax(prediction)
    return acc, skin_class, prediction

def alexnet_derma_model(img):
    #load the model
    model = keras.models.load_model('alexnet_derma_model.h5')
    
    #predicting the image
    prediction = model.predict(img)
    acc =np.amax(prediction)
    skin_class = np.argmax(prediction)
    return acc, skin_class, prediction

def resnet_derma_model(img):
    #load the model
    model = keras.models.load_model('resnet_derma_model.h5')
    
    #predicting the image
    prediction = model.predict(img)
    acc =np.amax(prediction)
    skin_class = np.argmax(prediction)
    return acc, skin_class, prediction


#User interface Title    
st.title("Eye and Derma Image Classification using Deep Learning")
    
#Uploading Eye/Skin scan 
uploaded_file = st.sidebar.file_uploader("Choose a Eye or Derma Scan ...", type=["png","jpg","jpeg"])

#Drop down to select Eye or Skin Classifiction
dropdown_type = st.sidebar.selectbox(
     'Select the type: ',
     ('Retina', 'Derma'))

st.sidebar.write('')

#Checking if image uploaded and dropdown selected
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=False)
    
    model_img = model_image_gen(image)
    model_n_img = model_image_gen_novel(image)
    #Model prediction for Retina scan
    if dropdown_type == 'Retina':
        
            #Drop down to select Model option
            dropdown_model = st.sidebar.selectbox(
                 'Select the Model for Classification: ',
                 ( 'Simplistic','ResNet','Multi_Conv','AlexNet'))
            st.sidebar.write('Best model is shown by default.\n Different model selection avilable in dropdown')
            
            if st.sidebar.button('PREDICT'):
                
                if dropdown_model == 'ResNet':
                    resnet_retina_label = resnet_retina_model(model_img)
                    st.write('Model Accuracy : 0.50')
                    st.write('Class predictions probablity :', round(resnet_retina_label[0],3))
                    st.write('Retina Class predicted: ', resnet_retina_label[1])
                    st.write('retina prediction array: ', resnet_retina_label[2])
            
                elif dropdown_model == 'Multi_Conv':
                    st.write('Model Accuracy : 0.47')
                    multi_conv_retina_label = multi_conv_retina_model(model_n_img)
                    st.write('Class prediction probablity: ', round(multi_conv_retina_label[0],3))
                    st.write('Retina Class prediction: ', multi_conv_retina_label[1])
                    st.write('Retina prediction array: ', multi_conv_retina_label[2])
            
                elif dropdown_model == 'Simplistic':
                    st.write('Model Accuracy : 0.63')
                    simplistic_retina_label = simplistic_retina_model(model_n_img)
                    st.write('Class prediction probablity: ', round(simplistic_retina_label[0],3))
                    st.write('Retina Class prediction: ', simplistic_retina_label[1])
                    st.write('Retina prediction array: ', simplistic_retina_label[2])
                
                elif dropdown_model == 'AlexNet':
                    alexnet_retina_label = alexnet_retina_model(model_img)
                    st.write('Model Accuracy : 0.45')
                    st.write('Class prediction probablity: ', round(alexnet_retina_label[0],3))
                    st.write('Retina class prediction: ', alexnet_retina_label[1])
                    st.write('Retina prediction array: ', alexnet_retina_label[2])
            
    #Model prediction for Derma scan  
    elif dropdown_type == 'Derma':
            #Drop down to select Model option
            dropdown_model = st.sidebar.selectbox(
                 'Select the Model for Classification: ',
                 ('AlexNet', 'Multi_Conv', 'Simplistic', 'ResNet'))
            st.sidebar.write('Best model is shown by default.\n Different model selection avilable in dropdown')
            
            if st.sidebar.button('PREDICT'):
                    
                if dropdown_model == 'ResNet':
                    resnet_derma_label = resnet_derma_model(model_img)
                    st.write('Model Accuracy : 0.67')
                    st.write('Class prediction probablity: ', round(resnet_derma_label[0],3))
                    st.write('Derma Class prediction: ', resnet_derma_label[1])
                    st.write('Derma prediction array: ', resnet_derma_label[2])
            
                
                elif dropdown_model == 'Multi_Conv':
                    st.write('Model Accuracy : 0.71')
                    multi_conv_derma_label = multi_conv_derma_model(model_n_img)
                    st.write('Class prediction probablity: ', round(multi_conv_derma_label[0],3))
                    st.write('Derma Class prediction: ', multi_conv_derma_label[1])
                    st.write('Derma prediction array: ', multi_conv_derma_label[2])
                
                elif dropdown_model == 'Simplistic':
                    st.write('Model Accuracy : 0.75')
                    simplistic_derma_label = simplistic_derma_model(model_n_img)
                    st.write('Class prediction probablity: ', round(simplistic_derma_label[0],3))
                    st.write('Derma Class prediction: ', simplistic_derma_label[1])
                    st.write('Derma prediction array: ', simplistic_derma_label[2])
                
            
                elif dropdown_model == 'AlexNet':
                    alexnet_derma_label = alexnet_derma_model(model_img)
                    st.write('Model Accuracy : 0.74')
                    st.write('Class prediction probablity: ', round(alexnet_derma_label[0],3))
                    st.write('Derma Class prediction: ', alexnet_derma_label[1])
                    st.write('Derma prediction: ', alexnet_derma_label[2])
                

        
        
  
        
        


