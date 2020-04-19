from random import shuffle, choice
from PIL import Image
import os
import numpy as np
import matplotlib as plt
import random
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
import pandas as pd
import cv2
import math
from random import randint



def load_rgb_data(IMAGE_DIRECTORY,IMAGE_SIZE, shuffle=True):
    print("Loading images...")
    data = []
    #labels=[]
    directories = next(os.walk(IMAGE_DIRECTORY))[1]
    print(directories)
    for diretcory_name in directories:
        print("Loading {0}".format(diretcory_name))
        file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, diretcory_name)))[2]
        print("we will load [", len(file_names), "] files from [",diretcory_name,"] class ..." )
        for i in range(len(file_names)):

          image_name = file_names[i]
          image_path = os.path.join(IMAGE_DIRECTORY, diretcory_name, image_name)
          if ('.DS_Store' not in image_path):
            #print(image_path)
            label = diretcory_name
            img = Image.open(image_path)
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img=rgbimg
            
            #print(np.array(img).shape)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            #print(np.array(img).shape)
            data.append([np.array(img), label])

    if (shuffle):
      random.shuffle(data)
    training_images = np.array([i[0] for i in data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    training_labels = np.array([i[1] for i in data])
    
    print("File loading completed.")

    return training_images, training_labels


def load_rgb_data_cv(IMAGE_DIRECTORY,IMAGE_SIZE, shuffle=True):
    print("Loading images...")
    data = []
    #labels=[]
    directories = next(os.walk(IMAGE_DIRECTORY))[1]

    for diretcory_name in directories:
        print("Loading {0}".format(diretcory_name))
        file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, diretcory_name)))[2]
        print("we will load [", len(file_names), "] files from [",diretcory_name,"] class ..." )
        for i in range(len(file_names)):
          image_name = file_names[i]
          image_path = os.path.join(IMAGE_DIRECTORY, diretcory_name, image_name)
          #print(image_path)
          label = diretcory_name

          img = cv2.imread(image_path)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))


          #print(np.array(img).shape)
          data.append([np.array(img), label])

    if (shuffle):
      random.shuffle(data)
    training_images = np.array([i[0] for i in data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    training_labels = np.array([i[1] for i in data])
    
    print("File loading completed.")

    return training_images, training_labels


def normalize_data(dataset):
  print("normalize data")
  dataset= dataset/255.0
  return dataset
     


def display_image(trainX, trainY, index=0):
  plt.imshow(trainX[index])
  print ("Label = " + str(np.squeeze(trainY[index])))
  print ("image shape: ",trainX[index].shape)

def display_one_image(one_image, its_label):
  plt.imshow(one_image)
  print ("Label = " + its_label)
  print ("image shape: ",one_image.shape)


def display_dataset_shape(X,Y):
  print("Shape of images: ", X.shape)
  print("Shape of labels: ", Y.shape)
  

def plot_sample_from_dataset(images, labels,rows=5, colums=5, width=8,height=8):

  plt.figure(figsize=(width,height))
  for i in range(rows*colums):
      plt.subplot(rows,colums,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(images[i], cmap=plt.cm.binary)
      plt.xlabel(labels[i])
  plt.show()

def display_dataset_folders(path):
  classes=os.listdir(path)
  classes.sort()
  print(classes)
  

def get_data_distribution(IMAGE_DIRECTORY, output_file=None,plot_stats=True):
    print("Loading images...")
    #list structure to collect the statistics
    stats=[]

    #get all image directories
    directories = next(os.walk(IMAGE_DIRECTORY))[1]

    for diretcory_name in directories:
        print("Loading {0}".format(diretcory_name))
        images_file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, diretcory_name)))[2]
        print("we will load [", len(images_file_names), "] files from [",diretcory_name,"] class ..." )
        for i in range(len(images_file_names)):
          image_name = images_file_names[i]
          image_path = os.path.join(IMAGE_DIRECTORY, diretcory_name, image_name)
          #print(image_path)

          #the class is assumed to be equal to the directorty name
          label = diretcory_name 

          img = Image.open(image_path)
          #convert any image to RGB to make sure that it has three channels
          rgbimg = Image.new("RGB", img.size)
          rgbimg.paste(img)
          img=rgbimg
          
          #get the width and the height of the image in pixels
          width,height = img.size
          #get the size of the image in KB
          size_kb=os.stat(image_path).st_size/1000
          #add the size to a list of sizes to be 
          stats.append([label,os.path.basename(image_name),width,height,size_kb])

    if (output_file is not None):
      #convert the list into a dataframe to make it easy to save into a CSV
      stats_dataframe = pd.DataFrame(stats,columns=['Class','Filename','Width','Height','Size_in_KB'])
      stats_dataframe.to_csv(output_file,index=False)
      print("Stats collected and saved in .",output_file)
    else:
      print("Stats collected");


    return stats


def plot_dataset_distribution (stats, num_cols=5, width=10, height=5, histogram_bins = 10, histogram_range=[0, 1000], figure_padding=4):
  #convert the list into a dataframe
  stats_frame = pd.DataFrame(stats,columns=['Class','Filename','Width','Height','Size_in_KB'])

  #extract the datframe related to sizes only
  list_sizes=stats_frame['Size_in_KB']

  #get the number of classes in the dataset
  number_of_classes=stats_frame['Class'].nunique()
  print(number_of_classes, " classes found in the dataset")

  #create a list of (list of sizes) for each class of images
  #we start by the the sizes of all images in the dataset
  list_sizes_per_class=[list_sizes] 
  class_names=['whole dataset']
  print("Images of the whole dataset have an average size of ", list_sizes.mean())
  
  for c in stats_frame['Class'].unique():
    print("sizes of class [", c, "] have an average size of ", list_sizes.loc[stats_frame['Class']== c].mean())
    #then, we append the sizes of images of a particular class
    list_sizes_per_class.append(list_sizes.loc[stats_frame['Class'] == c])
    class_names.append(c)

  num_rows=math.ceil((number_of_classes+1)/num_cols)
  if (number_of_classes<num_cols):
    num_cols=number_of_classes+1
  fig,axes = plt.subplots(num_rows, num_cols, figsize=(width,height))
    

  fig.tight_layout(pad=figure_padding)
  class_count=0
  if (num_rows==1 or num_cols==1):
    for i in range(num_rows):
      for j in range(num_cols): 
        axes[j+i].hist(list_sizes_per_class[num_cols*i+j], bins = histogram_bins, range=histogram_range)
        axes[j+i].set_xlabel('Image size (in KB)', fontweight='bold')
        axes[i+j].set_title(class_names[j+i] + ' images ', fontweight='bold')
        class_count=class_count+1
        if (class_count==number_of_classes+1):
          break
  
  else:
    for i in range(num_rows):
      for j in range(num_cols): 
        axes[i,j].hist(list_sizes_per_class[num_cols*i+j], bins = histogram_bins, range=histogram_range)
        axes[i,j].set_xlabel('Image size (in KB)', fontweight='bold')
        axes[i,j].set_title(class_names[i] + ' images ', fontweight='bold')
        class_count=class_count+1
        if (class_count==number_of_classes+1):
          break


def reshape_image_for_neural_network_input(image, IMAGE_SIZE=244):
    print ("flatten the image")
    image = np.reshape(image,[IMAGE_SIZE* IMAGE_SIZE*3,1])
    print ("image.shape", image.shape)
    print ("reshape the image to be similar to the input feature vector")
    #image = np.reshape(image,[1,IMAGE_SIZE, IMAGE_SIZE,3]).astype('float')
    image = image.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3).astype('float')
    print ("image.shape", image.shape)
    return image

def plot_loss_accuracy(H, EPOCHS, output_file=None):  
  N = EPOCHS
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
  plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
  plt.title("Training Loss and Accuracy on COVID-19 Dataset")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  if (output_file is not None):
    plt.savefig(output_file)



def draw_accuracy_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def draw_loss_graph(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


#-------------------------------#

def plot_test_image(testX, image_index, predictions_array, true_binary_labels):
    """
        testX: this is the test dataset
        image_index: index of the image that we will plot from the test dataset
        predictions_array: it is the array that contains all the predictions of the test dataset as output of model.predict(testX)
        true_binary_labels: these are the true label expressed as INTEGER values. It does not work with hot-encoding and string labels. 
    """
    single_predictions_array, true_binary_label, test_image = predictions_array, true_binary_labels[image_index], testX[image_index]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(test_image, cmap=plt.cm.binary)

    predicted_binary_label = np.argmax(predictions_array)
    #print ("predicted_binary_label:", predicted_binary_label)
    #print ("true_binary_label:",true_binary_label)
    
    if predicted_binary_label == true_binary_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("predicted: {} {:2.0f}% (true: {})".format(predicted_binary_label,
                                100*np.max(single_predictions_array),
                                true_binary_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label, number_of_classes=3):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.style.use(['classic'])
    plt.grid(False)
    plt.xticks(range(number_of_classes))
    plt.yticks([])
    thisplot = plt.bar(range(number_of_classes), 1, color="#FFFFFF")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    #print(true_label[0])
    #print(predicted_label)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label[0]].set_color('blue')



def plot_sample_predictions(testX, predictions_array, true_binary_labels,number_of_classes=3,num_rows = 10, num_cols = 4, width=None, height=None, is_random=True):
    """
        this method plots a sample of predictions from the test dataset and highlight correct and wrong predictions
        testX: this is the test dataset
        image_index: index of the image that we will plot from the test dataset
        predictions_array: it is the array that contains all the predictions of the test dataset as output of model.predict(testX)
        true_binary_labels: these are the true labels array expressed as INTEGER values. It does not work with hot-encoding and string labels. 
    """
    num_images = num_rows*num_cols

    if (num_images>testX.shape[0]):
      raise Exception("num_rows*num_cols is",(num_rows*num_cols), "must be smaller than number of images in the Test Dataset", testX.shape[0])

    if width is None:
      width=6*num_cols

    if height is None:
      height=2*num_rows

    plt.figure(figsize=(width, height))
    plt.style.use(['seaborn-bright'])
    
    image_index=-1
    for i in range(num_images):
        if (is_random==True):
          image_index=randint(0,testX.shape[0]-1)
        else:
          image_index=image_index+1

        #print(image_index)
        #---------------
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_test_image(testX, image_index, predictions_array[image_index], true_binary_labels)
        #---------------
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(image_index, predictions_array[image_index], true_binary_labels, number_of_classes)
    plt.tight_layout()
    plt.show()







