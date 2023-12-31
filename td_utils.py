# Packages
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools

# Checks that print out results ----------------------------------------------

# Function to print the number of items in each of the sub-folders of a folder
def print_sf_no(folder_name):
    total_images = 0
    dir_list = next(os.walk('./'+folder_name))[1]

    for folder in dir_list:
        img_list = next(os.walk('./'+folder_name+'/'+folder+"/"))[2]
        total_images += len(img_list)
        print(folder.replace("Tomato___", ""), len(img_list))
    
    print("Total", total_images)

# Function as above but for the sub-sub-folders
def print_sf2_no(folder_name):
    dir_list = next(os.walk('./'+folder_name))[1]

    for folder in dir_list:
        print(folder)
        print_sf_no(folder_name+"/"+folder)
        print("---")

# Function to print the results of checking the size of all of the images
def print_img_size_check(folder_name, img_size: tuple=(256,256,3)):
    dir_list = next(os.walk('./'+folder_name))[1]
    img_shape_new = []

    for folder in dir_list:
        print(folder.replace("Tomato___", ""))
        img_list = next(os.walk('./'+folder_name+'/'+folder+"/"))[2]
        img_count = 0

        for img in img_list:
            img = np.asarray(Image.open('./'+folder_name+'/'+folder+'/'+img))
            if img.shape != img_size:
                img_count += 1
                if img.shape not in img_shape_new:
                    img_shape_new.append(img.shape)
        print("Images of other sizes:",img_count)
        print("---")
    print("Other image dimensions:", img_shape_new)
    
# Image functions ------------------------------------------------------------

# Function to rescale the images
def img_resize_all(folder_name, dest_folder, target_size: tuple=(256,256)):
    dir_list = next(os.walk('./'+folder_name))[1]

    for folder in dir_list:
        img_list = next(os.walk('./'+folder_name+'/'+folder+"/"))[2]

        for image in img_list:
            img = Image.open('./'+folder_name+'/'+folder+'/'+image)
            img_array = np.asarray(img)
            if (img_array.shape[0], img_array.shape[1]) != target_size:
                img = img.resize(target_size, Image.LANCZOS)
                img.save("./"+dest_folder+'/'+folder+"/"+image)
            else:
                img.save("./"+dest_folder+'/'+folder+"/"+image)
        print(folder.replace("Tomato___", ""))

# Graphs --------------------------------------------------------------------

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # Adapted from code used for the UM Data Mining I course

    
def eval_plot(y1, y2, eval_type):
    plt.plot(y1)
    plt.plot(y2)
    plt.title('Model '+eval_type)
    plt.ylabel(eval_type)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


