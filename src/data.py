from __future__ import print_function
import os
import numpy as np 
import skimage.io as io
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator

# ?
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

'''
The purpose of this function is to parse the variables from a txt file
Returns a list of variables
'''
def parse_data(input_file):
    config = {}
    with(open(input_file, 'r') as f):
        lines = f.readlines()
    for line in lines:
        key, value = line.strip().split("=")
        config[key] = value
    return(config)


'''
The purpose of the function adjustData is to preprocess the input data for a machine learning model that performs image segmentation. 
It normalizes the image and mask, converts the mask to a one-hot encoding if it has multiple classes, and reshapes the mask to the appropriate format.
'''
def adjustData(img, mask, flag_multi_class, num_class):
    img = img / 255.0
    if flag_multi_class:
        mask = mask[:,:,:,0] if len(mask.shape) == 4 else mask[:,:,0]
        mask = np.eye(num_class)[mask.astype('int32')]
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3]))
    else:
        mask = np.asarray(mask > 0.5, dtype=np.float32)
    return img, mask

'''
The purpose of the function trainGenerator is data augmentation. Data augmentation is used to increase the size of the training dataset and reduce overfitting.
'''
def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", flag_multi_class=False, save_to_dir = None, num_class=2, target_size=(256, 256), seed=1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_mask_generator = zip(image_datagen.flow_from_directory(train_path, classes=[image_folder],
                                                                  class_mode=None, color_mode=image_color_mode,
                                                                  target_size=target_size, batch_size=batch_size,
                                                                  seed=seed),
                               mask_datagen.flow_from_directory(train_path, classes=[mask_folder],
                                                                 class_mode=None, color_mode=mask_color_mode,
                                                                 target_size=target_size, batch_size=batch_size,
                                                                 seed=seed))

    while True:
        for (img, mask) in image_mask_generator:
            img, mask = adjustData(img, mask, flag_multi_class, num_class)
            yield (img, mask)


'''
The function returns a generator that yields the preprocessed test images one by one. This generator can be used as input to a model for testing.
'''
def testGenerator(test_path, num_image = 30, target_size = (256,256), flag_multi_class = False, as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


'''
This function takes an input img with labeled pixels
Returns a visualized image where each label is assigned a corresponding
color according to a given dictionary
'''
def labelVisualize(num_class, color_dict, img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return (img_out/255) 


'''
This function save all the predicted images in the ouput directory
'''
def saveResult(save_path, npyfile, flag_multi_class = False, num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),(img*255).astype(np.uint8))

