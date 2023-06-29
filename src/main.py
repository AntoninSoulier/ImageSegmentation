from model import *
from data import *
import tensorflow as tf
import keras as ks

# Get variables from txt file
config = parse_data("config_variables.txt")

# Set variables
mode = config['mode']

batch_size = int(config['batch_size'])

train_dir = "data\membrane\\train"
test_dir = "data\membrane\\test"
output_dir = "data\membrane\\output"
hdf5_dir = "data\\membrane\\hdf5\\unet_membrane.hdf5"

image_folder = 'image'
label_folder = 'label'

if(config['pretrained_weights'] == "None") :
    pretrained_weights = None
else :
    pretrained_weights = str(config['pretrained_weights'])

padding = str(config['padding'])

# Number of pooling operations
number_pooling_operations = int(config['number_pooling_operations'])

# Number of layers before each pooling
number_layers = int(config['number_layers'])

# Type of block
block_type = config['block']

# Set the data generator argument. 
data_gen_args = dict(rotation_range = 0.2,
                    width_shift_range = 0.05,
                    height_shift_range = 0.05,
                    shear_range = 0.05,
                    zoom_range = 0.05,
                    horizontal_flip = True,
                    fill_mode = "nearest")

def GPU():
    return(len(tf.config.list_physical_devices('GPU'))>0)

def main():

    # Enable Keras model to use the GPU for faster training
    if(GPU()):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        ks.backend.backend()

    # Train
    if(mode == "train"):
        myGene = trainGenerator(batch_size, train_dir, image_folder, label_folder, data_gen_args, save_to_dir = None)
        model = unet(pretrained_weights, number_pooling_operations, number_layers, padding, block_type)
        model_checkpoint = ModelCheckpoint(hdf5_dir, monitor='loss',verbose=1, save_best_only=True)
        model.fit(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint]) #Validation set ne doit pas être le même que train et test

    # Predict
    elif(mode=="predict"):
        model = load_model(hdf5_dir)
        testGene = testGenerator(test_dir)
        results = model.predict(testGene, 30, verbose=1)
        saveResult(output_dir, results)
    
    else:
        print("Invalid Argument")
        
    return 0

main()