from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler # Used in main.py

def num_filters(num_pools):
    filters = []
    value = 64
    filters.append(value)
    for _ in range(num_pools):
        value *= 2
        filters.append(value)
    return(filters)

def resnet_block(filters, i, padding, conv):
    tmp_output = Conv2D(filters[i], 3, padding=padding, kernel_initializer='he_normal')(conv)
    tmp_output = BatchNormalization()(tmp_output)
    tmp_output = Activation('relu')(tmp_output)
    tmp_output = Conv2D(filters[i], 3, padding=padding, kernel_initializer='he_normal')(tmp_output)
    tmp_output = BatchNormalization()(tmp_output)
    tmp_output = Activation('relu')(tmp_output)
    tmp_output = Add()([conv, tmp_output])
    conv = tmp_output

def unet(pretrained_weights, num_pools, num_layers, padding, block_type, input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Define number of filters for each layer (in function of the number of poolings)
    filters = num_filters(num_pools)
    
    # Define the convolutional layers for each level of the U-Net
    conv_layers = []
    pool_layers = []

    for i in range(num_pools):

        # Downsampling path
        if i == 0:
            conv = Conv2D(filters[i], 3, activation='relu', padding=padding, kernel_initializer='he_normal')(inputs) # Usage of activation function ReLU helps to prevent the exponential growth in the computation required to operate the neural network.
        else:
            conv = Conv2D(filters[i], 3, activation='relu', padding=padding, kernel_initializer='he_normal')(pool_layers[-1])
        
        #Case of ResNet block
        if(block_type=='ResNet'):
            resnet_block(filters, i, padding, conv)

        for _ in range(num_layers-1):
            conv = Conv2D(filters[i], 3, activation='relu', padding=padding, kernel_initializer='he_normal')(conv)
        
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        conv_layers.append(Conv2D(filters[i], 3, activation='relu', padding=padding, kernel_initializer='he_normal')(pool))
        pool_layers.append(pool)

        # Bottleneck layer
        if i == num_pools - 1:
            conv = Conv2D(filters[i+1], 3, activation='relu', padding=padding, kernel_initializer='he_normal')(pool_layers[-1])
            for _ in range(num_layers-1):
                conv = Conv2D(filters[i+1], 3, activation='relu', padding=padding, kernel_initializer='he_normal')(conv)
            conv = Dropout(0.5)(conv)

    # Upsampling path
    for i in reversed(range(num_pools)):
        up = UpSampling2D(size=(2,2))(conv_layers[-1]) if i == num_pools - 1 else Conv2D(filters[i], 2, activation='relu', padding=padding, kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv_layers[-1]))
        merge = concatenate([conv_layers[i-1], up], axis=3) if i > 0 else concatenate([inputs, up], axis=3)
        conv = Conv2D(filters[i], 3, activation='relu', padding=padding, kernel_initializer='he_normal')(merge)
        conv_layers.append(Conv2D(filters[i], 3, activation='relu', padding=padding, kernel_initializer='he_normal')(conv))

    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(conv_layers[-1])

    # Define the model
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Load the pretrained weights if provided
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model



